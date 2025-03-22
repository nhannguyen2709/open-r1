# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# great reference: https://github.com/vllm-project/vllm/issues/11400
from collections import defaultdict
from dataclasses import dataclass, field
import gc
import logging
import math
from packaging import version
import os
import time
from typing import Callable, Optional, Union

from datasets import Dataset, IterableDataset
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.integrations.deepspeed import (
    deepspeed_load_checkpoint,
    is_deepspeed_zero3_enabled,
)
from transformers.trainer import (
    DEFAULT_CALLBACKS,
    DEFAULT_PROGRESS_CALLBACK,
    TRAINER_STATE_NAME,
    TrainerControl,
    TrainerState,
    _is_peft_model,
)
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
)
from transformers.trainer_utils import TrainOutput
from transformers.trainer_pt_utils import get_model_param_count
from transformers.utils import is_accelerate_available, is_liger_kernel_available, is_peft_available
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)

from trl.extras.profiling import profiling_decorator
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)

if is_accelerate_available():
    from accelerate.utils import DistributedType, is_peft_model, set_seed
    from accelerate.utils.other import is_compiled_module

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.configs import GRPOConfig
from open_r1.grpo_loss import LigerFusedLinearGRPOLoss
from open_r1.performance import log_gpu_memory_usage
from open_r1.vllm_rollout import vLLMRollout, VLLMShardingManager


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
logger = logging.getLogger(__name__)


def exact_div(a: int, b: int, custom_error_message: str = "") -> int:
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


class FastGRPOTrainer(Trainer):
    _tag_names = ["trl", "fast_grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.args = args
        self.reward_funcs = reward_funcs
        # Reward weights (move this logic to post_init of config?)
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward " f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = args.reward_weights
        else:
            self.reward_weights = ([1.0] * len(reward_funcs),)

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        self.processing_class = processing_class

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(reward_func, num_labels=1, **model_init_kwargs)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward " f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.data_collator = data_collator

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        # Multi-step
        self.beta = args.beta
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0

        self.local_dataloader_batch_size = exact_div(
            args.per_device_train_batch_size * args.gradient_accumulation_steps,
            args.num_generations,
            "per_device_train_batch_size * gradient_accumulation_steps must >= num_generations to remain on policy",
        )
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47
        self.create_accelerator_and_postprocess()

        set_seed(args.seed, device_specific=True)
        self.train_dataset = train_dataset
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        self.train_dataset_len = len(self.train_dataset)
        num_total_samples = int(self.args.num_train_epochs * self.train_dataset_len)
        self.total_steps_per_device = num_total_samples // (self.local_dataloader_batch_size * self.accelerator.num_processes)

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.current_flos = 0
        self.hp_search_backend = None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # # Build actor model + optimizer, reference model
        # if is_deepspeed_zero3_enabled() and peft_config is not None:
        #     raise ValueError(
        #         "PEFT (Parameter-Efficient Fine-Tuning) is not supported with DeepSpeed ZeRO-3. "
        #         "Please disable DeepSpeed ZeRO-3 or use a different training configuration without PEFT."
        #     )
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.model = model

        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance

                if isinstance(self.model, PreTrainedModel):
                    _apply_liger_kernel_to_instance(model=self.model)
                elif isinstance(self.model, PeftModel):
                    _apply_liger_kernel_to_instance(model=self.model.base_model.model)
                else:
                    logger.warning("The model is not an instance of PreTrainedModel. No liger kernels will be applied.")

                if self.ref_model is not None:
                    _apply_liger_kernel_to_instance(model=self.ref_model)
            else:
                raise ImportError(
                    "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with `pip install liger-kernel`"
                )
        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Accelerator prepare
        self.create_optimizer_and_scheduler(num_training_steps=self.total_steps_per_device)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Build vllm rollout
        infer_tp = self.args.vllm_config.tensor_parallel_size
        dp = self.accelerator.num_processes // infer_tp
        assert (
            self.accelerator.num_processes % infer_tp == 0
        ), f"rollout world_size: {self.accelerator.num_processes} is not divisible by infer_tp: {infer_tp}"
        if dp > 1:
            rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])
        else:
            rollout_device_mesh = None

        log_gpu_memory_usage("Before building vllm rollout", logger=logger)
        self.rollout = vLLMRollout(model_id, self.args.vllm_config, self.processing_class)
        logger.info(f"Sampling params: {self.args.vllm_config.sampling_params}")
        log_gpu_memory_usage("After building vllm rollout", logger=logger)
        self.rollout_sharding_manager = VLLMShardingManager(
            self.model, self.rollout.inference_engine, self.accelerator, model.config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage("After building vllm sharding manager", logger=logger)
        self._last_loaded_step = 0
        self.accelerator.wait_for_everyone()

        self.log_completions = args.log_completions

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    @torch.no_grad()
    def prepare_batch(self, batch):
        """
        This will:
        - generate k samples for each problem
        - compute ref logprobs for each generation
        - using internal reward model(s) to get rewards
        """
        prompts = [x["prompt"] for x in batch]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in batch]
        prompt_inputs = self.processing_class(prompts_text, add_special_tokens=False)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        all_prompts_text = []
        for prompt in prompts_text:
            all_prompts_text.extend([prompt] * self.args.num_generations)

        load_weights = self.state.global_step != self._last_loaded_step
        self.rollout_sharding_manager.load_weights = load_weights
        start = time.time()
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            all_prompts_text = self.rollout_sharding_manager.preprocess_data(all_prompts_text)
            completion_ids = self.rollout.generate_sequences(all_prompts_text)

            log_gpu_memory_usage("After rollout generation", logger=logger)
            logger.info(f"Rollout generation time: {time.time() - start:.2f}s")

            completion_ids = self.rollout_sharding_manager.postprocess_data(completion_ids)
        self._last_loaded_step = self.state.global_step

        # Decode the generated completions
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.args.num_generations)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(repeated_prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards = torch.zeros(len(repeated_prompts), len(self.reward_funcs))
        for (
            i,
            reward_func,
        ) in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in batch[0] if key not in ["prompt", "completion"]]
            reward_kwargs = defaultdict(list)
            for example in batch:
                for key in keys:
                    reward_kwargs[key].extend([example[key]] * self.args.num_generations)
            output_reward_func = reward_func(prompts=repeated_prompts, completions=completions, **reward_kwargs)
            rewards[:, i] = torch.tensor(output_reward_func, dtype=torch.float32) * self.reward_weights[i]

        # calculate the advantages, the prompt is all on the same device to no need to gather here
        grouped_rewards = rewards.sum(dim=1).view(len(prompts), self.args.num_generations)
        # grouped_advantages = (grouped_rewards - grouped_rewards.mean(-1, keepdim=True)) / (grouped_rewards.std(-1, keepdim=True) + EPS)
        grouped_advantages = grouped_rewards - grouped_rewards.mean(-1, keepdim=True)
        advantages = grouped_advantages.flatten().tolist()

        torch.distributed.barrier()
        # build batch as list of dicts
        examples = []
        for i, prompt in enumerate(repeated_prompts):
            example = {
                "prompt": prompt,
                "prompt_ids": prompt_ids[i // self.args.num_generations],
                "prompt_mask": prompt_mask[i // self.args.num_generations],
                "completion": completions_text[i],
                "completion_ids": completion_ids[i],
                "advantages": advantages[i],
                "rewards": rewards[i],
            }
            examples.append(example)
        # sort examples by length of prompt_ids and completion_ids
        examples.sort(key=lambda x: len(x["prompt_ids"]) + len(x["completion_ids"]))

        return examples

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ):
        self.callback_handler = CallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)],
        )

        if self.args.logging_steps is not None:
            if self.args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * self.args.logging_steps)
            else:
                self.state.logging_steps = self.args.logging_steps

        if self.args.save_steps is not None:
            if self.args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * self.args.save_steps)
            else:
                self.state.save_steps = self.args.save_steps

        self.state.max_steps = self.total_steps_per_device
        self.state.num_train_epochs = self.args.num_train_epochs

        self.model.train()
        self.deepspeed = self.model_wrapped = self.model

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model,
                    resume_from_checkpoint,
                    load_module_strict=not _is_peft_model(self.model),
                )
            else:
                self._load_from_checkpoint(resume_from_checkpoint, self.model)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {self.train_dataset_len:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Number of GRPO iterations = {self.num_iterations}")
        logger.info(f"  Total optimization steps = {self.total_steps_per_device:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        # Set up training state for resuming
        start_step = 1
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Load training state if available
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            start_step = self.state.global_step
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // self.total_steps_per_device)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first" f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        def repeat_generator():
            while True:
                yield from self.dataloader

        iter_dataloader = iter(repeat_generator())

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        # Training loss setup
        num_updates = 0
        device = self.accelerator.device
        tr_loss = torch.tensor(0.0).to(device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = start_step
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in range(start_step, self.total_steps_per_device + 1):
            batch = next(iter_dataloader)
            batch = self.prepare_batch(batch)
            gen_dataset = Dataset.from_list(batch)

            iteration_losses = []
            iteration_grad_norms = []
            # store the per-token logps for each mini-batch
            self._buffer = [None] * (len(gen_dataset) // self.args.per_device_train_batch_size)
            for iteration in range(self.num_iterations):
                mini_batch_dataloader = DataLoader(
                    gen_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    shuffle=False,
                    drop_last=True,
                    collate_fn=lambda x: mini_batch_collator(x, self.processing_class, self.args.max_prompt_length),
                )
                for idx, mini_batch in enumerate(mini_batch_dataloader):
                    loss = self._optimization_step(mini_batch, idx, iteration + 1)
                    iteration_losses.append(loss)
                # Add proper gradient clipping
                _grad_norm = None
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    _grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm,
                    )
                if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    grad_norm = self.model.get_global_grad_norm()
                    # In some cases the grad norm may not return a float
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                else:
                    grad_norm = _grad_norm.item()
                iteration_grad_norms.append(grad_norm)
                num_updates += 1

            # TODO: Maybe use _maybe_log_save_evaluate
            self.lr_scheduler.step()
            self.state.global_step = step
            self.state.epoch = step / self.total_steps_per_device
            tr_loss_step = sum(iteration_losses) / len(iteration_losses)
            tr_loss = tr_loss + tr_loss_step
            self._total_loss_scalar += tr_loss_step.item()
            metrics = {
                "loss": tr_loss_step.item(),
                "grad_norm": sum(iteration_grad_norms) / len(iteration_grad_norms),
                "learning_rate": self.lr_scheduler.get_last_lr()[0],
                "epoch": self.state.epoch,
                "step": self.state.global_step,
                "num_updates": num_updates,
            }
            self.log(metrics, start_time)

            self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        return TrainOutput(
            self.state.global_step,
            tr_loss.item() / self.state.global_step if self.state.global_step > 0 else 0.0,
            {k: sum(v) / len(v) if v else 0.0 for k, v in self._metrics["train"].items()},
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _optimization_step(self, mini_batch: dict[str, torch.Tensor | list[str]], idx: int, iteration: int):
        device = self.accelerator.device
        prompts = mini_batch.pop("prompts")
        mini_batch = {k: v.to(device) for k, v in mini_batch.items()}
        prompt_ids = mini_batch["prompt_ids"]
        prompt_mask = mini_batch["prompt_mask"]
        completion_ids = mini_batch["completion_ids"]
        completion_mask = mini_batch["completion_mask"]
        advantages = mini_batch["advantages"].unsqueeze(1)
        rewards = mini_batch["rewards"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                    )
        with self.accelerator.accumulate(self.model):
            per_token_logps = self._get_per_token_logps(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
            )
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            old_per_token_logps = self._buffer[idx]
            if old_per_token_logps is None:
                old_per_token_logps = per_token_logps.detach()
            self._buffer[idx] = per_token_logps.detach()
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = torch.min(per_token_loss1, per_token_loss2)
            per_token_loss = -(per_token_loss - self.args.beta * per_token_kl)
            # loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self.accelerator.backward(loss)
            # Run callbacks and optimizer steps
            self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
            self.optimizer.step()
            self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)
            self.optimizer.zero_grad()

        # Log the metrics
        with torch.no_grad():
            completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            min_completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().min().item()
            max_completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().max().item()
            self._metrics["train"]["completion_length"].append(completion_length)
            self._metrics["train"]["min_completion_length"].append(min_completion_length)
            self._metrics["train"]["max_completion_length"].append(max_completion_length)
            rewards = self.accelerator.gather_for_metrics(rewards)
            rewards_per_func = rewards.mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, torch.nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics["train"][f"rewards/{reward_func_name}"].append(rewards_per_func[i].item())
            rewards = rewards.sum(dim=1)
            self._metrics["train"]["rewards"].append(rewards.mean().item())
            self._metrics["train"]["rewards_std"].append(rewards.std().item())
            self._metrics["train"]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
            self._metrics["train"]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss.detach()

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()


def mini_batch_collator(examples, processing_class, max_prompt_length):
    pad_token_id = processing_class.pad_token_id
    prompt_ids = []
    prompt_mask = []
    completion_ids = []
    advantages = []
    rewards = []
    prompts = []
    for example in examples:
        prompt_ids.append(torch.tensor(example["prompt_ids"]))
        prompt_mask.append(torch.tensor(example["prompt_mask"]))
        completion_ids.append(torch.tensor(example["completion_ids"]))
        advantages.append(example["advantages"])
        rewards.append(example["rewards"])
        prompts.append(example["prompt"])
    prompt_ids = pad(prompt_ids, pad_token_id, "left")
    prompt_mask = pad(prompt_mask, 0, "left")
    if max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -max_prompt_length:]
        prompt_mask = prompt_mask[:, -max_prompt_length:]
    completion_ids = pad(completion_ids, pad_token_id, "right")

    is_eos = completion_ids == processing_class.eos_token_id
    if is_eos.any(dim=1).any().item():
        # Mask everything after the first EOS token
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    else:
        completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": torch.tensor(advantages),
        "rewards": torch.tensor(rewards),
        "prompts": prompts,
    }
