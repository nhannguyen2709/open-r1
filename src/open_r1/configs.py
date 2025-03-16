# coding=utf-8
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

from dataclasses import asdict, dataclass, field
from typing import Optional

from transformers.training_args import TrainingArguments
import trl
from vllm import SamplingParams


@dataclass
class vLLMConfig:
    """
    Args for vLLM rollout.
    """

    dtype: str = field(default="bfloat16", metadata={"help": "The data type to use."})
    enable_chunked_prefill: bool = field(default=False, metadata={"help": "Whether to enable chunked prefill."})
    max_num_batched_tokens: int = field(default=8192, metadata={"help": "The maximum number of tokens to batch."})
    max_model_len: int = field(default=8192, metadata={"help": "The maximum model length."})
    enforce_eager: bool = field(default=False, metadata={"help": "Whether to enforce eager execution."})
    enable_prefix_caching: bool = field(default=False, metadata={"help": "Whether to enable prefix caching."})
    tensor_parallel_size: int = field(default=1, metadata={"help": "The number of tensor parallel size."})
    gpu_memory_utilization: float = field(default=0.6, metadata={"help": "The GPU memory utilization."})
    disable_log_stats: bool = field(default=False, metadata={"help": "Whether to disable log stats."})
    sampling_params_dict: dict = field(
        default_factory=lambda: {}, metadata={"help": "Dictionary of sampling parameters to initialize SamplingParams."}
    )

    @property
    def sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.sampling_params_dict)


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(TrainingArguments):
    """
    args for callbacks, benchmarks etc
    """

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."},
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving " "training speed."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={"help": "Whether to log the completions during training."},
    )
    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(default="main", metadata={"help": "The Hub model branch to push the model to."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig, metadata={"help": "The vLLM configuration."})

    def __post_init__(self):
        super().__post_init__()

        # Convert vllm_config dict to vLLMConfig object with defaults preserved
        if isinstance(self.vllm_config, dict):
            # Create a new vLLMConfig with defaults
            default_config = vLLMConfig()

            # Update with values from the provided dict
            config_dict = {**asdict(default_config), **self.vllm_config}

            # Handle nested sampling_params_dict specially
            if "sampling_params_dict" in self.vllm_config:
                config_dict["sampling_params_dict"] = self.vllm_config["sampling_params_dict"]
                config_dict["sampling_params_dict"]["max_tokens"] = self.max_completion_length
                config_dict["max_model_len"] = self.max_prompt_length + self.max_completion_length
                config_dict["max_num_batched_tokens"] = self.max_prompt_length + self.max_completion_length

            # Create a new vLLMConfig with the merged values
            self.vllm_config = vLLMConfig(**config_dict)


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class ModelConfig(trl.ModelConfig):
    """
    Configuration class for the models.
    """

    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "Storage type to pack the quanitzed 4-bit params."},
    )
