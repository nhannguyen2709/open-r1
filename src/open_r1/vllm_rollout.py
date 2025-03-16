import logging

from accelerate import Accelerator
from accelerate.utils import is_peft_model
from accelerate.utils.other import is_compiled_module
import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.models import unwrap_model_for_generation

from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from open_r1.configs import vLLMConfig
from open_r1.performance import log_gpu_memory_usage
from open_r1.deepspeed_utils import (
    offload_deepspeed_model_to_cpu,
    offload_deepspeed_optimizer,
    load_deepspeed_model_to_gpu,
    load_deepspeed_optimizer,
)

logger = logging.getLogger(__name__)


def all_gather_data(data, process_group):
    group_size = torch.distributed.get_world_size(process_group)
    all_data = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_data, data, group=process_group)
    return [x for y in all_data for x in y]


class vLLMRollout:
    def __init__(self, model: str, config: vLLMConfig, tokenizer):
        super().__init__()
        self.config = config

        tensor_parallel_size = config.tensor_parallel_size
        if torch.distributed.is_initialized():
            assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = config.max_num_batched_tokens

        max_model_len = config.max_model_len

        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        self.inference_engine = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            enable_sleep_mode=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=max_model_len,
            # load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            distributed_executor_backend="external_launcher" if torch.distributed.is_initialized() else None,
        )
        # # Offload vllm model to reduce peak memory usage
        # self.inference_engine.sleep(level=1)

        self.sampling_params = config.sampling_params
        self.pad_token_id = tokenizer.pad_token_id

    @torch.no_grad()
    def generate_sequences(self, data: list[str], **kwargs) -> list[list[int]]:
        outputs = self.inference_engine.generate(data, sampling_params=self.sampling_params, use_tqdm=False)

        completion_ids = []
        for output in outputs:
            completion_ids.append(output.outputs[0].token_ids)
        return completion_ids


class VLLMShardingManager:
    def __init__(
        self,
        module: DeepSpeedEngine,
        inference_engine: LLM,
        accelerator: Accelerator,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        load_weights: bool = True,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.accelerator = accelerator
        self.load_weights = load_weights

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):

        state_dict = {}
        if self.load_weights:
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            with unwrap_model_for_generation(
                self.module,
                self.accelerator,
                gather_deepspeed3_params=is_deepspeed_zero3_enabled(),
            ) as unwrapped_model:
                if is_compiled_module(unwrapped_model):
                    unwrapped_model = unwrapped_model._orig_mod
                if is_peft_model(unwrapped_model):
                    unwrapped_model.merge_adapter()
                    state_dict = unwrapped_model.state_dict()
                    # Remove base_model and base_layer prefixes
                    state_dict = {k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()}
                    # Remove values with adapter prefix (example: "_lora")
                    state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {k.replace("modules_to_save.default.", ""): v for k, v in state_dict.items() if "original_module" not in k}
                else:
                    state_dict = unwrapped_model.state_dict()
                # Unmerge the adapter to restore the model to its original state.
                # This must be done after loading weights to ensure they correspond to the merged state.
                if is_peft_model(unwrapped_model):
                    unwrapped_model.unmerge_adapter()
                state_dict = {k: v.cpu() for k, v in state_dict.items()}

            torch.cuda.empty_cache()

            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

        self.inference_engine.wake_up()
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

        if self.load_weights:
            loaded_params = []
            device_id = torch.cuda.current_device()
            for name, param in state_dict.items():
                param = param.to(torch.device(f"cuda:{device_id}"), non_blocking=True)
                loaded_params.append(model.load_weights([(name, param)]))
            logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")

        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
        del state_dict
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        offload_deepspeed_model_to_cpu(self.module)
        offload_deepspeed_optimizer(self.module.optimizer)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After offload model weights in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_val, traceback):
        log_gpu_memory_usage("Before vllm offload in sharding manager", logger=logger)
        self.inference_engine.sleep(level=1)
        log_gpu_memory_usage("After vllm offload in sharding manager", logger=logger)

        load_deepspeed_model_to_gpu(self.module)
        load_deepspeed_optimizer(self.module.optimizer, torch.cuda.current_device())

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: list[str]):
        if self.tp_size == 1:
            return data

        group = vllm_ps.get_tensor_model_parallel_group().device_group
        data = all_gather_data(data, group)
        return data

    def postprocess_data(self, data: list[str]):
        if self.tp_size == 1:
            return data

        chunk_size = len(data) // self.tp_size
        return data[self.tp_rank * chunk_size : (self.tp_rank + 1) * chunk_size]
