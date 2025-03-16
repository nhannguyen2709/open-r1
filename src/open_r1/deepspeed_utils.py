from deepspeed.runtime.engine import DeepSpeedEngine
import torch
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


@torch.no_grad()
def offload_deepspeed_model_to_cpu(model: DeepSpeedEngine):
    if is_deepspeed_zero3_enabled():
        model.empty_partition_cache()
    else:
        for param in model.parameters():
            param.data.to("cpu", non_blocking=True)
            

@torch.no_grad()
def load_deepspeed_model_to_gpu(model: DeepSpeedEngine):
    if is_deepspeed_zero3_enabled():
        model.reload_states(non_blocking=True)
    else:
        device_id = torch.cuda.current_device()
        for param in model.parameters():
            param.data.to(torch.device(f"cuda:{device_id}"), non_blocking=True)


@torch.no_grad()
def offload_deepspeed_optimizer(optimizer: torch.optim.Optimizer):
    if is_deepspeed_zero3_enabled():
        # Set offload_optimizer_device: "cpu" in deepspeed_config for offloading in zero3
        # optimizer.offload_states() is supported only for DeepSpeed FusedAdam.
        return
    
    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_deepspeed_optimizer(optimizer: torch.optim.Optimizer, device_id: torch.device):
    if is_deepspeed_zero3_enabled():
        return

    if not optimizer.state:
        return
    
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
