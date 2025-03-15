from typing import Any, Callable

import torch
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


@torch.no_grad()
def offload_deepspeed_model_to_cpu(model: DeepSpeedEngine, empty_cache: bool = True):
    if is_deepspeed_zero3_enabled():
        model.empty_partition_cache()
    else:
        for param in model.parameters():
            param.data.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


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
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)


class _DeepSpeedForwardRedirection:
    """
    A utility class that redirects method calls through DeepSpeedEngine.forward.

    This redirection ensures that DeepSpeedEngine's pre-forward and post-forward hooks
    are properly executed around the method call. The class is essential when working
    with submodules of DeepSpeedEngine-wrapped models.

    Use cases:
    1. When accessing only part of a model (e.g., calling only the `LlamaModel` part
       of a DeepSpeedEngine-wrapped `LlamaForCausalLM`) to get hidden states without
       involving GPU-memory-heavy components like `lm_head`
    2. When direct calls (e.g., `model.model.forward()`) would bypass parameter
       all-gathering for the first layers in transformer-based wrapping policies

    Without this redirection, direct calls to submodules may fail with errors like
    "RuntimeError: 'weight' must be 2-D" because parameters aren't properly all-gathered.

    Adapted from PyTorch Lightning's strategy implementation:
    https://github.com/Lightning-AI/pytorch-lightning/blob/d3f9c83d6efa4f1def36aa6c199600946cdb9117/src/lightning/pytorch/strategies/strategy.py#L601-L648
    """

    def __call__(
        self,
        wrapper_module: DeepSpeedEngine,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.
        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
        """
        assert isinstance(wrapper_module, DeepSpeedEngine)
        original_module = wrapper_module.module
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]
        wrapper_output = wrapper_module(*args, **kwargs)
        return wrapper_output
