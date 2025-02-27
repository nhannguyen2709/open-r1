from typing import Optional

import torch
from torch.nn import functional as F
from torch import Tensor
from trl.trainer.utils import selective_log_softmax

from functools import partial

import torch
import torch.nn.functional as F


class LigerFusedLinearGRPOLossFunction(torch.autograd.Function):
    @staticmethod
    def rlhf_loss_fn(
        _input: Tensor,
        weight: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_input: Tensor,
        ref_weight: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        ref_bias: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
    ):
        """GRPO loss function."""
        # Get policy model log probabilities
        logits = F.linear(_input, weight, bias)

        # Get reference model log probabilities
        with torch.no_grad():
            ref_logits = F.linear(ref_input, ref_weight, ref_bias)

        # Compute chunk loss and metrics using the provided loss function
        per_token_logps = selective_log_softmax(logits, input_id)
        ref_per_token_logps = selective_log_softmax(ref_logits, input_id)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)

        per_token_loss = -(per_token_loss - beta * per_token_kl)
        loss = (per_token_loss * attention_mask).sum() / attention_mask.sum()

        # Calculate metrics
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * attention_mask).sum() / attention_mask.sum()
        metrics = (
            per_token_logps.mean(),  # mean log prob
            per_token_logps.std(),  # std log prob
            F.log_softmax(logits, dim=-1).mean(),  # mean all log probs
            (per_token_kl * attention_mask).sum() / attention_mask.sum(),  # mean KL div
            clip_ratio,  # clip ratio
            per_token_logps,  # log prob
        )

        return loss, metrics

    @staticmethod
    def forward(
        ctx,
        _input: Tensor,
        weight: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_input: Tensor,
        ref_weight: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        ref_bias: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
        compiled: bool = True,
    ):
        """Chunked forward pass for RLHF loss computation."""

        # Initialize accumulators
        loss_acc = torch.zeros((), device=_input.device)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = torch.zeros_like(_input)  # [B, H]
        grad_bias = torch.zeros_like(bias) if bias is not None else None  # [V]
        aggregated_metrics = []

        # Create a partial function with fixed arguments
        compute_loss = partial(
            LigerFusedLinearGRPOLossFunction.rlhf_loss_fn,
            beta=beta,
            epsilon=epsilon,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
        )

        def fused_fwd_bwd(
            input_chunk,
            input_id_chunk,
            attention_mask_chunk,
            advantage_chunk,
            ref_input_chunk,
            old_per_token_logps_chunk=None,
        ):
            """Fused forward and backward for a chunk."""
            if bias is not None:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 6), has_aux=True
                )(
                    input_chunk,
                    weight,
                    input_id_chunk,
                    attention_mask_chunk,
                    advantage_chunk,
                    ref_input_chunk,
                    bias=bias,
                    old_per_token_logps=old_per_token_logps_chunk,
                )
            else:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1), has_aux=True
                )(
                    input_chunk,
                    weight,
                    input_id_chunk,
                    attention_mask_chunk,
                    advantage_chunk,
                    ref_input_chunk,
                    old_per_token_logps=old_per_token_logps_chunk,
                )

        def accumulate_chunk(
            inputs_chunk,
            input_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_inputs_chunk,
            old_per_token_logps_chunk=None,
            chunk_idx: int = 0,
        ):
            (chunk_grads, (chunk_loss, chunk_metrics)) = fused_fwd_bwd(
                inputs_chunk,
                input_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_inputs_chunk,
                old_per_token_logps_chunk,
            )
            chunk_grad_input = chunk_grads[0]
            chunk_grad_weight = chunk_grads[1]

            # Accumulate gradients and loss
            grad_weight.add_(chunk_grad_weight)
            grad_inputs[
                chunk_idx
                * inputs_chunk.size(0) : (chunk_idx + 1)
                * inputs_chunk.size(0)
            ].copy_(chunk_grad_input)
            loss_acc.add_(chunk_loss)
            if bias is not None:
                chunk_grad_bias = chunk_grads[2]
                grad_bias.add_(chunk_grad_bias)

            # Initialize storage for metrics on first chunk
            if len(aggregated_metrics) == 0:
                for metric in chunk_metrics:
                    if metric.ndim == 0:
                        aggregated_metrics.append(torch.zeros((), device=metric.device))
                    else:
                        aggregated_metrics.append([])

            # Accumulate metrics
            for i, metric in enumerate(chunk_metrics):
                if metric.ndim == 0:
                    aggregated_metrics[i].add_(metric)
                else:
                    aggregated_metrics[i].append(metric)

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        # Process input in chunks
        chunks = _input.shape[0]
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        input_id_chunks = torch.chunk(input_id, chunks=chunks, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        advantage_chunks = torch.chunk(advantage, chunks=chunks, dim=0)
        ref_input_chunks = torch.chunk(ref_input, chunks=chunks, dim=0)
        if old_per_token_logps is not None:
            old_per_token_logps_chunks = torch.chunk(
                old_per_token_logps, chunks=chunks, dim=0
            )
        else:
            old_per_token_logps_chunks = [None] * chunks

        for chunk_idx, (
            input_chunk,
            input_id_chunk,
            attention_mask_chunk,
            advantage_chunk,
            ref_input_chunk,
            old_per_token_logps_chunk,
        ) in enumerate(
            zip(
                input_chunks,
                input_id_chunks,
                attention_mask_chunks,
                advantage_chunks,
                ref_input_chunks,
                old_per_token_logps_chunks,
            )
        ):
            accumulate_chunk(
                input_chunk,
                input_id_chunk,
                attention_mask_chunk,
                advantage_chunk,
                ref_input_chunk,
                old_per_token_logps_chunk,
                chunk_idx,
            )

        # Scale accumulated loss by number of chunks since we're averaging
        loss_acc = loss_acc / chunks

        # Save for backward
        ctx.save_for_backward(
            grad_inputs / chunks,
            grad_weight / chunks,
            grad_bias / chunks if bias is not None else None,
        )

        # Finalize metrics
        final_metrics = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):
                final_metrics.append(torch.cat(metric, dim=0))
            else:
                final_metrics.append(metric / chunks)

        return loss_acc, tuple(final_metrics)

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for RLHF loss."""
        grad_input, grad_weight, grad_bias = ctx.saved_tensors

        return (
            grad_input,
            grad_weight,
            None,  # grad_input_id
            None,  # grad_attention_mask
            None,  # grad_advantage
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_old_per_token_logps
            grad_bias,
            None,  # grad_ref_bias
            None,  # grad_beta
            None,  # grad_epsilon
            None,  # grad_compiled
        )


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.04,
        epsilon: float = 0.2,
        compiled: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.compiled = compiled

    def forward(
        self,
        _input: Tensor,
        weight: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_input: Tensor,
        ref_weight: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        ref_bias: Optional[Tensor] = None,
    ):
        return LigerFusedLinearGRPOLossFunction.apply(
            _input,
            weight,
            input_id,
            attention_mask,
            advantage,
            ref_input,
            ref_weight,
            old_per_token_logps,
            bias,
            ref_bias,
            self.beta,
            self.epsilon,
            self.compiled,
        )
