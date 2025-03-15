from abc import abstractmethod
from functools import partial
from typing import Optional

import torch
from torch import Tensor
from trl.trainer.utils import selective_log_softmax


class LigerFusedLinearRLHFBase(torch.autograd.Function):
    @abstractmethod
    def rlhf_loss_fn(*args, **kwargs):
        """
        To be extended by subclasses.
        """
        raise NotImplementedError("RLHF loss function must be implemented.")

    @staticmethod
    def forward(
        cls,
        ctx,
        _input: Tensor,
        weight: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_per_token_logps: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
        compiled: bool = True,
        chunk_size: int = 1024,
    ):
        """Chunked forward pass for RLHF loss computation."""

        # Initialize accumulators
        loss_acc = torch.zeros((), device=_input.device)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None  # [V]
        aggregated_metrics = []

        # Create a partial function with fixed arguments
        compute_loss = partial(
            LigerFusedLinearRLHFBase._compute_chunk_loss,
            beta=beta,
            epsilon=epsilon,
            rlhf_loss_fn=cls.rlhf_loss_fn,
        )

        def fused_fwd_bwd(
            input_chunk,
            input_id_chunk,
            attention_mask_chunk,
            advantage_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk=None,
        ):
            """Fused forward and backward for a chunk."""
            if bias is not None:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1, 7), has_aux=True)(
                    input_chunk,
                    weight,
                    input_id_chunk,
                    attention_mask_chunk,
                    advantage_chunk,
                    ref_per_token_logps_chunk,
                    old_per_token_logps_chunk,
                    bias,
                )
            else:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
                    input_chunk,
                    weight,
                    input_id_chunk,
                    attention_mask_chunk,
                    advantage_chunk,
                    ref_per_token_logps_chunk,
                    old_per_token_logps_chunk,
                )

        def accumulate_chunk(
            inputs_chunk,
            input_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_per_token_logps_chunk,
            old_per_token_logps_chunk=None,
        ):
            (chunk_grads, (chunk_loss, chunk_metrics)) = fused_fwd_bwd(
                inputs_chunk,
                input_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
            )
            chunk_grad_input = chunk_grads[0]
            chunk_grad_weight = chunk_grads[1]

            # Accumulate gradients and loss
            grad_weight.add_(chunk_grad_weight)
            grad_inputs.append(chunk_grad_input)
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
        chunks = max(1, _input.shape[0] // chunk_size)
        input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
        input_id_chunks = torch.chunk(input_id, chunks=chunks, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        advantage_chunks = torch.chunk(advantage, chunks=chunks, dim=0)
        ref_per_token_logps_chunks = torch.chunk(ref_per_token_logps, chunks=chunks, dim=0)
        if old_per_token_logps is not None:
            old_per_token_logps_chunks = torch.chunk(old_per_token_logps, chunks=chunks, dim=0)
        else:
            old_per_token_logps_chunks = [None] * chunks

        for input_chunk, input_id_chunk, attention_mask_chunk, advantage_chunk, ref_per_token_logps_chunk, old_per_token_logps_chunk in zip(
            input_chunks,
            input_id_chunks,
            attention_mask_chunks,
            advantage_chunks,
            ref_per_token_logps_chunks,
            old_per_token_logps_chunks,
        ):
            accumulate_chunk(
                input_chunk,
                input_id_chunk,
                attention_mask_chunk,
                advantage_chunk,
                ref_per_token_logps_chunk,
                old_per_token_logps_chunk,
            )

        # Scale accumulated loss by number of chunks since we're averaging
        loss_acc = loss_acc / chunks

        # Save for backward
        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0),
            grad_weight,
            grad_bias if bias is not None else None,
        )

        # Finalize metrics
        final_metrics = []
        for metric in aggregated_metrics:
            if isinstance(metric, list):
                final_metrics.append(torch.cat(metric, dim=0))
            else:
                final_metrics.append(metric / chunks)

        per_token_logps = final_metrics[0]
        mean_kl = (final_metrics[1] * attention_mask).sum() / attention_mask.sum()
        clip_ratio = (final_metrics[2] * attention_mask).sum() / attention_mask.sum()

        return loss_acc, (per_token_logps, mean_kl, clip_ratio)

    @staticmethod
    def chunk_forward(input_chunk, weight, bias=None):
        """Forward pass computation for a single chunk without explicit reshaping."""
        # Directly compute logits via batched matrix multiplication: [B, T, H] @ [H, V] -> [B, T, V]
        logits = torch.matmul(input_chunk, weight.t())
        if bias is not None:
            logits = logits + bias  # Broadcasts bias to [B, T, V])
        return logits

    @staticmethod
    def _compute_chunk_loss(
        input_chunk,
        weight,
        input_id_chunk,
        attention_mask_chunk,
        advantage_chunk,
        ref_per_token_logps_chunk,
        old_per_token_logps_chunk,
        bias: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
        rlhf_loss_fn=None,
    ):
        logits = LigerFusedLinearRLHFBase.chunk_forward(input_chunk, weight, bias)
        chunk_loss, chunk_metrics = rlhf_loss_fn(
            logits, input_id_chunk, attention_mask_chunk, advantage_chunk, ref_per_token_logps_chunk, old_per_token_logps_chunk, beta, epsilon
        )
        return chunk_loss, chunk_metrics

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for RLHF loss."""
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        if grad_output != 1.0:
            grad_input = grad_input * grad_output
            grad_weight = grad_weight * grad_output
            if grad_bias is not None:
                grad_bias = grad_bias * grad_output

        return (
            grad_input,
            grad_weight,
            None,  # grad_input_id
            None,  # grad_attention_mask
            None,  # grad_advantage
            None,  # grad_ref_per_token_logps
            None,  # grad_old_per_token_logps
            grad_bias,
            None,  # grad_beta
            None,  # grad_epsilon
            None,  # grad_compiled
            None,  # grad_chunk_size
        )


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        logits: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_per_token_logps: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
    ):
        """GRPO loss function."""
        # Get policy model log probabilities
        per_token_logps = selective_log_softmax(logits, input_id)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # Compute policy gradient loss with importance sampling ratio
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)

        # Combine losses
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        loss = (per_token_loss * attention_mask).sum() / attention_mask.sum()

        # Calculate metrics
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        metrics = (
            per_token_logps,  # log prob
            per_token_kl,  # KL div
            is_clipped,  # clip ratio
        )
        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input: Tensor,
        weight: Tensor,
        input_id: Tensor,
        attention_mask: Tensor,
        advantage: Tensor,
        ref_per_token_logps: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        beta: float = 0.04,
        epsilon: float = 0.2,
        compiled: bool = True,
        chunk_size: int = 1024,
    ):
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            input_id=input_id,
            attention_mask=attention_mask,
            advantage=advantage,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            bias=bias,
            beta=beta,
            epsilon=epsilon,
            compiled=compiled,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        grads = LigerFusedLinearRLHFBase.backward(ctx, grad_output, *grad_metrics)
        return grads


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
        ref_per_token_logps: Tensor,
        old_per_token_logps: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        chunk_size: int = 1024,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            _input,
            weight,
            input_id,
            attention_mask,
            advantage,
            ref_per_token_logps,
            old_per_token_logps,
            bias,
            self.beta,
            self.epsilon,
            self.compiled,
            chunk_size,
        )
