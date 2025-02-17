from functools import partial

import torch
import torch.nn.functional as F


class LigerFusedLinearRLHFBase(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        input_ids,
        attention_mask,
        advantages,
        bias=None,
        loss_fn=None,
        beta=0.1,
        compiled=True,
        ref_inputs=None,
        ref_weight=None,
        ref_bias=None,
    ):
        """Chunked forward pass for RLHF loss computation."""
        # Save for backward
        ctx.beta = beta
        ctx.advantages = advantages

        # Initialize accumulators
        loss_acc = torch.zeros((), device=inputs.device)
        grad_weight = torch.zeros_like(weight)  # [V, H]
        grad_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None  # [V]
        aggregated_metrics = []

        # Create a partial function with fixed arguments
        compute_loss = partial(
            LigerFusedLinearRLHFBase._compute_chunk_loss,
            beta=beta,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            rlhf_loss_fn=loss_fn,
        )

        def fused_fwd_bwd(
            inputs_chunk,
            input_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_inputs_chunk,
        ):
            """Fused forward and backward for a chunk."""
            if bias is not None:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 6), has_aux=True
                )(
                    inputs_chunk,  # arg 0
                    weight,  # arg 1
                    input_ids_chunk,  # arg 2
                    attention_mask_chunk,  # arg 3
                    advantages_chunk,  # arg 4
                    ref_inputs_chunk,  # arg 5
                    bias,  # arg 6
                )
            else:
                return torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 5), has_aux=True
                )(
                    inputs_chunk,  # arg 0
                    weight,  # arg 1
                    input_ids_chunk,  # arg 2
                    attention_mask_chunk,  # arg 3
                    advantages_chunk,  # arg 4
                    ref_inputs_chunk,  # arg 5
                )

        def accumulate_chunk(
            inputs_chunk,
            input_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_inputs_chunk=None,
        ):
            (chunk_grads, (chunk_loss, chunk_metrics)) = fused_fwd_bwd(
                inputs_chunk,
                input_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_inputs_chunk,
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
        chunks = inputs.shape[0]
        input_chunks = torch.chunk(inputs, chunks=chunks, dim=0)
        input_ids_chunks = torch.chunk(input_ids, chunks=chunks, dim=0)
        attention_mask_chunks = torch.chunk(attention_mask, chunks=chunks, dim=0)
        advantages_chunks = torch.chunk(advantages, chunks=chunks, dim=0)
        ref_input_chunks = torch.chunk(ref_inputs, chunks=chunks, dim=0)

        for (
            input_chunk,
            input_ids_chunk,
            attention_mask_chunk,
            advantages_chunk,
            ref_input_chunk,
        ) in zip(
            input_chunks,
            input_ids_chunks,
            attention_mask_chunks,
            advantages_chunks,
            ref_input_chunks,
        ):
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(input_ids_chunk, 1)
            torch._dynamo.mark_dynamic(attention_mask_chunk, 1)
            torch._dynamo.mark_dynamic(advantages_chunk, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1)
            accumulate_chunk(
                input_chunk,
                input_ids_chunk,
                attention_mask_chunk,
                advantages_chunk,
                ref_input_chunk,
            )

        # Scale accumulated loss by number of chunks since we're averaging
        loss_acc = loss_acc / chunks

        # Save for backward
        ctx.save_for_backward(
            torch.cat(grad_inputs, dim=0) / chunks,
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
    def _compute_chunk_loss(
        inputs_chunk,
        weight,
        input_ids_chunk,
        attention_mask_chunk,
        advantages_chunk,
        ref_inputs_chunk=None,
        bias=None,
        beta=0.04,
        ref_weight=None,
        ref_bias=None,
        rlhf_loss_fn=None,
    ):
        """Compute loss for a single chunk."""
        # Get policy log probabilities using chunk_forward
        logits, logits_mean = LigerFusedLinearRLHFBase.chunk_forward(
            inputs_chunk, weight, bias=bias
        )

        # Get reference log probabilities if needed
        with torch.no_grad():
            ref_logits, _ = LigerFusedLinearRLHFBase.chunk_forward(
                ref_inputs_chunk, ref_weight, bias=ref_bias
            )

        # Compute chunk loss and metrics using the provided loss function
        chunk_loss, chunk_metrics = rlhf_loss_fn(
            logits=logits,
            input_ids=input_ids_chunk,
            attention_mask=attention_mask_chunk,
            advantages=advantages_chunk,
            ref_logits=ref_logits,
            beta=beta,
        )

        return chunk_loss, (logits_mean, *chunk_metrics)

    @staticmethod
    def chunk_forward(inputs_chunk, weight, bias=None):
        """Forward pass computation for a single chunk."""
        logits = F.linear(inputs_chunk, weight)
        if bias is not None:
            logits = logits + bias.view(1, -1)

        return logits, logits.mean()

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for RLHF loss."""
        grad_input, grad_weight, grad_bias = ctx.saved_tensors

        return (
            grad_input,
            grad_weight,
            None,  # grad_input_ids
            None,  # grad_attention_mask
            None,  # grad_advantages
            grad_bias,
            None,  # grad_loss_fn
            None,  # grad_beta
            None,  # grad_compiled
            None,  # grad_ref_input
            None,  # grad_ref_weight
            None,  # grad_ref_bias
        )
