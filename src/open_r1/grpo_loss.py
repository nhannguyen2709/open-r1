import torch
from torch.nn import functional as F
from trl.trainer.utils import selective_log_softmax

from open_r1.fused_linear_rlhf import LigerFusedLinearRLHFBase


class LigerFusedLinearGRPOFunction(LigerFusedLinearRLHFBase):
    @staticmethod
    def rlhf_loss_fn(
        logits,  # B*T, V
        input_ids,  # B*T
        attention_mask,  # B*T
        advantages,  # B*T
        ref_logits,  # B*T, V
        beta=0.1,
        **kwargs,
    ):
        """
        Compute GRPO loss on chunked inputs.
        """
        per_token_logps = selective_log_softmax(logits, input_ids)
        ref_per_token_logps = selective_log_softmax(ref_logits, input_ids)

        # Compute the KL divergence between the model and the reference model
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = (
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        )
        per_token_loss = -(per_token_loss - beta * per_token_kl)
        loss = (per_token_loss * attention_mask).sum() / (attention_mask.sum() + 1e-8)

        # Calculate metrics
        metrics = (
            per_token_logps.mean(),  # mean log prob
            per_token_logps.std(),  # std log prob
            F.log_softmax(logits, dim=-1).mean(),  # mean all log probs
            (per_token_kl * attention_mask).sum() / attention_mask.sum(),  # mean KL div
        )

        return loss, metrics

    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        input_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_inputs=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.1,
        compiled=True,
    ):
        return LigerFusedLinearRLHFBase.forward(
            ctx=ctx,
            inputs=inputs,
            weight=weight,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_fn=LigerFusedLinearGRPOFunction.rlhf_loss_fn,
            advantages=advantages,
            bias=bias,
            ref_inputs=ref_inputs,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            compiled=compiled,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        """Backward pass for GRPO loss.

        Args:
            grad_output: Gradient of the loss (scalar)
            grad_metrics: Gradients of the metrics (not used in backward computation)
        """
        return LigerFusedLinearRLHFBase.backward(ctx, grad_output)


class LigerFusedLinearGRPOLoss(torch.nn.Module):
    """Fused linear layer with GRPO loss."""

    def __init__(
        self,
        beta: float = 0.1,
        compiled: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.compiled = compiled

    def forward(
        self,
        weight,
        inputs,
        input_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_inputs=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearGRPOFunction.apply(
            inputs,
            weight,
            input_ids,
            attention_mask,
            advantages,
            bias,
            ref_inputs,
            ref_weight,
            ref_bias,
            self.beta,
            self.compiled,
        )
