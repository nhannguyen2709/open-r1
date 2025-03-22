from typing import Optional, List, Union, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from open_r1.grpo_loss import LigerFusedLinearGRPOLoss


def lce_forward(
    self: PreTrainedModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    advantages: Optional[torch.Tensor] = None,
    completion_size: int = 0,
    ref_per_token_logps: Optional[torch.Tensor] = None,
    old_per_token_logps: Optional[torch.Tensor] = None,
    sft_input_ids: Optional[torch.Tensor] = None,
    sft_attention_mask: Optional[torch.Tensor] = None,
    sft_completion_size: int = 0,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    import torch.distributed as dist

    dist.breakpoint()

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )[0]
    hidden_states = hidden_states[:, :-1, :]

    hidden_states = hidden_states[:, -completion_size:]
    loss, metrics = LigerFusedLinearGRPOLoss()(
        hidden_states, self.lm_head.weight, input_ids, attention_mask, advantages, ref_per_token_logps, old_per_token_logps
    )
