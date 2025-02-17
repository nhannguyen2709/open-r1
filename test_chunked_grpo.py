import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from open_r1.grpo_loss import LigerFusedLinearGRPOLoss

from trl.trainer.utils import pad, selective_log_softmax


def _get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]

    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
model.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)


inputs = torch.load("inputs.pth", weights_only=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
inputs: dict[str, torch.Tensor]

# eager mode
prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
completion_ids, completion_mask = (
    inputs["completion_ids"],
    inputs["completion_mask"],
)
input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
beta = 0.04
G = 2
B = input_ids.shape[0]
T = input_ids.shape[1]
logits_to_keep = completion_ids.size(1)

per_token_logps = _get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
# Compute the KL divergence between the model and the reference model
with torch.no_grad():
    ref_per_token_logps = _get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep
    )
per_token_kl = (
    torch.exp(ref_per_token_logps - per_token_logps)
    - (ref_per_token_logps - per_token_logps)
    - 1
)
# x - x.detach() allows for preserving gradients from x
advantages = inputs["advantages"]
per_token_loss = torch.exp(
    per_token_logps - per_token_logps.detach()
) * advantages.unsqueeze(1)
per_token_loss = -(per_token_loss - beta * per_token_kl)
loss = (
    (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
).mean()
loss.backward()
grad_weight = model.lm_head.weight.grad
model.zero_grad()

chunked_loss_fn = LigerFusedLinearGRPOLoss(
    beta=beta,
    compiled=False,
)

hidden_states = model.model(input_ids, attention_mask, output_hidden_states=True)[0]
hidden_states = hidden_states
with torch.no_grad():
    ref_hidden_states = model.model(
        input_ids, attention_mask, output_hidden_states=True
    )[0]

weight = model.lm_head.weight
ref_weight = model.lm_head.weight.clone().detach()

# # exclude the last logit, only compute loss on the completion tokens
# # force model to predict <｜Assistant｜><think>\n
# completion_size = completion_ids.size(1) + 3

completion_size = completion_ids.size(1)
hidden_states = hidden_states[:, :-1, :]
ref_hidden_states = ref_hidden_states[:, :-1, :]
hidden_states = hidden_states[:, -completion_size:].contiguous()
ref_hidden_states = ref_hidden_states[:, -completion_size:].contiguous()
attention_mask = attention_mask[:, -completion_size:].contiguous()
input_ids = input_ids[:, -completion_size:].contiguous()
advantages = inputs["advantages"]


chunked_loss, _ = chunked_loss_fn(
    weight=weight,
    inputs=hidden_states,
    input_ids=input_ids,
    attention_mask=attention_mask,
    advantages=advantages,
    ref_inputs=ref_hidden_states,
    ref_weight=ref_weight,
)
chunked_loss.backward()
print(grad_weight.shape)
print(loss)
print(chunked_loss)
torch.testing.assert_close(grad_weight, weight.grad)
