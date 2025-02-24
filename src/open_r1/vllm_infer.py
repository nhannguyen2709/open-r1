import os
import gc
import time
import warnings
import pandas as pd
import numpy as np
import torch
import re
import keyword
from collections import Counter
import random
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset


pd.set_option('display.max_colwidth', None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 180 * 60, 50 + 1)]

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_pth = "agentica-org/DeepScaleR-1.5B-Preview"
# llm_model_pth = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 8192

llm = LLM(
    llm_model_pth,
    dtype="half",                 # The data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,    # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN,  # Model context length
    trust_remote_code=True,       # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,       # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
    enable_lora=True,
    max_lora_rank=256,
)

tokenizer = llm.get_tokenizer()


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer


def batch_message_generate(list_of_messages) -> list[list[dict]]:
    max_tokens = MAX_MODEL_LEN
    if time.time() > cutoff_times[-1]:
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(
        temperature=0.6,              # randomness of the sampling
        min_p=0.01,
        skip_special_tokens=True,     # Whether to skip special tokens in the output
        max_tokens=max_tokens,
    )
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("my_adapter", 1, "data/DeepScaleR-1.5B-Simple-RL/checkpoint-60")
    )

    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []

    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print()
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]

    return list_of_messages


def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers


# def create_starter_messages(question, index):
#     options = []
#     for _ in range(2):
#         options.append(
#             [
#                 {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
#                 {"role": "user", "content": question},
#             ]
#         )
#     for _ in range(1):
#         options.append(
#             [
#                 {"role": "system", "content": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
#                 {"role": "user", "content": question},
#             ],
#         )
#     for _ in range(1):
#         options.append(
#             [
#                 {"role": "system", "content": "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}."},
#                 {"role": "user", "content": question},
#             ],
#         )
#     return options[index % len(options)]

def create_starter_messages(question, index):
    messages = [
        {"role": "user", "content": question + "\nPlease put the final answer within \\boxed{}."},
    ]
    return messages


def predict_for_question(question: str) -> int:
    print(question)
    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * MAX_NUM_SEQS // 3
    list_of_messages = [create_starter_messages(question, index) for index in range(num_seqs)]
    all_extracted_answers = []
    list_of_messages = batch_message_generate(list_of_messages)
    list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
    all_extracted_answers.extend(extracted_answers)

    print(all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(answer)

    print("\n\n")
    cutoff_times.pop()
    return answer


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_, question):
    print("------")
    print(id_)

    answer = predict_for_question(question)
    print(question)
    print("------\n\n\n")
    return answer


if __name__ == "__main__":
    # Load dataset from HuggingFace
    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    # Convert to DataFrame
    df = pd.DataFrame({
        'id': dataset['ID'],
        'question': dataset['Problem'],
        'answer': dataset['Answer'],
    })
    # Process each row
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        result = predict(row['id'], row['question'])
        results.append(result)
    df['prediction'] = results
    # Calculate accuracy
    df['correct'] = df['prediction'] == df['answer']
    accuracy = df['correct'].mean()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
