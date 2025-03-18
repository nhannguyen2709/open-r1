import argparse
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
from transformers import PreTrainedTokenizer

from open_r1.rewards import answer_parser
from sympy import simplify
import sys
from fire import Fire


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


seed_everything(seed=3407)


sys.set_int_max_str_digits(1000000)


def create_starter_messages(question: str) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags.",
        },
        {
            "role": "user",
            "content": question + "\nPlease put the final answer within \\boxed{}.",
        },
    ]
    return messages


def extract_boxed_text(text):
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""
    parsed = answer_parser(text)
    try:
        parsed = simplify(parsed[0])
        parsed = int(parsed)
        if parsed > 1e6:
            return ""
        else:
            return parsed
    except:
        return ""


def select_answer(answers: list[str]) -> int:
    """
    Majority vote with random tie-breaker.
    """
    counter = Counter()
    for i, answer in enumerate(answers):
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + 1 / (i + 1)  # weight longest answers more
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return str(answer)


def generate_responses(
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    messages: dict,
    sampling_params: SamplingParams,
) -> list[list[dict]]:
    start = time.time()
    prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
    request_output = llm.generate(prompts=prompt, sampling_params=sampling_params, use_tqdm=False)

    sort_keys_and_list_of_messages: list[tuple[int, list[dict]]] = []

    for single_request_output in request_output[0].outputs:
        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.token_ids),
                messages
                + [
                    {
                        "role": "assistant",
                        "content": single_request_output.text,
                    }
                ],
            )
        )

    sort_keys_and_list_of_messages.sort(key=lambda x: x[0], reverse=True)
    max_len = sort_keys_and_list_of_messages[0][0]
    min_len = sort_keys_and_list_of_messages[-1][0]
    mean_len = sum(x[0] for x in sort_keys_and_list_of_messages) / len(sort_keys_and_list_of_messages)
    print(f"Max length: {max_len}, Min length: {min_len}, Mean length: {mean_len}")

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]

    print(f"Time taken: {time.time() - start:.2f} seconds")
    return list_of_messages


def predict_for_question(
    llm,
    tokenizer,
    cutoff_times,
    question: str,
    ground_truth: int,
    max_num_seqs: int,
    max_model_len: int,
) -> tuple[int, list[str]]:
    if time.time() > cutoff_time:
        return 210, []

    num_seqs = max_num_seqs
    max_tokens = max_model_len

    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * max_num_seqs // 3
        max_tokens = 2 * max_model_len // 3

    sampling_params = SamplingParams(
        temperature=0.6,  # randomness of the sampling
        min_p=0.05,
        top_p=0.95,
        repetition_penalty=1.05,
        skip_special_tokens=True,  # Whether to skip special tokens in the output
        max_tokens=max_tokens,
        stop="</think>",
        seed=3407,
        n=num_seqs,
    )
    messages = create_starter_messages(question)
    list_of_messages = generate_responses(llm, tokenizer, messages, sampling_params)
    all_extracted_answers = []
    predictions = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]["content"])
        if answer:
            all_extracted_answers.append(answer)
            predictions.append(messages[-1]["content"])

    print("Candidates: ", all_extracted_answers)
    answer = select_answer(all_extracted_answers)
    print(f"Final answer: {answer} - Ground truth: {ground_truth}")

    cutoff_times.pop()
    return answer, predictions


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(
    llm,
    tokenizer,
    cutoff_times,
    id_,
    question,
    ground_truth,
    max_num_seqs,
    max_model_len,
):
    print(f"ID: {id_} | Question: {question}")
    answer, predictions = predict_for_question(
        llm,
        tokenizer,
        cutoff_times,
        question,
        ground_truth,
        max_num_seqs,
        max_model_len,
    )
    print("=" * 80)
    return answer, predictions


os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option("display.max_colwidth", None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 180 * 60, 50 + 1)]
warnings.simplefilter("ignore")


def main(
    llm_model_pth: str,
    max_num_seqs: int = 4,
    max_model_len: int = 12282,
    csv_file: str = "~/open-r1/reference-aime-hmmt.csv",
    output_file: str = "~/open-r1/generation/output.csv",
    quantization: str = "compressed-tensors",
):
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"problem": "question"})
    df["answer"] = df["answer"].astype(str)

    llm = LLM(
        llm_model_pth,
        quantization=quantization,
        max_num_seqs=max_num_seqs,  # Maximum number of sequences per iteration. Default is 256
        max_model_len=max_model_len,  # Model context length
        trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=torch.cuda.device_count(),  # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
        seed=3407,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    tokenizer = llm.get_tokenizer()

    # Process each row
    results = []
    predictions_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        result, predictions = predict(
            llm,
            tokenizer,
            cutoff_times,
            row["id"],
            row["question"],
            row["answer"],
            max_num_seqs,
            max_model_len,
        )
        results.append(result)
        predictions_list.append(predictions)
    df["prediction"] = results
    df["generations"] = predictions_list
    # Calculate accuracy
    df["correct"] = df["prediction"] == df["answer"]
    accuracy = df["correct"].mean()
    print(f"Accuracy: {accuracy:.4f}")
    time_taken = time.time() - start_time
    # convert to hours
    time_taken = time_taken / 3600
    print(f"Time taken: {time_taken:.2f} hours")

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    Fire(main)
