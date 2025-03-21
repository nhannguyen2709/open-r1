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
from transformers import AutoTokenizer, PreTrainedTokenizer

# from open_r1.rewards import answer_parser
from sympy import simplify
import sys
from fire import Fire
from openai import OpenAI


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


def select_answer(answers: list[str], scores: list[float]) -> int:
    """
    Majority vote with random tie-breaker.
    """
    counter = Counter()
    for i, answer in enumerate(answers):
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += scores[i]
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return str(answer)


def get_rewards(prm: OpenAI, prm_tokenizer: AutoTokenizer, question: str, output_texts: list[str]) -> list[float]:
    rm_prompts = []
    for text in output_texts:
        rm_messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(text.split("\n\n")) + "<extra_0>"},
        ]
        rm_prompt = prm_tokenizer.apply_chat_template(
            conversation=rm_messages, tokenize=True, add_generation_prompt=False, max_length=4096, truncation=True
        )
        rm_prompts.append(rm_prompt)
    prm_outputs = prm.embeddings.create(
        input=rm_prompts,
        model=prm.models.list().data[0].id,
    )
    all_probs = [np.array(output.embedding).reshape(-1, 2) for output in prm_outputs.data]
    all_rewards = []
    for step_probs in all_probs:
        step_probs = step_probs[:, 1]
        all_rewards.append(step_probs.prod())
    return all_rewards


def predict_for_question(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prm: OpenAI,
    prm_tokenizer: AutoTokenizer,
    cutoff_times: list[int],
    question: str,
    ground_truth: int,
    max_num_seqs: int,
    max_model_len: int,
) -> tuple[int, list[str]]:
    if time.time() > cutoff_time:
        return 210, []

    num_seqs = max_num_seqs

    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * max_num_seqs // 3

    start = time.time()
    turn_1_max_tokens = 2048
    num_seqs_to_keep = 16
    stop = None
    sampling_kwargs = {
        "temperature": 0.6,
        "min_p": 0.05,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "skip_special_tokens": True,
        "seed": 3407,
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process as an internal monologue and then provide the user with the answer. The reasoning process MUST BE enclosed within <think> and </think> tags.",
        },
        {
            "role": "user",
            "content": question + "\nPlease put the final answer within \\boxed{}, after taking modulo 1000.",
        },
    ]
    prompt_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True)

    request_output = llm.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=SamplingParams(**sampling_kwargs, max_tokens=turn_1_max_tokens, n=num_seqs, stop=stop),
    )
    output_texts = [output.text for output in request_output[0].outputs]
    all_rewards = get_rewards(prm, prm_tokenizer, question, output_texts)

    # sort by rewards, infer top num_seqs_to_keep sequences
    sorted_idxs = np.argsort(all_rewards)[::-1][:num_seqs_to_keep]
    remaining_prompts_ids = []
    for idx in sorted_idxs:
        remaining_prompts_ids.append(prompt_ids + list(request_output[0].outputs[idx].token_ids))
    remaining_outputs = llm.generate(
        prompt_token_ids=remaining_prompts_ids,
        sampling_params=SamplingParams(**sampling_kwargs, max_tokens=max_model_len - turn_1_max_tokens, stop="</think>"),
    )
    lengths = []
    all_extracted_answers = []
    predictions = []
    for idx, output in zip(sorted_idxs, remaining_outputs):
        completion_text = request_output[0].outputs[idx].text + " " + output.outputs[0].text
        completion_ids = list(request_output[0].outputs[idx].token_ids) + list(output.outputs[0].token_ids)
        answer = extract_boxed_text(completion_text)
        if answer:
            all_extracted_answers.append(answer)
            predictions.append(completion_text)
            lengths.append(len(completion_ids))
    # re-calculate rewards, then select answer with highest total reward
    if len(predictions) > 0:
        all_rewards = get_rewards(prm, prm_tokenizer, question, predictions)
        answer = select_answer(all_extracted_answers, all_rewards)
    else:
        answer = 210
        print(f"No prediction contains \\boxed{{}}, using 210 as answer")

    print(f"Max length: {max(lengths)}, Min length: {min(lengths)}, Mean length: {sum(lengths) / len(lengths)}")
    print(f"Time taken: {time.time() - start:.2f} seconds")
    print(f"Candidates: {[(answer, reward) for answer, reward in zip(all_extracted_answers, all_rewards)]}")
    print(f"Final answer: {answer} - Ground truth: {ground_truth}")

    cutoff_times.pop()
    return answer, predictions


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(
    llm,
    tokenizer,
    prm,
    prm_tokenizer,
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
        prm,
        prm_tokenizer,
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
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 6 * 60, 50 + 1)]
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

    prm = OpenAI(base_url="http://localhost:8000/v1", api_key="NVIDIA")
    prm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B")
    llm = LLM(
        llm_model_pth,
        quantization=quantization,
        max_num_seqs=max_num_seqs,  # Maximum number of sequences per iteration. Default is 256
        max_model_len=max_model_len,  # Model context length
        trust_remote_code=True,  # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
        tensor_parallel_size=torch.cuda.device_count(),  # The number of GPUs to use for distributed execution with tensor parallelism
        gpu_memory_utilization=0.8,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
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
            prm,
            prm_tokenizer,
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
