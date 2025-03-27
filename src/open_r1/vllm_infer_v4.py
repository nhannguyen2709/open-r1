import copy
import os
import time
import warnings
import pandas as pd
import numpy as np
import torch
import re
from collections import Counter
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from sympy import simplify
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


# seed_everything(seed=3407)


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


def get_rewards(
    prm: OpenAI, prm_tokenizer: AutoTokenizer, question: str, output_texts: list[str]
) -> list[float]:
    rm_prompts = []
    for text in output_texts:
        rm_messages = [
            {
                "role": "system",
                "content": "Please reason step by step, and put your final answer within \\boxed{}.",
            },
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "<extra_0>".join(text.split("\n\n")) + "<extra_0>",
            },
        ]
        rm_prompt = prm_tokenizer.apply_chat_template(
            conversation=rm_messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=4096,
            truncation=True,
        )
        rm_prompts.append(rm_prompt)
    prm_outputs = prm.embeddings.create(
        input=rm_prompts,
        model=prm.models.list().data[0].id,
    )
    all_probs = [
        np.array(output.embedding).reshape(-1, 2) for output in prm_outputs.data
    ]
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
    turn_1_max_tokens: int = 3072,
    num_seqs_to_keep: int = 16,
) -> tuple[int, list[str]]:
    if time.time() > cutoff_time:
        return 210, []

    num_seqs = max_num_seqs

    if time.time() > cutoff_times[-1]:
        num_seqs = 2 * max_num_seqs // 3

    start = time.time()
    sampling_kwargs = {
        "temperature": 0.6,
        "min_p": 0.05,
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
            "content": question
            + "\nPlease put the final answer within \\boxed{}, after taking modulo 1000.",
        },
    ]
    prompt_ids = tokenizer.apply_chat_template(
        conversation=messages, tokenize=True, add_generation_prompt=True
    )
    max_tokens = max_model_len - len(prompt_ids)
    num_rounds = 2

    lengths = []
    predictions = []
    all_extracted_answers = []

    request_output = llm.generate(
        prompt_token_ids=[prompt_ids],
        sampling_params=SamplingParams(
            **sampling_kwargs,
            max_tokens=max_tokens,
            n=num_seqs // num_rounds,
            stop="</think>",
        ),
    )
    output_texts = [output.text for output in request_output[0].outputs]
    extracted_answers = [extract_boxed_text(text) for text in output_texts]

    lengths.extend([len(output.token_ids) for output in request_output[0].outputs])
    predictions.extend(output_texts)
    all_extracted_answers.extend(extracted_answers)

    for _ in range(1, num_rounds):
        refined_prompts = []
        old_answers = []
        for answer in extracted_answers:
            if answer != "":
                refined_prompts.append(
                    tokenizer.decode(prompt_ids)
                    + f"<think>\nOkay, so my previous answer is: {answer}, and I need to re-answer the question."
                )
                old_answers.append(answer)

        if len(refined_prompts) > 0:
            request_output = llm.generate(
                prompts=refined_prompts,
                sampling_params=SamplingParams(
                    **sampling_kwargs,
                    max_tokens=max_tokens,
                    n=1,
                    stop="</think>",
                ),
            )
            output_texts = [output.outputs[0].text for output in request_output]
            extracted_answers = [extract_boxed_text(text) for text in output_texts]
            lengths.extend(
                [len(output.outputs[0].token_ids) for output in request_output]
            )
            predictions.extend(output_texts)
            all_extracted_answers.extend(extracted_answers)
            for old, new in zip(old_answers, extracted_answers):
                if old != new:
                    print(f"Model answer updated from {old} to {new}")

    predictions = [
        prediction
        for prediction, answer in zip(predictions, all_extracted_answers)
        if answer != ""
    ]
    lengths = [
        length for length, answer in zip(lengths, all_extracted_answers) if answer != ""
    ]
    all_extracted_answers = [answer for answer in all_extracted_answers if answer != ""]

    # re-calculate rewards, then select answer with highest total reward
    if len(predictions) > 0:
        all_rewards = get_rewards(prm, prm_tokenizer, question, predictions)
        answer = select_answer(all_extracted_answers, all_rewards)
        print(
            f"Max length: {max(lengths)}, Min length: {min(lengths)}, Mean length: {sum(lengths) / len(lengths)}"
        )
        print(
            f"Candidates: {[(answer, reward) for answer, reward in zip(all_extracted_answers, all_rewards)]}"
        )
    else:
        answer = 210
        print(f"No prediction contains \\boxed{{}}, using 210 as answer")

    print(f"Time taken: {time.time() - start:.2f} seconds")
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
    turn_1_max_tokens: int = 3072,
    num_seqs_to_keep: int = 16,
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
        turn_1_max_tokens,
        num_seqs_to_keep,
    )
    print("=" * 80)
    return answer, predictions


os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option("display.max_colwidth", None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 6 * 60, 50 + 1)]
print(time.ctime(start_time))
print([time.ctime(x) for x in cutoff_times])
warnings.simplefilter("ignore")


def main(
    llm_model_pth: str,
    max_num_seqs: int = 40,
    turn_1_max_tokens: int = 3072,
    num_seqs_to_keep: int = 16,
    max_model_len: int = 12282,
    csv_file: str = "~/open-r1/reference-aime-hmmt.csv",
    output_file: str = "~/open-r1/generation/output.csv",
    quantization: str = "compressed-tensors",
):
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"problem": "question"}).sample(len(df))
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
        gpu_memory_utilization=0.9,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
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
            turn_1_max_tokens,
            num_seqs_to_keep,
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
