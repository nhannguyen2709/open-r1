"""Reward functions for GRPO training."""

from functools import partial
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


answer_parser = partial(
    parse,
    extraction_config=[
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                equations=True,
                boxed="all",
                units=True,
            ),
            # Ensures that boxed is tried first
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ],
    extraction_mode="first_match",
)
solution_parser = partial(
    parse, extraction_config=[LatexExtractionConfig()], extraction_mode="first_match"
)


# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, sol in zip(contents, solution):
#         gold_parsed = solution_parser(sol)
#         if len(gold_parsed) != 0:
#             # We require the answer to be provided in correct latex (no malformed operators)
#             answer_parsed = answer_parser(content)
#             # Reward 1 if the content is the same as the ground truth, 0 otherwise
#             reward = float(verify(answer_parsed, gold_parsed))
#         else:
#             # If the gold solution is not parseable, we reward 1 to skip this example
#             reward = 1.0
#             # print("Failed to parse gold solution: ", sol)
#         rewards.append(reward)

#     return rewards

def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, ans in zip(contents, answer):
        # print(f"content: {content}")
        # print(f"ans: {ans}")
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = extract_boxed_text(content)
        
        # Reward 1 if the answer_parsed is the same as the ground truth, 0 otherwise
        # print(f"answer_parsed: {answer_parsed}")
        try:
            parsed_value = float(answer_parsed) if answer_parsed else -1e9
            reward = float(int(parsed_value) == int(ans))
        except ValueError:
            reward = 0.0
        # print(f"reward: {reward}")
        rewards.append(reward)

    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
#     pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [
#         re.match(pattern, content, re.DOTALL | re.MULTILINE)
#         for content in completion_contents
#     ]
#     print(f"completion_contents: {completion_contents}")
#     print(f"matches: {matches}")
#     return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, **kwargs):
    """Reward function that gives:
    - 0.25 points if </think> tag is present
    - 0.25 points if \boxed{ is present
    - Additional 0.5 points if both are present
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        reward = 0.0
        if "</think>" in content:
            reward += 0.25
        if "\\boxed{" in content:
            reward += 0.25
        if "</think>" in content and "\\boxed{" in content:
            reward += 0.5
        rewards.append(reward)
    
    # print(f"completion_contents: {completion_contents[0]}")
    # print(rewards[0])
    
    return rewards


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(
    completions: list[Dict[str, str]], solutions: list[str], **kwargs
) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
