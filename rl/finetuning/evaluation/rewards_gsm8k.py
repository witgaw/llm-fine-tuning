"""
Reward functions for GSM8K mathematical reasoning task.

Evaluates generated solutions to math word problems.
"""

import re


def extract_answer(solution_str):
    """Extract the numerical answer from the solution string."""
    # Look for answer in <answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, solution_str, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
        # Extract number from answer text (handles formats like "$42" or "42 apples")
        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
        if numbers:
            return float(numbers[-1])  # Take last number found
    return None


def extract_target_answer(answer_str):
    """Extract numerical answer from GSM8K answer string."""
    # GSM8K answers are in format: "#### 42"
    if "####" in answer_str:
        answer_text = answer_str.split("####")[-1].strip()
        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
        if numbers:
            return float(numbers[0])

    # Fallback: extract any number
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if numbers:
        return float(numbers[-1])

    return None


def compute_reward(response, target_answer):
    """
    Compute reward for GSM8K reasoning task.

    Args:
        response: Generated reasoning chain with answer
        target_answer: Ground truth answer string (from GSM8K dataset)

    Returns:
        Reward score between 0 and 1
    """
    # Extract predicted answer
    predicted = extract_answer(response)
    target = extract_target_answer(target_answer)

    if predicted is None:
        # No valid answer format
        return 0.1

    if target is None:
        # Can't extract target (shouldn't happen with GSM8K)
        return 0.0

    # Check if answers match (with small tolerance for floating point)
    if abs(predicted - target) < 0.01:
        return 1.0

    # Partial credit for valid format but wrong answer
    return 0.15


def score_batch(responses, target_answers):
    """
    Score a batch of responses.

    Args:
        responses: List of generated responses
        target_answers: List of target answer strings

    Returns:
        List of reward scores
    """
    return [compute_reward(resp, tgt) for resp, tgt in zip(responses, target_answers)]
