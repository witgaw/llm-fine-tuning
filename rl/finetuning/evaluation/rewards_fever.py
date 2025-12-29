"""
Reward functions for FEVER claim verification task.

Evaluates generated reasoning chains for factual claims.
"""

import re


def extract_answer(solution_str):
    """Extract the verdict from the solution string."""
    # Look for answer in <answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, solution_str, re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()
    return None


def normalize_label(label):
    """Normalize FEVER labels."""
    label = label.strip().upper()

    # Map variations to standard labels
    if "SUPPORT" in label:
        return "SUPPORTS"
    elif "REFUTE" in label:
        return "REFUTES"
    elif "NOT" in label and "ENOUGH" in label:
        return "NOT ENOUGH INFO"

    return label


def compute_reward(response, target_label):
    """
    Compute reward for FEVER reasoning task.

    Args:
        response: Generated reasoning chain with answer
        target_label: Ground truth label (0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO)

    Returns:
        Reward score between 0 and 1
    """
    FEVER_LABELS = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

    target = FEVER_LABELS[target_label]

    # Extract predicted answer
    predicted = extract_answer(response)

    if predicted is None:
        # No valid answer format
        return 0.1

    # Normalize both labels
    predicted = normalize_label(predicted)
    target = normalize_label(target)

    # Exact match
    if predicted == target:
        return 1.0

    # Partial credit for valid format but wrong answer
    return 0.2


def score_batch(responses, targets):
    """
    Score a batch of responses.

    Args:
        responses: List of generated responses
        targets: List of target labels

    Returns:
        List of reward scores
    """
    return [compute_reward(resp, tgt) for resp, tgt in zip(responses, targets)]
