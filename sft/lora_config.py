"""Shared LoRA configurations for fine-tuning tasks."""

from peft import LoraConfig


def get_classification_lora_config():
    """Standard LoRA configuration for sequence classification tasks.

    Used by:
    - propaganda_detection.py
    - claim_verification.py
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )


def get_generation_lora_config():
    """Standard LoRA configuration for causal language modeling tasks.

    Used by:
    - factcheck_generation.py
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
