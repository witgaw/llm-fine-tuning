"""
Fact-check explanation generation using transformers.

This script trains a model to generate explanations for fact-check verdicts
using the LIAR or FEVER datasets.

Usage:
    python factcheck_generation.py --dataset liar --use_gpu
    python factcheck_generation.py --dataset fever --model_name gpt2
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from datasets_internal import load_claim_data
from models.external_model import (
    BaseModelType,
    FineTuningConfig,
    get_fine_tuned_external_gpt2,
)
from peft import LoraConfig
from transformers import AutoTokenizer
from utils import get_device, seed_everything

# Label mappings
LIAR_LABELS = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

FEVER_LABELS = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO",
}


def get_lora_config():
    """Standard LoRA configuration for causal LM."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["liar", "fever"],
        required=True,
        help="Dataset to use: liar or fever",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Data directory (default: data/{dataset})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name (e.g., 'gpt2', 'Qwen/Qwen2.5-0.5B')",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ft_config",
        type=str,
        default="base",
        choices=["base", "lora"],
        help="Fine-tuning config: base (full), lora",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"

    if args.output_dir is None:
        args.output_dir = f"outputs/{args.dataset}_generation"

    seed_everything(args.seed)
    device = get_device(args.use_gpu)

    print(f"Loading {args.dataset.upper()} dataset from {args.data_dir}...")
    train_df, dev_df, test_df = load_claim_data(args.dataset, args.data_dir)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"Model: {args.model_name}")
    print(f"Fine-tuning config: {args.ft_config}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine text column and label mapping
    text_column = "statement" if args.dataset == "liar" else "claim"
    label_map = LIAR_LABELS if args.dataset == "liar" else FEVER_LABELS

    # Add explanation column (for LIAR, we can use context field)
    # For FEVER, we can use evidence field
    if args.dataset == "liar":
        if "context" in train_df.columns:
            train_df["explanation"] = train_df["context"].fillna("No explanation provided.")
            dev_df["explanation"] = dev_df["context"].fillna("No explanation provided.")
            test_df["explanation"] = test_df["context"].fillna("No explanation provided.")
        else:
            print("Warning: No context field found. Using placeholder explanations.")
            train_df["explanation"] = "No explanation provided."
            dev_df["explanation"] = "No explanation provided."
            test_df["explanation"] = "No explanation provided."
    else:  # fever
        if "evidence" in train_df.columns:
            train_df["explanation"] = train_df["evidence"].apply(
                lambda x: f"Evidence: {x}" if x else "No evidence available."
            )
            dev_df["explanation"] = dev_df["evidence"].apply(
                lambda x: f"Evidence: {x}" if x else "No evidence available."
            )
            test_df["explanation"] = test_df["evidence"].apply(
                lambda x: f"Evidence: {x}" if x else "No evidence available."
            )
        else:
            print("Warning: No evidence field found. Using placeholder explanations.")
            train_df["explanation"] = "No evidence available."
            dev_df["explanation"] = "No evidence available."
            test_df["explanation"] = "No evidence available."

    # Prepare datasets with prompt format
    def create_prompt(row):
        claim = row[text_column]
        label = label_map[row["label"]]
        explanation = row["explanation"]

        # Create prompt: "Claim: X. Verdict: Y. Explanation:"
        prompt = f"Claim: {claim}\nVerdict: {label}\nExplanation:"
        full_text = f"{prompt} {explanation}"

        return full_text

    train_df["full_text"] = train_df.apply(create_prompt, axis=1)
    dev_df["full_text"] = dev_df.apply(create_prompt, axis=1)
    test_df["full_text"] = test_df.apply(create_prompt, axis=1)

    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples["full_text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = Dataset.from_pandas(train_df[["full_text"]])
    dev_dataset = Dataset.from_pandas(dev_df[["full_text"]])
    test_dataset = Dataset.from_pandas(test_df[["full_text"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Configure fine-tuning
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = get_lora_config() if args.ft_config == "lora" else None

    ft_config = FineTuningConfig(
        directory=str(output_dir),
        base_model_type=BaseModelType.CausalLM,
        base_model_name=args.model_name,
        dtype=torch.float32,
        peft_config=lora_config,
        optimise_all_base_model_layers=(args.ft_config == "base"),
    )

    print("\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Fine-tuning: {args.ft_config}")
    print("  Task: Causal LM (Generation)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")

    print("\nStarting training...")
    model, trainer = get_fine_tuned_external_gpt2(
        config=ft_config,
        training_data=train_dataset,
        args=args,
        device=device,
    )

    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
