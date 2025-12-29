"""
Claim verification using transformers with LoRA/ReFT.

This script trains a model to verify claims using LIAR or FEVER datasets.

Usage:
    python claim_verification.py --dataset liar --use_gpu --ft_config=lora
    python claim_verification.py --dataset fever --model_name gpt2 --ft_config=base
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from datasets_internal import load_claim_data
from lora_config import get_classification_lora_config
from models.external_model import (
    BaseModelType,
    FineTuningConfig,
    get_fine_tuned_external_gpt2,
)
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
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
        args.output_dir = f"outputs/{args.dataset}_verification"

    seed_everything(args.seed)
    device = get_device(args.use_gpu)

    print(f"Loading {args.dataset.upper()} dataset from {args.data_dir}...")
    train_df, dev_df, test_df = load_claim_data(args.dataset, args.data_dir)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"Model: {args.model_name}")
    print(f"Fine-tuning config: {args.ft_config}")

    # Get number of unique labels
    num_labels = train_df["label"].nunique()
    label_names = LIAR_LABELS if args.dataset == "liar" else FEVER_LABELS
    print(f"Number of classes: {num_labels}")
    print(f"Labels: {list(label_names.values())}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine text column name
    text_column = "statement" if args.dataset == "liar" else "claim"

    # Prepare datasets
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = Dataset.from_pandas(train_df[[text_column, "label"]])
    dev_dataset = Dataset.from_pandas(dev_df[[text_column, "label"]])
    test_dataset = Dataset.from_pandas(test_df[[text_column, "label"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Rename label column to labels for HuggingFace Trainer
    train_dataset = train_dataset.rename_column("label", "labels")
    dev_dataset = dev_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # Set format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Configure fine-tuning
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = get_classification_lora_config() if args.ft_config == "lora" else None

    ft_config = FineTuningConfig(
        directory=str(output_dir),
        base_model_type=BaseModelType.SequenceClassification,
        base_model_name=args.model_name,
        dtype=torch.float32,
        peft_config=lora_config,
        optimise_all_base_model_layers=(args.ft_config == "base"),
        num_labels=num_labels,
    )

    print("\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Fine-tuning: {args.ft_config}")
    print(f"  Classes: {num_labels}")
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
