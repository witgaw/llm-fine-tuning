"""
Fact-check explanation generation using GPT-2.

This script trains a GPT-2 model to generate explanations for fact-check verdicts
based on the LIAR or FEVER datasets.

Usage:
    python factcheck_generation.py --dataset liar --use_gpu
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from utils import get_device, seed_everything

TQDM_DISABLE = False

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


def load_liar_data(data_dir: str = "data/liar"):
    """Load LIAR dataset for generation."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    # Add explanation column (for LIAR, we can use context field)
    for df in [train_df, dev_df, test_df]:
        df["explanation"] = df["context"].fillna("No explanation provided.")

    return train_df, dev_df, test_df


def load_fever_data(data_dir: str = "data/fever"):
    """Load FEVER dataset for generation."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    # Add explanation from evidence
    for df in [train_df, dev_df, test_df]:
        df["explanation"] = df["evidence"].apply(
            lambda x: f"Evidence: {x}" if x else "No evidence available."
        )

    return train_df, dev_df, test_df


class FactCheckDataset(torch.utils.data.Dataset):
    """Dataset for fact-check explanation generation."""

    def __init__(self, data, tokenizer, dataset_type="liar", max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.label_map = LIAR_LABELS if dataset_type == "liar" else FEVER_LABELS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.dataset_type == "liar":
            claim = row["statement"]
        else:  # fever
            claim = row["claim"]

        label = self.label_map[row["label"]]
        explanation = row["explanation"]

        # Create prompt: "Claim: X. Verdict: Y. Explanation:"
        prompt = f"Claim: {claim}\nVerdict: {label}\nExplanation:"
        full_text = f"{prompt} {explanation}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt": prompt,
            "explanation": explanation,
        }


def generate_explanation(model, tokenizer, claim, verdict, device, max_length=200):
    """Generate an explanation for a given claim and verdict."""
    prompt = f"Claim: {claim}\nVerdict: {verdict}\nExplanation:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the explanation part
    explanation = generated_text[len(prompt) :].strip()

    return explanation


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
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="outputs/factcheck_gen")

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"

    seed_everything(args.seed)
    device = get_device(args.use_gpu)

    print(f"Loading {args.dataset.upper()} dataset from {args.data_dir}...")

    if args.dataset == "liar":
        train_df, dev_df, test_df = load_liar_data(args.data_dir)
    else:  # fever
        train_df, dev_df, test_df = load_fever_data(args.data_dir)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # Show example
    print("\nExample prompt:")
    row = train_df.iloc[0]
    if args.dataset == "liar":
        print(f"Claim: {row['statement']}")
        label = LIAR_LABELS[row["label"]]
    else:
        print(f"Claim: {row['claim']}")
        label = FEVER_LABELS[row["label"]]

    print(f"Verdict: {label}")
    print(f"Target explanation: {row['explanation'][:100]}...")

    # TODO: Implement model initialization and training
    print("\nNOTE: This script requires further adaptation to:")
    print("  1. Initialize GPT-2 tokenizer and model for generation")
    print("  2. Implement training loop with causal LM loss")
    print("  3. Add generation evaluation metrics (BLEU, ROUGE, etc.)")
    print("  4. Implement inference mode for generating explanations")


if __name__ == "__main__":
    main()
