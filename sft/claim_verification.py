"""
Claim verification using GPT-2 with LoRA/ReFT.

Supports both LIAR (6-class) and FEVER (3-class) datasets.

Usage:
    # Train on LIAR dataset
    python claim_verification.py --dataset liar --use_gpu

    # Train on FEVER dataset
    python claim_verification.py --dataset fever --use_gpu
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
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
    """Load LIAR fact-checking dataset."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    return train_df, dev_df, test_df, LIAR_LABELS


def load_fever_data(data_dir: str = "data/fever"):
    """Load FEVER claim verification dataset."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    return train_df, dev_df, test_df, FEVER_LABELS


class ClaimDataset(torch.utils.data.Dataset):
    """Dataset for claim verification."""

    def __init__(self, data, tokenizer, text_column="statement", max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row[self.text_column]
        label = row["label"]

        # Tokenize with padding
        encoding = self.tokenizer(
            f"Claim: {text}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train(model, train_loader, optimizer, device, epoch):
    """Training loop for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=TQDM_DISABLE):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, eval_loader, device, label_names):
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", disable=TQDM_DISABLE):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(label_names.values())))

    return {"accuracy": accuracy, "f1": f1}


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
    parser.add_argument("--output_dir", type=str, default="outputs/claim_verification")

    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"data/{args.dataset}"

    seed_everything(args.seed)
    device = get_device(args.use_gpu)

    print(f"Loading {args.dataset.upper()} dataset from {args.data_dir}...")

    if args.dataset == "liar":
        train_df, dev_df, test_df, label_names = load_liar_data(args.data_dir)
        text_column = "statement"
        num_labels = 6
    else:  # fever
        train_df, dev_df, test_df, label_names = load_fever_data(args.data_dir)
        text_column = "claim"
        num_labels = 3

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    print(f"Number of classes: {num_labels}")
    print(f"Labels: {list(label_names.values())}")

    # TODO: Implement model initialization and training
    print("\nNOTE: This script requires further adaptation to:")
    print("  1. Initialize tokenizer and classification model")
    print("  2. Create proper dataset loaders")
    print("  3. Implement training loop with LoRA/ReFT support")
    print("  4. Add evaluation and saving logic")


if __name__ == "__main__":
    main()
