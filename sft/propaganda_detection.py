"""
Propaganda technique detection using GPT-2 with LoRA/ReFT.

This script trains a GPT-2 model to detect propaganda techniques in text
using the SemEval-2020 Task 11 dataset.

Usage:
    python propaganda_detection.py --use_gpu --ft_config=lora_std
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from utils import get_device, seed_everything

TQDM_DISABLE = False


def load_propaganda_data(data_dir: str = "data/semeval2020"):
    """Load SemEval-2020 propaganda detection dataset."""
    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    return train_df, dev_df, test_df


class PropagandaDataset(torch.utils.data.Dataset):
    """Dataset for propaganda technique detection."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label = row["label"]

        # Tokenize with padding
        encoding = self.tokenizer(
            text,
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


def evaluate(model, eval_loader, device):
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
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/semeval2020")
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
        "--ft_config", type=str, default="base", help="Fine-tuning config: base, lora_std, reft_*"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/propaganda")

    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.use_gpu)

    print(f"Loading propaganda detection data from {args.data_dir}...")
    train_df, dev_df, test_df = load_propaganda_data(args.data_dir)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # TODO: Implement tokenizer and model initialization
    print("\nNOTE: This script requires further adaptation to:")
    print("  1. Initialize tokenizer")
    print("  2. Create model for multi-class classification with LoRA/ReFT")
    print("  3. Implement proper dataset classes")
    print("  4. Add LoRA/ReFT configurations")


if __name__ == "__main__":
    main()
