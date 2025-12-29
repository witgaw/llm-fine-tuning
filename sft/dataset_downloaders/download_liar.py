"""
Download and prepare LIAR dataset for fact-checking.

Dataset: LIAR - Fact-Checking Dataset
Task: Given a statement, classify its veracity (6 classes)
Classes: pants-fire, false, barely-true, half-true, mostly-true, true
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

VERACITY_LABELS = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}


def download_and_prepare(output_dir: str = "data/liar"):
    """
    Download LIAR dataset and prepare it for training.

    Args:
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading LIAR dataset from HuggingFace...")

    # Load dataset from HuggingFace
    dataset = load_dataset("liar")

    print("Dataset loaded:")
    print(f"  - train: {len(dataset['train'])} examples")
    print(f"  - validation: {len(dataset['validation'])} examples")
    print(f"  - test: {len(dataset['test'])} examples")

    # Convert to DataFrame format expected by the codebase
    def to_dataframe(data_split):
        """Convert HuggingFace dataset to pandas DataFrame."""
        df = pd.DataFrame(
            {
                "id": range(len(data_split)),
                "statement": data_split["statement"],
                "label": data_split["label"],
                "subject": data_split["subject"],
                "speaker": data_split["speaker"],
                "context": data_split["context"],
            }
        )
        return df

    train_df = to_dataframe(dataset["train"])
    dev_df = to_dataframe(dataset["validation"])
    test_df = to_dataframe(dataset["test"])

    # Save to CSV files (TSV format to match original codebase)
    train_df.to_csv(output_path / "train.csv", index=False, sep="\t")
    dev_df.to_csv(output_path / "dev.csv", index=False, sep="\t")
    test_df.to_csv(output_path / "test.csv", index=False, sep="\t")

    print(f"\nSaved to {output_path}:")
    print(f"  - train.csv: {len(train_df)} examples")
    print(f"  - dev.csv: {len(dev_df)} examples")
    print(f"  - test.csv: {len(test_df)} examples")

    # Print label distribution
    print("\nLabel distribution (train):")
    for label_id, label_name in VERACITY_LABELS.items():
        count = (train_df["label"] == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(train_df) * 100:.1f}%)")

    return output_path


if __name__ == "__main__":
    download_and_prepare()
