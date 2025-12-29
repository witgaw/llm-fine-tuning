"""
Download and prepare FEVER dataset for claim verification.

Dataset: FEVER (Fact Extraction and VERification)
Task: Given a claim, classify whether it is SUPPORTED, REFUTED, or NOT ENOUGH INFO
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

FEVER_LABELS = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO",
}


def download_and_prepare(output_dir: str = "data/fever"):
    """
    Download FEVER dataset and prepare it for training.

    Args:
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading FEVER dataset from HuggingFace...")

    # Load dataset from HuggingFace
    dataset = load_dataset("fever", "v1.0")

    print("Dataset loaded:")
    print(f"  - train: {len(dataset['train'])} examples")
    print(f"  - labelled_dev: {len(dataset['labelled_dev'])} examples")

    # Convert to DataFrame format expected by the codebase
    def to_dataframe(data_split):
        """Convert HuggingFace dataset to pandas DataFrame."""
        df = pd.DataFrame(
            {
                "id": data_split["id"],
                "claim": data_split["claim"],
                "label": data_split["label"],
                "evidence": [str(e) for e in data_split["evidence_wiki_url"]],
            }
        )
        return df

    train_df = to_dataframe(dataset["train"])
    dev_df = to_dataframe(dataset["labelled_dev"])

    # Create a test split from dev (50/50 split)
    dev_size = len(dev_df) // 2
    test_df = dev_df.iloc[dev_size:].copy().reset_index(drop=True)
    dev_df = dev_df.iloc[:dev_size].copy().reset_index(drop=True)

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
    for label_id, label_name in FEVER_LABELS.items():
        count = (train_df["label"] == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(train_df) * 100:.1f}%)")

    return output_path


if __name__ == "__main__":
    download_and_prepare()
