"""
Download and prepare GSM8K dataset for mathematical reasoning.

Dataset: GSM8K (Grade School Math 8K)
Task: Given a grade school math word problem, generate a solution with reasoning steps
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def download_and_prepare(output_dir: str = "data/gsm8k"):
    """
    Download GSM8K dataset and prepare it for training.

    Args:
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset from HuggingFace...")

    # Load dataset from HuggingFace
    dataset = load_dataset("gsm8k", "main")

    print("Dataset loaded:")
    print(f"  - train: {len(dataset['train'])} examples")
    print(f"  - test: {len(dataset['test'])} examples")

    # Convert to DataFrame format
    def to_dataframe(data_split):
        """Convert HuggingFace dataset to pandas DataFrame."""
        df = pd.DataFrame(
            {
                "id": range(len(data_split)),
                "question": data_split["question"],
                "answer": data_split["answer"],
            }
        )
        return df

    train_df = to_dataframe(dataset["train"])
    test_df = to_dataframe(dataset["test"])

    # Create dev split from train (90/10 split)
    train_size = int(0.9 * len(train_df))
    dev_df = train_df.iloc[train_size:].copy().reset_index(drop=True)
    train_df = train_df.iloc[:train_size].copy().reset_index(drop=True)

    # Save to CSV files (TSV format to match original codebase)
    train_df.to_csv(output_path / "train.csv", index=False, sep="\t")
    dev_df.to_csv(output_path / "dev.csv", index=False, sep="\t")
    test_df.to_csv(output_path / "test.csv", index=False, sep="\t")

    print(f"\nSaved to {output_path}:")
    print(f"  - train.csv: {len(train_df)} examples")
    print(f"  - dev.csv: {len(dev_df)} examples")
    print(f"  - test.csv: {len(test_df)} examples")

    # Show example
    print("\nExample problem:")
    example = train_df.iloc[0]
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer'][:100]}...")

    return output_path


if __name__ == "__main__":
    download_and_prepare()
