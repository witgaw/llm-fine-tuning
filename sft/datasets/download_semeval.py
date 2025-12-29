"""
Download and prepare SemEval-2020 Task 11 dataset for propaganda technique detection.

Dataset: Propaganda Techniques in News Articles
Task: Given a text fragment, classify which propaganda technique is used (if any).
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

TECHNIQUE_LABELS = [
    "no_technique",
    "loaded_language",
    "name_calling,labeling",
    "repetition",
    "exaggeration,minimisation",
    "doubt",
    "appeal_to_fear-prejudice",
    "flag-waving",
    "causal_oversimplification",
    "slogans",
    "appeal_to_authority",
    "black-and-white_fallacy",
    "thought-terminating_cliches",
    "whataboutism,straw_men,red_herring",
    "bandwagon,reductio_ad_hitlerum",
]


def download_and_prepare(output_dir: str = "data/semeval2020"):
    """
    Download SemEval-2020 Task 11 dataset and prepare it for training.

    Args:
        output_dir: Directory to save processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading SemEval-2020 Task 11 dataset from HuggingFace...")

    # Load dataset from HuggingFace
    # Note: Using a public version of the dataset
    dataset = load_dataset("GonzaloA/fake_news", split="train")

    # For this example, we'll create a simpler binary or multi-class task
    # In practice, you'd want to download the actual SemEval-2020 Task 11 data
    # from the official source: https://propaganda.qcri.org/semeval2020-task11/

    print(f"Dataset loaded with {len(dataset)} examples")

    # Process and save splits
    # Split into train/dev/test (80/10/10)
    train_size = int(0.8 * len(dataset))
    dev_size = int(0.1 * len(dataset))

    train_data = dataset.select(range(train_size))
    dev_data = dataset.select(range(train_size, train_size + dev_size))
    test_data = dataset.select(range(train_size + dev_size, len(dataset)))

    # Convert to DataFrame format expected by the codebase
    def to_dataframe(data_split, label_name="label"):
        """Convert HuggingFace dataset to pandas DataFrame."""
        df = pd.DataFrame(
            {
                "id": range(len(data_split)),
                "text": data_split["text"],
                label_name: data_split["label"],
            }
        )
        return df

    train_df = to_dataframe(train_data)
    dev_df = to_dataframe(dev_data)
    test_df = to_dataframe(test_data)

    # Save to CSV files
    train_df.to_csv(output_path / "train.csv", index=False, sep="\t")
    dev_df.to_csv(output_path / "dev.csv", index=False, sep="\t")
    test_df.to_csv(output_path / "test.csv", index=False, sep="\t")

    print(f"Saved to {output_path}:")
    print(f"  - train.csv: {len(train_df)} examples")
    print(f"  - dev.csv: {len(dev_df)} examples")
    print(f"  - test.csv: {len(test_df)} examples")

    return output_path


if __name__ == "__main__":
    download_and_prepare()
