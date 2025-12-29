"""
Dataset loading utilities for propaganda detection and claim verification tasks.
"""


def load_propaganda_data(data_dir="data/semeval2020"):
    """Load SemEval-2020 propaganda detection dataset."""
    from pathlib import Path

    import pandas as pd

    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    return train_df, dev_df, test_df


def load_claim_data(dataset_name="liar", data_dir=None):
    """Load LIAR or FEVER claim verification dataset."""
    from pathlib import Path

    import pandas as pd

    if data_dir is None:
        data_dir = f"data/{dataset_name}"

    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "train.csv", sep="\t")
    dev_df = pd.read_csv(data_path / "dev.csv", sep="\t")
    test_df = pd.read_csv(data_path / "test.csv", sep="\t")

    return train_df, dev_df, test_df
