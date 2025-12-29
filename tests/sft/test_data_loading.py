"""
Tests for dataset loading functions.

These tests verify that data loading functions correctly parse and return
the expected DataFrame structures.
"""

import pandas as pd
import pytest
from datasets_internal import load_claim_data, load_propaganda_data


@pytest.fixture
def tmp_propaganda_data(tmp_path):
    """Create temporary propaganda dataset files."""
    data_dir = tmp_path / "propaganda"
    data_dir.mkdir()

    # Create mock train.csv
    train_data = pd.DataFrame(
        {
            "text": [
                "Example propaganda text 1",
                "Example propaganda text 2",
                "Example propaganda text 3",
            ],
            "label": [0, 1, 2],
        }
    )
    train_data.to_csv(data_dir / "train.csv", sep="\t", index=False)

    # Create mock dev.csv
    dev_data = pd.DataFrame(
        {
            "text": ["Dev example 1", "Dev example 2"],
            "label": [0, 1],
        }
    )
    dev_data.to_csv(data_dir / "dev.csv", sep="\t", index=False)

    # Create mock test.csv
    test_data = pd.DataFrame(
        {
            "text": ["Test example 1"],
            "label": [2],
        }
    )
    test_data.to_csv(data_dir / "test.csv", sep="\t", index=False)

    return data_dir


@pytest.fixture
def tmp_liar_data(tmp_path):
    """Create temporary LIAR dataset files."""
    data_dir = tmp_path / "liar"
    data_dir.mkdir()

    # Create mock train.csv with LIAR structure
    train_data = pd.DataFrame(
        {
            "statement": [
                "This is a political claim",
                "Another claim about politics",
                "A third claim",
            ],
            "label": [0, 3, 5],  # pants-fire, half-true, true
        }
    )
    train_data.to_csv(data_dir / "train.csv", sep="\t", index=False)

    # Create mock dev.csv
    dev_data = pd.DataFrame(
        {
            "statement": ["Dev claim 1", "Dev claim 2"],
            "label": [1, 4],  # false, mostly-true
        }
    )
    dev_data.to_csv(data_dir / "dev.csv", sep="\t", index=False)

    # Create mock test.csv
    test_data = pd.DataFrame(
        {
            "statement": ["Test claim"],
            "label": [2],  # barely-true
        }
    )
    test_data.to_csv(data_dir / "test.csv", sep="\t", index=False)

    return data_dir


@pytest.fixture
def tmp_fever_data(tmp_path):
    """Create temporary FEVER dataset files."""
    data_dir = tmp_path / "fever"
    data_dir.mkdir()

    # Create mock train.csv with FEVER structure
    train_data = pd.DataFrame(
        {
            "claim": [
                "The Earth is round",
                "Water freezes at 0 degrees Celsius",
                "Cats can fly",
            ],
            "label": [0, 0, 1],  # SUPPORTS, SUPPORTS, REFUTES
        }
    )
    train_data.to_csv(data_dir / "train.csv", sep="\t", index=False)

    # Create mock dev.csv
    dev_data = pd.DataFrame(
        {
            "claim": ["Dev claim 1", "Dev claim 2"],
            "label": [1, 2],  # REFUTES, NOT ENOUGH INFO
        }
    )
    dev_data.to_csv(data_dir / "dev.csv", sep="\t", index=False)

    # Create mock test.csv
    test_data = pd.DataFrame(
        {
            "claim": ["Test claim"],
            "label": [0],  # SUPPORTS
        }
    )
    test_data.to_csv(data_dir / "test.csv", sep="\t", index=False)

    return data_dir


def test_load_propaganda_data_returns_three_dataframes(tmp_propaganda_data):
    """Verify load_propaganda_data returns train, dev, test DataFrames."""
    train_df, dev_df, test_df = load_propaganda_data(str(tmp_propaganda_data))

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(dev_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)


def test_load_propaganda_data_correct_sizes(tmp_propaganda_data):
    """Verify propaganda data has correct number of samples."""
    train_df, dev_df, test_df = load_propaganda_data(str(tmp_propaganda_data))

    assert len(train_df) == 3
    assert len(dev_df) == 2
    assert len(test_df) == 1


def test_load_propaganda_data_has_required_columns(tmp_propaganda_data):
    """Verify propaganda data has text and label columns."""
    train_df, dev_df, test_df = load_propaganda_data(str(tmp_propaganda_data))

    for df in [train_df, dev_df, test_df]:
        assert "text" in df.columns
        assert "label" in df.columns


def test_load_propaganda_data_labels_are_integers(tmp_propaganda_data):
    """Verify propaganda labels are numeric."""
    train_df, dev_df, test_df = load_propaganda_data(str(tmp_propaganda_data))

    for df in [train_df, dev_df, test_df]:
        assert df["label"].dtype in [int, "int64"]


def test_load_propaganda_data_missing_directory():
    """Verify load_propaganda_data raises error for missing directory."""
    with pytest.raises(FileNotFoundError):
        load_propaganda_data("nonexistent_directory")


def test_load_claim_data_liar_returns_three_dataframes(tmp_liar_data):
    """Verify load_claim_data returns train, dev, test for LIAR."""
    train_df, dev_df, test_df = load_claim_data("liar", str(tmp_liar_data))

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(dev_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)


def test_load_claim_data_liar_correct_sizes(tmp_liar_data):
    """Verify LIAR data has correct number of samples."""
    train_df, dev_df, test_df = load_claim_data("liar", str(tmp_liar_data))

    assert len(train_df) == 3
    assert len(dev_df) == 2
    assert len(test_df) == 1


def test_load_claim_data_liar_has_statement_column(tmp_liar_data):
    """Verify LIAR data has statement and label columns."""
    train_df, dev_df, test_df = load_claim_data("liar", str(tmp_liar_data))

    for df in [train_df, dev_df, test_df]:
        assert "statement" in df.columns
        assert "label" in df.columns


def test_load_claim_data_fever_returns_three_dataframes(tmp_fever_data):
    """Verify load_claim_data returns train, dev, test for FEVER."""
    train_df, dev_df, test_df = load_claim_data("fever", str(tmp_fever_data))

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(dev_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)


def test_load_claim_data_fever_correct_sizes(tmp_fever_data):
    """Verify FEVER data has correct number of samples."""
    train_df, dev_df, test_df = load_claim_data("fever", str(tmp_fever_data))

    assert len(train_df) == 3
    assert len(dev_df) == 2
    assert len(test_df) == 1


def test_load_claim_data_fever_has_claim_column(tmp_fever_data):
    """Verify FEVER data has claim and label columns."""
    train_df, dev_df, test_df = load_claim_data("fever", str(tmp_fever_data))

    for df in [train_df, dev_df, test_df]:
        assert "claim" in df.columns
        assert "label" in df.columns


def test_load_claim_data_labels_are_integers(tmp_liar_data):
    """Verify claim labels are numeric."""
    train_df, dev_df, test_df = load_claim_data("liar", str(tmp_liar_data))

    for df in [train_df, dev_df, test_df]:
        assert df["label"].dtype in [int, "int64"]


def test_load_claim_data_default_data_dir():
    """Verify load_claim_data constructs default path correctly."""
    # This test verifies the path construction logic without actual file access
    # We expect it to raise FileNotFoundError since the default path won't exist
    with pytest.raises(FileNotFoundError):
        load_claim_data("liar", data_dir=None)


def test_load_claim_data_missing_directory():
    """Verify load_claim_data raises error for missing directory."""
    with pytest.raises(FileNotFoundError):
        load_claim_data("fever", "nonexistent_directory")


def test_propaganda_data_text_content(tmp_propaganda_data):
    """Verify propaganda data text content is preserved correctly."""
    train_df, _, _ = load_propaganda_data(str(tmp_propaganda_data))

    assert train_df["text"].iloc[0] == "Example propaganda text 1"
    assert train_df["label"].iloc[0] == 0


def test_liar_data_statement_content(tmp_liar_data):
    """Verify LIAR statement content is preserved correctly."""
    train_df, _, _ = load_claim_data("liar", str(tmp_liar_data))

    assert train_df["statement"].iloc[0] == "This is a political claim"
    assert train_df["label"].iloc[0] == 0


def test_fever_data_claim_content(tmp_fever_data):
    """Verify FEVER claim content is preserved correctly."""
    train_df, _, _ = load_claim_data("fever", str(tmp_fever_data))

    assert train_df["claim"].iloc[0] == "The Earth is round"
    assert train_df["label"].iloc[0] == 0
