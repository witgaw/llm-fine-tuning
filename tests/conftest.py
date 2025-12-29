"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path so we can import from sft/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "sft"))
sys.path.insert(0, str(PROJECT_ROOT / "rl"))


@pytest.fixture
def data_dir():
    """Base data directory."""
    return PROJECT_ROOT / "data"


@pytest.fixture
def sft_dir():
    """SFT code directory."""
    return PROJECT_ROOT / "sft"
