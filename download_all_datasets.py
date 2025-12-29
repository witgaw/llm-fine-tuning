"""
Master script to download all datasets for the project.

This script downloads and prepares all datasets needed for both
supervised fine-tuning and RL experiments.
"""

import argparse
from pathlib import Path


def download_sft_datasets(data_dir: str = "data"):
    """Download all datasets for supervised fine-tuning tasks."""
    from sft.datasets import download_fever, download_liar, download_semeval

    print("=" * 80)
    print("Downloading SFT datasets...")
    print("=" * 80)

    # SemEval-2020 Task 11 for propaganda detection
    print("\n1. SemEval-2020 Task 11 (Propaganda Detection)")
    print("-" * 80)
    download_semeval.download_and_prepare(f"{data_dir}/semeval2020")

    # LIAR for claim verification
    print("\n2. LIAR (Fact-Checking)")
    print("-" * 80)
    download_liar.download_and_prepare(f"{data_dir}/liar")

    # FEVER for claim verification
    print("\n3. FEVER (Claim Verification)")
    print("-" * 80)
    download_fever.download_and_prepare(f"{data_dir}/fever")


def download_rl_datasets(data_dir: str = "data"):
    """Download all datasets for RL experiments."""
    from sft.datasets import download_gsm8k

    print("\n" + "=" * 80)
    print("Downloading RL datasets...")
    print("=" * 80)

    # GSM8K for mathematical reasoning
    print("\n4. GSM8K (Mathematical Reasoning)")
    print("-" * 80)
    download_gsm8k.download_and_prepare(f"{data_dir}/gsm8k")

    # UltraFeedback and SmolTalk are already handled by RL code
    print("\n5. UltraFeedback & SmolTalk")
    print("-" * 80)
    print("These will be downloaded automatically by the RL training scripts.")


def main():
    parser = argparse.ArgumentParser(description="Download all datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save datasets (default: data)",
    )
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Download only SFT datasets",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Download only RL datasets",
    )

    args = parser.parse_args()

    # Create data directory
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.sft_only:
        download_sft_datasets(args.data_dir)
    elif args.rl_only:
        download_rl_datasets(args.data_dir)
    else:
        download_sft_datasets(args.data_dir)
        download_rl_datasets(args.data_dir)

    print("\n" + "=" * 80)
    print("All datasets downloaded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
