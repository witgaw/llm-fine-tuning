"""
Base trainer class for SFT tasks.

Provides common training infrastructure for all supervised fine-tuning tasks,
while delegating task-specific logic to subclasses.
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from models.external_model import (
    BaseModelType,
    FineTuningConfig,
    get_fine_tuned_external_gpt2,
)
from transformers import AutoTokenizer
from utils import get_device, seed_everything


class BaseTrainer(ABC):
    """
    Base class for SFT training scripts.

    Handles common training infrastructure while delegating task-specific
    logic to subclasses via abstract methods.
    """

    def __init__(self):
        self.args = None
        self.device = None
        self.tokenizer = None
        self.train_df = None
        self.dev_df = None
        self.test_df = None

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with common arguments."""
        parser = argparse.ArgumentParser()

        # Common arguments
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
            "--ft_config",
            type=str,
            default="base",
            choices=["base", "lora"],
            help="Fine-tuning config: base (full), lora",
        )
        parser.add_argument("--max_length", type=int, default=512)

        # Allow subclasses to add task-specific arguments
        self.add_task_arguments(parser)

        return parser

    @abstractmethod
    def add_task_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add task-specific arguments to the parser."""
        pass

    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and return train, dev, test DataFrames."""
        pass

    @abstractmethod
    def get_text_columns(self) -> List[str]:
        """Return list of column names needed for tokenization."""
        pass

    @abstractmethod
    def get_label_info(self) -> Dict:
        """
        Return label information for the task.

        Returns:
            dict with keys:
                - 'num_labels': int (for classification tasks)
                - 'label_names': dict (optional, for logging)
        """
        pass

    @abstractmethod
    def prepare_dataset_row(self, row: pd.Series) -> str:
        """
        Convert a DataFrame row to text for tokenization.

        For classification: return the text to classify
        For generation: return the full prompt + target text
        """
        pass

    @abstractmethod
    def get_model_type(self) -> BaseModelType:
        """Return the model type for this task."""
        pass

    @abstractmethod
    def get_lora_config_fn(self) -> Optional[Callable]:
        """Return the LoRA config function for this task (or None)."""
        pass

    def setup(self, args: argparse.Namespace) -> None:
        """Common setup: seed, device, data loading."""
        self.args = args

        # Initialize random seeds and device
        seed_everything(args.seed)
        self.device = get_device(args.use_gpu)

        # Load data
        print("Loading data...")
        self.train_df, self.dev_df, self.test_df = self.load_data()
        print(f"Train: {len(self.train_df)}, Dev: {len(self.dev_df)}, Test: {len(self.test_df)}")

        # Initialize tokenizer
        print(f"Loading tokenizer for {args.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare HuggingFace datasets from DataFrames."""
        # Apply row preparation
        self.train_df["prepared_text"] = self.train_df.apply(
            self.prepare_dataset_row, axis=1
        )
        self.dev_df["prepared_text"] = self.dev_df.apply(self.prepare_dataset_row, axis=1)
        self.test_df["prepared_text"] = self.test_df.apply(
            self.prepare_dataset_row, axis=1
        )

        # Tokenization function
        def tokenize_function(examples):
            result = self.tokenizer(
                examples["prepared_text"],
                padding="max_length",
                truncation=True,
                max_length=self.args.max_length,
            )

            # For CausalLM, labels are same as input_ids
            if self.get_model_type() == BaseModelType.CausalLM:
                result["labels"] = result["input_ids"].copy()

            return result

        # Determine which columns to keep
        columns_to_keep = ["prepared_text"]
        if "label" in self.train_df.columns:
            columns_to_keep.append("label")

        # Create datasets
        train_dataset = Dataset.from_pandas(self.train_df[columns_to_keep])
        dev_dataset = Dataset.from_pandas(self.dev_df[columns_to_keep])
        test_dataset = Dataset.from_pandas(self.test_df[columns_to_keep])

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        dev_dataset = dev_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        # Rename label column if it exists
        if "label" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("label", "labels")
            dev_dataset = dev_dataset.rename_column("label", "labels")
            test_dataset = test_dataset.rename_column("label", "labels")

        # Set format
        format_columns = ["input_ids", "attention_mask", "labels"]
        train_dataset.set_format("torch", columns=format_columns)
        dev_dataset.set_format("torch", columns=format_columns)
        test_dataset.set_format("torch", columns=format_columns)

        return train_dataset, dev_dataset, test_dataset

    def create_training_config(self, output_dir: Path) -> FineTuningConfig:
        """Create fine-tuning configuration."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get LoRA config if needed
        lora_config = None
        if self.args.ft_config == "lora":
            lora_config_fn = self.get_lora_config_fn()
            if lora_config_fn:
                lora_config = lora_config_fn()

        # Get label info
        label_info = self.get_label_info()

        ft_config = FineTuningConfig(
            directory=str(output_dir),
            base_model_type=self.get_model_type(),
            base_model_name=self.args.model_name,
            dtype=torch.float32,
            peft_config=lora_config,
            optimise_all_base_model_layers=(self.args.ft_config == "base"),
            num_labels=label_info.get("num_labels", 2),
        )

        return ft_config

    def print_configuration(self, output_dir: Path, label_info: Dict) -> None:
        """Print training configuration."""
        print("\nConfiguration:")
        print(f"  Model: {self.args.model_name}")
        print(f"  Fine-tuning: {self.args.ft_config}")
        print(f"  Task: {self.get_model_type().value}")

        if "num_labels" in label_info:
            print(f"  Classes: {label_info['num_labels']}")
        if "label_names" in label_info:
            print(f"  Labels: {list(label_info['label_names'].values())}")

        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Epochs: {self.args.num_epochs}")
        print(f"  Learning rate: {self.args.lr}")
        print(f"  Output: {output_dir}")
        print(f"  Device: {self.device}")

    def train(self) -> None:
        """Main training pipeline."""
        # Parse arguments
        parser = self.create_parser()
        args = parser.parse_args()

        # Setup
        self.setup(args)

        # Prepare datasets
        print("\nPreparing datasets...")
        train_dataset, dev_dataset, test_dataset = self.prepare_datasets()

        # Get output directory
        output_dir = Path(args.output_dir)

        # Create training config
        ft_config = self.create_training_config(output_dir)

        # Print configuration
        label_info = self.get_label_info()
        self.print_configuration(output_dir, label_info)

        # Train
        print("\nStarting training...")
        model, trainer = get_fine_tuned_external_gpt2(
            config=ft_config,
            training_data=train_dataset,
            args=args,
            device=self.device,
        )

        print("\nTraining complete!")
        print(f"Model saved to: {output_dir}")
