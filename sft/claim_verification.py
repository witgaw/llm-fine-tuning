"""
Claim verification using transformers with LoRA/ReFT.

This script trains a model to verify claims using LIAR or FEVER datasets.

Usage:
    python claim_verification.py --dataset liar --use_gpu --ft_config=lora
    python claim_verification.py --dataset fever --model_name gpt2 --ft_config=base
"""

from base_trainer import BaseTrainer
from datasets_internal import load_claim_data
from lora_config import get_classification_lora_config
from models.external_model import BaseModelType

# Label mappings
LIAR_LABELS = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

FEVER_LABELS = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO",
}


class ClaimVerificationTrainer(BaseTrainer):
    """Trainer for claim verification task."""

    def add_task_arguments(self, parser):
        parser.add_argument(
            "--dataset",
            type=str,
            choices=["liar", "fever"],
            required=True,
            help="Dataset to use: liar or fever",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default=None,
            help="Data directory (default: data/{dataset})",
        )
        parser.add_argument("--output_dir", type=str, default=None)

    def setup(self, args):
        # Set defaults based on dataset
        if args.data_dir is None:
            args.data_dir = f"data/{args.dataset}"
        if args.output_dir is None:
            args.output_dir = f"outputs/{args.dataset}_verification"

        # Call parent setup
        super().setup(args)

    def load_data(self):
        return load_claim_data(self.args.dataset, self.args.data_dir)

    def get_text_columns(self):
        text_column = "statement" if self.args.dataset == "liar" else "claim"
        return [text_column]

    def get_label_info(self):
        num_labels = self.train_df["label"].nunique()
        label_names = LIAR_LABELS if self.args.dataset == "liar" else FEVER_LABELS
        return {
            "num_labels": num_labels,
            "label_names": label_names,
        }

    def prepare_dataset_row(self, row):
        text_column = "statement" if self.args.dataset == "liar" else "claim"
        return row[text_column]

    def get_model_type(self):
        return BaseModelType.SequenceClassification

    def get_lora_config_fn(self):
        return get_classification_lora_config


if __name__ == "__main__":
    trainer = ClaimVerificationTrainer()
    trainer.train()
