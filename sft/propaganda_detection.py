"""
Propaganda technique detection using transformers with LoRA/ReFT.

This script trains a model to detect propaganda techniques in text
using the SemEval-2020 Task 11 dataset.

Usage:
    python propaganda_detection.py --use_gpu --ft_config=lora
    python propaganda_detection.py --model_name gpt2 --ft_config=base
"""

from base_trainer import BaseTrainer
from datasets_internal import load_propaganda_data
from lora_config import get_classification_lora_config
from models.external_model import BaseModelType


class PropagandaDetectionTrainer(BaseTrainer):
    """Trainer for propaganda detection task."""

    def add_task_arguments(self, parser):
        parser.add_argument("--data_dir", type=str, default="data/semeval2020")
        parser.add_argument("--output_dir", type=str, default="outputs/propaganda")

    def load_data(self):
        return load_propaganda_data(self.args.data_dir)

    def get_text_columns(self):
        return ["text"]

    def get_label_info(self):
        num_labels = self.train_df["label"].nunique()
        return {
            "num_labels": num_labels,
        }

    def prepare_dataset_row(self, row):
        return row["text"]

    def get_model_type(self):
        return BaseModelType.SequenceClassification

    def get_lora_config_fn(self):
        return get_classification_lora_config


if __name__ == "__main__":
    trainer = PropagandaDetectionTrainer()
    trainer.train()
