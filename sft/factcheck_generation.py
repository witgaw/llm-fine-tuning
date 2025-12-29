"""
Fact-check explanation generation using transformers.

This script trains a model to generate explanations for fact-check verdicts
using the LIAR or FEVER datasets.

Usage:
    python factcheck_generation.py --dataset liar --use_gpu
    python factcheck_generation.py --dataset fever --model_name gpt2
"""

from base_trainer import BaseTrainer
from datasets_internal import load_claim_data
from lora_config import get_generation_lora_config
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


class FactcheckGenerationTrainer(BaseTrainer):
    """Trainer for fact-check explanation generation task."""

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

        # Override defaults for generation task
        parser.set_defaults(batch_size=8, lr=5e-5)

    def setup(self, args):
        # Set defaults based on dataset
        if args.data_dir is None:
            args.data_dir = f"data/{args.dataset}"
        if args.output_dir is None:
            args.output_dir = f"outputs/{args.dataset}_generation"

        # Call parent setup
        super().setup(args)

        # Add explanation column to dataframes
        self._add_explanation_column()

    def _add_explanation_column(self):
        """Add explanation column based on dataset type."""
        if self.args.dataset == "liar":
            if "context" in self.train_df.columns:
                self.train_df["explanation"] = self.train_df["context"].fillna(
                    "No explanation provided."
                )
                self.dev_df["explanation"] = self.dev_df["context"].fillna(
                    "No explanation provided."
                )
                self.test_df["explanation"] = self.test_df["context"].fillna(
                    "No explanation provided."
                )
            else:
                print("Warning: No context field found. Using placeholder explanations.")
                self.train_df["explanation"] = "No explanation provided."
                self.dev_df["explanation"] = "No explanation provided."
                self.test_df["explanation"] = "No explanation provided."
        else:  # fever
            if "evidence" in self.train_df.columns:
                self.train_df["explanation"] = self.train_df["evidence"].apply(
                    lambda x: f"Evidence: {x}" if x else "No evidence available."
                )
                self.dev_df["explanation"] = self.dev_df["evidence"].apply(
                    lambda x: f"Evidence: {x}" if x else "No evidence available."
                )
                self.test_df["explanation"] = self.test_df["evidence"].apply(
                    lambda x: f"Evidence: {x}" if x else "No evidence available."
                )
            else:
                print("Warning: No evidence field found. Using placeholder explanations.")
                self.train_df["explanation"] = "No evidence available."
                self.dev_df["explanation"] = "No evidence available."
                self.test_df["explanation"] = "No evidence available."

    def load_data(self):
        return load_claim_data(self.args.dataset, self.args.data_dir)

    def get_text_columns(self):
        text_column = "statement" if self.args.dataset == "liar" else "claim"
        return [text_column, "label", "explanation"]

    def get_label_info(self):
        # No num_labels for CausalLM
        label_names = LIAR_LABELS if self.args.dataset == "liar" else FEVER_LABELS
        return {
            "label_names": label_names,
        }

    def prepare_dataset_row(self, row):
        """Create prompt: Claim: X. Verdict: Y. Explanation:"""
        text_column = "statement" if self.args.dataset == "liar" else "claim"
        label_map = LIAR_LABELS if self.args.dataset == "liar" else FEVER_LABELS

        claim = row[text_column]
        label = label_map[row["label"]]
        explanation = row["explanation"]

        prompt = f"Claim: {claim}\nVerdict: {label}\nExplanation:"
        full_text = f"{prompt} {explanation}"

        return full_text

    def get_model_type(self):
        return BaseModelType.CausalLM

    def get_lora_config_fn(self):
        return get_generation_lora_config


if __name__ == "__main__":
    trainer = FactcheckGenerationTrainer()
    trainer.train()
