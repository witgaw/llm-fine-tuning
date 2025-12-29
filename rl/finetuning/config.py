import logging
import os
from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import finetuning.data_handlers as data_handlers
import peft
import yaml


class DataSplit(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


@dataclass
class Environment:
    random_number_seed: int = 1
    preferred_device: Optional[str] = None
    report_tensorboard: bool = True
    report_wandb: bool = True
    use_vllm: bool = False
    cuda_autocast: bool = False
    torch_compile: bool = False
    output_dir: str = "./experiments"
    model_cache: str = "./models"

    def __post_init__(self):
        self.autocast_supported_devices = ()
        if self.cuda_autocast:
            self.autocast_supported_devices = "cuda"


@dataclass
class Generation:
    temperature: float = 0
    beam_search: bool = False
    beam_n: int = 1
    top_k: int = 0
    top_p: float = 0.9
    repetition_penalty: float = 1
    verifier_threshold: float = 0
    verifier_threshold_secondary: float = 0
    n_generations: int = 1


@dataclass
class Model:
    base_model: str
    peft_model: Optional[str] = None
    outpath_save_locally: Optional[str] = None
    output_path_save_to_hf_hub: Optional[str] = None
    special_tokens: Optional[list[str]] = None
    variant: str = ""


class Data:
    def __init__(
        self,
        path: str,
        split: Optional[DataSplit],
        batch_size: int,
        reduce_to_n_examples: Optional[int] = None,
        stream: bool = False,
        shuffle: bool = False,
    ):
        self.path = path
        self.split = split
        self.batch_size = batch_size
        self.reduce_to_n_examples = reduce_to_n_examples
        self.stream = stream
        self.shuffle = shuffle


class TrainingData(Data):
    def __init__(
        self,
        path: str,
        split: DataSplit,
        # TODO (WG 03/06/2025): HACK
        handler: data_handlers.DataHandlerTraining | data_handlers.DataHandlerEvaluation,
        batch_size: int,
        reduce_to_n_examples: Optional[int] = None,
        stream: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(
            path=path,
            split=split,
            batch_size=batch_size,
            reduce_to_n_examples=reduce_to_n_examples,
            stream=stream,
            shuffle=shuffle,
        )

        self.handler = handler


class EvaluationData(Data):
    def __init__(
        self,
        path: str,
        split: Optional[DataSplit],
        handler: data_handlers.DataHandlerEvaluation,
        batch_size: int,
        reduce_to_n_examples: Optional[int] = None,
        stream: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(
            path=path,
            split=split,
            batch_size=batch_size,
            reduce_to_n_examples=reduce_to_n_examples,
            stream=stream,
            shuffle=shuffle,
        )

        self.handler = handler


@dataclass
class Optimisation:
    steps_total: int = 100
    steps_warmup_fraction: float = 0
    target_scheduler: Optional[str] = None
    learning_rate: float = 1e-5
    weight_decay: float = 0
    gradient_accumulation_sub_steps: int = 1
    report_n_steps: int = 10
    eval_n_steps: Optional[int] = None
    checkpoint_n_epochs: Optional[int] = None

    def __post_init__(self):
        if self.eval_n_steps is not None and self.eval_n_steps % self.report_n_steps != 0:
            raise ValueError(
                f"'eval_n_steps' ('{self.eval_n_steps}') must be a multiple of 'report_n_steps' ('{self.report_n_steps}') "
            )

        if self.steps_warmup_fraction < 0 or self.steps_warmup_fraction > 1:
            raise ValueError(
                f"steps_warmup_fraction must be in [0,1], got '{self.steps_warmup_fraction}'"
            )


@dataclass
class Training(ABC):
    optimisation: Optimisation
    model: Model
    lora: Optional[peft.LoraConfig]
    data_training: TrainingData
    data_evaluation: Optional[EvaluationData] = None
    eval_save_model: bool = False
    eval_save_json: bool = False

    def __post_init__(self):
        if self.lora and self.lora.task_type != peft.TaskType.CAUSAL_LM:
            raise ValueError(f"task type must be set to 'CAUSAL_LM', got '{self.lora.task_type}'")


@dataclass
class SFT(Training):
    pass


@dataclass
class DPO(Training):
    beta: float = 0.1


@dataclass
class RLOO(Training):
    num_generations: int = 4
    kl_penalty: float = 0
    add_length_penalty: bool = False


@dataclass
class Experiment:
    name: str
    env: Environment
    gen: Generation
    comment: str = ""
    sft: Optional[SFT] = None
    rl: Optional[DPO | RLOO] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("experiment name must not be empty")
        self.save_dir = os.path.join(
            self.env.output_dir, f"{self.name}_{self.comment.replace('.', '_').replace(',', '_')}"
        )

    def save_to_disk(self):
        logging.getLogger(__name__).info(f"Saving experiment configuration to {self.save_dir}")
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "experiment_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
