import gc
import json
import logging
import os
import random
import tempfile
from typing import Any, Dict, Optional, Tuple

import colorlog
import datasets
import finetuning.config as config
import numpy as np
import peft
import torch
import transformers.models.auto.modeling_auto as tma
import transformers.models.auto.tokenization_auto as tta

SPLITS_FILE = "splits.json"
INFO_FILE = "info.json"


class Environment:
    @staticmethod
    def set_random_seed(seed: int, log=None) -> None:
        if log:
            log.info(f"Setting random number seed across generators to {seed}")

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_logger(
        name: str, debug: bool = False, log_file: Optional[str] = None
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Configure only the first time this method is called
        if not logging.getLogger().hasHandlers():
            handlers = []

            console_handler = colorlog.StreamHandler()
            console_handler.setFormatter(
                colorlog.ColoredFormatter(
                    "%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(name)-30s %(message)s",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                )
            )
            handlers.append(console_handler)

            if log_file is not None:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)-8s %(name)-30s %(message)s")
                )
                handlers.append(file_handler)

            logging.basicConfig(
                level=logging.DEBUG if debug else logging.INFO,
                handlers=handlers,
            )

        return logger

    @staticmethod
    def get_device(preferred: Optional[str] = None):
        log = logging.getLogger(__name__)
        gc.collect()

        if (preferred is None or preferred.lower() in ("cuda", "gpu")) and (
            hasattr(torch, "cuda") and torch.cuda.is_available()
        ):
            log.info("Using CUDA")

            # not sure if it does anything, put let's try to free some memory if possible
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            device_id = torch.cuda.current_device()
            log.info(f"Device ID: {device_id}")
            log.info(f"Device Name: {torch.cuda.get_device_name(device_id)}")
            log.info(
                f"Total Memory (GB): {torch.cuda.get_device_properties(device_id).total_memory / 1e9}"
            )
            log.info(f"Device Properties: {torch.cuda.get_device_properties(device_id)}")

            return torch.device("cuda")
        if (preferred is None or preferred.lower() == "mps") and (
            hasattr(torch, "mps") and torch.mps.is_available()
        ):
            log.info("Using MPS")
            return torch.device("mps")
        log.info("Using CPU")
        return torch.device("cpu")


class Load:
    @staticmethod
    def dataset_from_hf(path: str, cache_dir: Optional[str] = "./data", streaming: bool = False):
        return datasets.load_dataset(path, cache_dir=cache_dir, streaming=streaming)

    @staticmethod
    def dataset_from_disk(
        path: str, streaming: bool = False
    ) -> Tuple[datasets.DatasetDict | datasets.IterableDatasetDict, Dict[str, Any]]:
        log = logging.getLogger(__name__)
        log.info(f"Loading data from disk: '{path}'")
        if not os.path.isdir(path):
            msg = f"incorrect argument 'path'='{path}', expecting a valid directory"
            log.error(msg)
            raise ValueError(msg)
        with open(f"{path}/{SPLITS_FILE}") as f:
            splits = json.load(f)
        with open(f"{path}/{INFO_FILE}") as f:
            info = json.load(f)

        return datasets.load_dataset("parquet", data_files=splits, streaming=streaming), info  # type: ignore

    @staticmethod
    def datasplit_from_disk(data_cfg: "config.Data"):
        datasplit = None
        size = -1
        if data_cfg.split is None:
            datasplit = datasets.Dataset.from_json(data_cfg.path)
            size = len(datasplit)
        else:
            dataset, sizes = Load.dataset_from_disk(data_cfg.path, streaming=data_cfg.stream)
            datasplit = dataset[data_cfg.split.value]
            size = sizes[data_cfg.split.value]["size"]

        if data_cfg.reduce_to_n_examples is not None:
            logging.getLogger(__name__).info(
                f"reducing '{data_cfg.path}' (split: '{data_cfg.split.value if data_cfg.split else 'default'}') from '{size:,}' to '{data_cfg.reduce_to_n_examples:,}' data points"
            )
            datasplit = datasplit.take(data_cfg.reduce_to_n_examples)
        return datasplit

    @staticmethod
    def model_causal_lm(
        config: "config.Model",
        cache_dir: Optional[str] = "./models",
    ):
        log = logging.getLogger(__name__)
        base_model_path = config.base_model
        log.info(
            f"Loading base model: '{base_model_path}'{' (cache_dir=' + str(cache_dir) + ')' if cache_dir is not None else ''}"
        )
        model = tma.AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=cache_dir,
            attn_implementation="eager",
        )
        if config.peft_model:
            peft_model_path = config.peft_model
            log.info(
                f"Loading PEFT model: '{peft_model_path}'"
                + (f" ('cache_dir'='{cache_dir}')" if cache_dir is not None else "")
            )
            model = peft.PeftModel.from_pretrained(model, peft_model_path)

            # Somehow the model loaded from HF has 0 trainable parameters
            model.train()
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    log.info(f"Enabled training for: {name}")
        return model

    @staticmethod
    def tokenizer(
        config: "config.Model",
        cache_dir: Optional[str] = "./models",
    ):
        path = config.base_model
        log = logging.getLogger(__name__)
        log.info(
            f"Loading tokenizer: '{path}'{' (cache_dir=' + str(cache_dir) + ')' if cache_dir is not None else ''}"
        )
        tokenizer = tta.AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            log.warning("tokenizer has no 'pad_token' set, setting to 'eos_token'")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if config.special_tokens:
            tokenizer.add_tokens(config.special_tokens, special_tokens=True)

        return tokenizer


class Save:
    @staticmethod
    def to_disk(model, tokenizer, local_path: Optional[str] = None):
        if local_path is not None and len(local_path) > 0:
            os.makedirs(local_path, exist_ok=True)
            model.save_pretrained(local_path, safe_serialization=True)
            model.config.save_pretrained(local_path)
            # if hasattr(model, "base_model"):
            #     model.base_model.save_pretrained(local_path, safe_serialization=True)
            #     if hasattr(model.base_model, "base_model"):
            #         model.base_model.base_model.save_pretrained(local_path, safe_serialization=True)
            tokenizer.save_pretrained(local_path)

    @staticmethod
    def to_hf_hub(model, tokenizer, hf_hub_path: Optional[str] = None):
        if hf_hub_path is not None and len(hf_hub_path) > 0:
            model.push_to_hub(hf_hub_path, private=True, safe_serialization=True)
            tokenizer.push_to_hub(hf_hub_path, private=True)


class Splits:
    @staticmethod
    def prepare_local_splits(
        path_download: str,
        path_save: str,
        key_train: str,
        key_test: Optional[str] = None,
        size_dev: float = 0.2,
        size_test: float = 0,
        shuffle: bool = True,
        seed: int = 1,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = Load.dataset_from_hf(path=path_download, cache_dir=tmpdir, streaming=False)
            if not isinstance(downloaded, datasets.DatasetDict):
                raise RuntimeError(
                    f"expected datasets to be downloaded as 'DatasetDict', got '{type(downloaded)}' instead"
                )

            if key_train not in downloaded:
                raise ValueError(
                    f"'key_train'='{key_train}' not present in the downloaded dataset, available keys: '{downloaded.keys()}'"
                )

            if key_test is not None and key_test not in downloaded:
                raise ValueError(
                    f"'key_test'='{key_test}' not present in the downloaded dataset, available keys: '{downloaded.keys()}'"
                )

            sum_non_train_size = size_dev + size_test
            if sum_non_train_size >= 1:
                raise ValueError(
                    f"Sum of 'size_test'='{size_dev}' and 'size_test'='{size_test}' must be less than 1"
                )

            if key_test is not None:
                splits = downloaded[key_train].train_test_split(
                    test_size=size_dev, seed=seed, shuffle=shuffle
                )
                dataset_dict = datasets.DatasetDict(
                    {
                        config.DataSplit.TRAIN.value: splits["train"],
                        config.DataSplit.DEV.value: splits["test"],
                        config.DataSplit.TEST.value: downloaded[key_test],
                    }
                )
            else:
                first_split = downloaded[key_train].train_test_split(
                    test_size=sum_non_train_size, seed=seed, shuffle=shuffle
                )

                second_split = first_split["test"].train_test_split(
                    test_size=size_test / sum_non_train_size
                )
                dataset_dict = datasets.DatasetDict(
                    {
                        config.DataSplit.TRAIN.value: first_split["train"],
                        config.DataSplit.DEV.value: second_split["train"],
                        config.DataSplit.TEST.value: second_split["test"],
                    }
                )

            log = logging.getLogger(__name__)

            size_train = len(dataset_dict[config.DataSplit.TRAIN.value])
            size_dev = len(dataset_dict[config.DataSplit.DEV.value])
            size_test = len(dataset_dict[config.DataSplit.TEST.value])
            size_total = size_train + size_dev + size_test
            log.info(
                f"Resulting splits for {path_download}:\n\ttrain={size_train} ({size_train / size_total:0.2%})\n\tdev={size_dev} ({size_dev / size_total:0.2%})\n\ttest={size_test} ({size_test / size_total:0.2%})"
            )

            split_metadata = {}
            split_info = {}
            for split, dataset in dataset_dict.items():
                parquet_path = f"{path_save}/{split}.parquet"
                dataset.to_parquet(parquet_path)
                split_metadata[split] = parquet_path
                split_info[split] = {"size": len(dataset)}

            with open(f"{path_save}/{SPLITS_FILE}", "w") as f:
                json.dump(split_metadata, f)

            with open(f"{path_save}/{INFO_FILE}", "w") as f:
                json.dump(split_info, f)

            # This would save automatically in arrow format, however then it cannot be streamed
            # dataset_dict.save_to_disk(path_save)
