import os
import shutil

import finetuning.utils as utils
from invoke.context import Context
from invoke.tasks import task

# git config --global user.name "Witold"
# git config --global user.email "gawlikowicz@gmail.com"


@task
def tensorboard(ctx: Context, logdir: str = "./experiments") -> None:
    ctx.run(f"uv run tensorboard --logdir={logdir}", pty=True)


@task
def purge(
    ctx: Context,
    experiments_path: str = "./experiments",
) -> None:
    if os.path.exists(experiments_path):
        shutil.rmtree(experiments_path)
        print(f"Removed {experiments_path}")
    else:
        print(f"Path {experiments_path} does not exist")


@task
def propaganda(
    ctx: Context,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    ft_config: str = "lora",
    use_gpu: bool = True,
) -> None:
    """Train propaganda detection model."""
    gpu_flag = "--use_gpu" if use_gpu else ""
    ctx.run(
        f"cd sft && uv run python propaganda_detection.py "
        f"--model_name {model_name} --ft_config {ft_config} {gpu_flag}",
        pty=True,
    )


@task
def claim_verification(
    ctx: Context,
    dataset: str = "liar",
    model_name: str = "Qwen/Qwen2.5-0.5B",
    ft_config: str = "lora",
    use_gpu: bool = True,
) -> None:
    """Train claim verification model."""
    gpu_flag = "--use_gpu" if use_gpu else ""
    ctx.run(
        f"cd sft && uv run python claim_verification.py "
        f"--dataset {dataset} --model_name {model_name} --ft_config {ft_config} {gpu_flag}",
        pty=True,
    )


@task
def factcheck_gen(
    ctx: Context,
    dataset: str = "liar",
    model_name: str = "Qwen/Qwen2.5-0.5B",
    use_gpu: bool = True,
) -> None:
    """Train fact-check generation model."""
    gpu_flag = "--use_gpu" if use_gpu else ""
    ctx.run(
        f"cd sft && uv run python factcheck_generation.py "
        f"--dataset {dataset} --model_name {model_name} {gpu_flag}",
        pty=True,
    )


@task
def prepare_local_splits(ctx: Context, overwrite=False, seed: int = 1):
    datasets = {
        "HuggingFaceTB/smol-smoltalk": ("./data/smoltalk", "train", "test", 0.2),
        "Asap7772/cog_behav_all_strategies": ("./data/warmstart", "train", "test", 0.2),
        "HuggingFaceH4/ultrafeedback_binarized": (
            "./data/ultrafeedback",
            "train_prefs",
            "test_prefs",
            0.2,
        ),
    }

    for ext, cfg in datasets.items():
        local_path = cfg[0]
        key_train = cfg[1]
        key_test = cfg[2]
        dev_size = cfg[3]
        test_size = 0 if len(cfg) < 5 else cfg[4]
        if os.path.exists(local_path) and not overwrite:
            continue
        utils.Splits.prepare_local_splits(
            path_download=ext,
            path_save=local_path,
            size_dev=dev_size,
            size_test=test_size,
            seed=seed,
            key_train=key_train,
            key_test=key_test,
            shuffle=True,
        )
