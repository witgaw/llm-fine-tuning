import argparse
import logging
import os

import finetuning.experiment as experiment
import huggingface_hub
import settings
import wandb


def main(
    log,
    epochs: int,
    report_n: int,
    eval_n: int,
    batch_train: int,
    batch_eval: int,
    device: str,
    dev_mode: bool,
):
    # Load API keys from environment variables
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        log.warning("WANDB_API_KEY not set - skipping wandb login")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)
    else:
        log.warning("HF_TOKEN not set - skipping huggingface login")

    runs = (
        [settings.dev_verifier_rloo]
        if dev_mode
        else [
            # VERIFIER:
            # settings.verifier_full,
            # settings.verifier_lora_1,
            # settings.verifier_lora_2,
            # settings.verifier_lora_3,
            # settings.prod_verifier_rloo_1,
            # settings.prod_verifier_rloo_2,
            # settings.prod_verifier_rloo_3,
            # settings.prod_verifier_rloo_4,
            # settings.prod_verifier_rloo_5,
            # settings.prod_verifier_rloo_6,
            # settings.prod_verifier_rloo_7,
            # settings.prod_verifier_rloo_8,
            # settings.prod_verifier_rloo_9,
            # settings.prod_verifier_rloo_11,
            # settings.prod_verifier_rloo_10,
            # settings.prod_verifier_rloo_target1,
            # settings.verifier_gen_leaderboard_post_dpo,
            # settings.verifier_extension_temp_1,
            # settings.verifier_extension_temp_2,
            # settings.verifier_extension_temp_3,
            # settings.verifier_extension_temp_4,
            # settings.verifier_extension_temp_5,
            # settings.verifier_extension_temp_6,
            # settings.verifier_extension_beam_1,
            # settings.verifier_extension_beam_2,
            # settings.verifier_extension_beam_3,
            # settings.verifier_extension_beam_4,
            # settings.verifier_extension_generation_length_1,
            # settings.verifier_extension_generation_length_2,
            # settings.verifier_extension_generation_length_3,
            # settings.verifier_extension_generation_length_4,
            # settings.verifier_extension_top_k_1,
            # settings.verifier_extension_top_k_2,
            # settings.verifier_extension_top_k_3,
            # settings.verifier_extension_top_k_4,
            # settings.verifier_extension_top_p_1,
            # settings.verifier_extension_top_p_2,
            # settings.verifier_extension_top_p_3,
            # settings.verifier_extension_top_p_4,
            # settings.verifier_extension_repetition_1,
            # settings.verifier_extension_repetition_2,
            # settings.verifier_extension_repetition_3,
            # settings.verifier_extension_repetition_4,
            # settings.verifier_extension_combined_1,
            # settings.verifier_extension_verifier_1,
            # settings.verifier_extension_verifier_2,
            # settings.verifier_extension_verifier_3,
            # settings.verifier_extension_verifier_4,
            # settings.verifier_extension_verifier_4b,
            # settings.verifier_extension_verifier_5,
            # settings.verifier_extension_verifier_6,
            # settings.verifier_extension_leaderboard,
            # settings.verifier_extension_leaderboard_2,
            # PREFERENCES:
            # settings.preferences_full,
            # settings.preferences_lora_1,
            # settings.preferences_lora_2,
            # settings.preferences_dpo_1,
            # settings.preferences_dpo_2,
            # settings.preferences_dpo_3,
            # settings.preferences_gen_leaderboard_post_dpo,
            # settings.preferences_dpo_1
        ]
    )

    for r in runs:
        if r.sft is not None:
            if batch_train > -1:
                r.sft.data_training.batch_size = batch_train
            if batch_eval > -1:
                if r.sft.data_evaluation is not None:
                    r.sft.data_evaluation.batch_size = batch_eval
            if len(device) > 0:
                r.env.preferred_device = device
            if epochs > -1:
                r.sft.optimisation.steps_total = epochs
            if report_n > -1:
                r.sft.optimisation.report_n_steps = report_n
            if eval_n > -1:
                r.sft.optimisation.eval_n_steps = eval_n if eval_n > 0 else None

    for exp in runs:
        try:
            trainer = experiment.Manager(exp)
            trainer.train()
            del trainer
        except Exception as e:
            log.error(f"run '{exp.name}' failed, exception: '{e}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RL fine-tuning training, flags overwrite defaults."
    )
    parser.add_argument("--dev", action="store_true", help="Run in dev mode")
    parser.add_argument("--epochs", type=int, default=-1, help="Number of epochs")
    parser.add_argument(
        "--report_n", type=int, default=-1, help="Report training metrics every n steps"
    )
    parser.add_argument(
        "--eval_n",
        type=int,
        default=-1,
        help="Report training eval metrics every n steps",
    )
    parser.add_argument("--batch_train", type=int, default=-1, help="Batch size for training")
    parser.add_argument("--batch_eval", type=int, default=-1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Preferred device")
    args = parser.parse_args()

    main(
        log=logging.getLogger("script_train"),
        epochs=args.epochs,
        report_n=args.report_n,
        eval_n=args.eval_n,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        device=args.device,
        dev_mode=args.dev,
    )
