import copy
import datetime
import gc
import os
import zoneinfo

import finetuning.config as config
import finetuning.evaluation as evaluation
import finetuning.training
import finetuning.training as training
import finetuning.utils as utils
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class Manager:
    def __init__(self, experiment: config.Experiment) -> None:
        self.config = experiment
        self.config.save_to_disk()

        self.log_file = (
            f"{self.config.save_dir}/{self.config.name}_{self.config.comment.replace('.', '_')}.log"
        )
        self.log = utils.Environment.setup_logger(__name__, log_file=self.log_file)

        if self.config.env.report_tensorboard:
            self.tensorboard = SummaryWriter(log_dir=f"{self.config.save_dir}/tensorboard")

        self.wandb_project = f"cs224r_rl_finefuning_{self.config.name}"
        self.loss_label = "train/loss"
        self.lr_label = "train/learning_rate"

        self.cleanup_complete = False

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if self.cleanup_complete:
            return

        if self.config.env.report_tensorboard and hasattr(self, "tensorboard"):
            self.tensorboard.close()
        if self.config.env.report_wandb and wandb is not None:
            if self.wanb_run and not getattr(self.wanb_run, "_is_finished", False):
                self.wanb_run.finish(exit_code=1)

        self.cleanup_complete = True

    def train(self, skip_sft=False, skip_rl=False):
        try:
            if skip_sft:
                self.log.warning("Skipping SFT ('skip_sft'='True')")
            else:
                self.sft()

            if skip_rl:
                self.log.warning("Skipping RL ('skip_rl'='True')")
            else:
                self.rl()
        except Exception as e:
            self._cleanup()
            self.log.error(f"Experiment '{self.config.name}' failed, exception: '{e}'")

    def sft(self):
        if self.config.sft is None:
            self.log.warning("SFT config not specified, skipping this step")
            return

        engine_training = training.SFT(self.config.env, self.config.sft)

        self._generic_training_handling(self.config.sft, engine_training)

    def rl(self):
        if self.config.rl is None:
            self.log.warning("RL config not specified, skipping this step")
            return

        if isinstance(self.config.rl, config.DPO):
            return self.dpo()
        elif isinstance(self.config.rl, config.RLOO):
            return self.rloo()
        else:
            msg = f"RL task type '{type(self.config.rl)}' unsupported"
            self.log.warning(msg)
            raise NotImplementedError(msg)

    def rloo(self):
        if not isinstance(self.config.rl, config.RLOO):
            raise ValueError("Expecting 'config.rl' field to be of type 'training.RLOO'")
        engine_training = training.RLOO(self.config.env, self.config.rl)

        self._generic_training_handling(self.config.rl, engine_training)

    def dpo(self):
        if not isinstance(self.config.rl, config.DPO):
            raise ValueError("Expecting 'config.rl' field to be of type 'training.DPO'")
        engine_training = training.DPO(self.config.env, self.config.rl)

        self._generic_training_handling(self.config.rl, engine_training)

    def _generic_training_handling(
        self,
        config_training: config.Training,
        engine_training: finetuning.training.Trainer,
    ):
        training_kind = engine_training.kind

        if self.config.env.report_wandb:
            name = f"[{datetime.datetime.now(zoneinfo.ZoneInfo('Europe/Warsaw')).strftime('%d %b %H:%M %Z')}] {training_kind}"
            if len(self.config.comment) > 0:
                name = f"{name} ({self.config.comment})"
            self.wanb_run = wandb.init(
                project=self.wandb_project, name=name, reinit="finish_previous"
            )

        path_eval_results = None
        engine_eval = None
        if (
            config_training.optimisation.eval_n_steps is not None
            and config_training.data_evaluation is not None
        ):
            training_dir = f"{self.config.save_dir}/{training_kind}"
            path_eval_results = f"{training_dir}/train_eval.csv"

            engine_eval = evaluation.Manager(
                path_results=path_eval_results,
                env_cfg=self.config.env,
                gen_cfg=self.config.gen,
                data_cfg=config_training.data_evaluation,
                preferred_device=self.config.env.preferred_device,
                purge_stored_results=True,
            )
        success = False

        try:
            start = datetime.datetime.now()
            self.log.info(f"Starting {training_kind}...")
            reported_step = 0
            last_epoch = -1
            for intermediate_result in engine_training.train(
                config_training.optimisation.report_n_steps, yield_initial=True
            ):
                if intermediate_result.epoch >= 0:
                    for i, loss in enumerate(intermediate_result.losses):
                        learning_rate = intermediate_result.learning_rates[i]
                        if self.config.env.report_wandb:
                            self.wanb_run.log(
                                data={
                                    self.loss_label: loss,
                                    self.lr_label: learning_rate,
                                },
                                step=reported_step,
                            )
                        if self.config.env.report_tensorboard:
                            self.tensorboard.add_scalar(
                                tag=self.loss_label,
                                scalar_value=loss,
                                global_step=reported_step,
                            )
                            self.tensorboard.add_scalar(
                                tag=self.lr_label,
                                scalar_value=learning_rate,
                                global_step=reported_step,
                            )

                        if last_epoch != intermediate_result.epoch:
                            last_epoch = intermediate_result.epoch
                            if self.config.env.report_wandb:
                                self.wanb_run.log(
                                    data={"epoch": intermediate_result.epoch},
                                    step=reported_step,
                                )
                            if self.config.env.report_tensorboard:
                                self.tensorboard.add_scalar(
                                    tag="epoch",
                                    scalar_value=intermediate_result.epoch,
                                    global_step=reported_step,
                                )
                        reported_step += 1
                    assert reported_step == intermediate_result.step, "step mismatch"
                if (
                    engine_eval is not None
                    # and intermediate_result.step > 0
                    and intermediate_result.step % config_training.optimisation.eval_n_steps  # type: ignore
                    == 0
                ):
                    cleanup_memory()

                    suffix = f"_step_{intermediate_result.step}"

                    eval_model = copy.deepcopy(intermediate_result.model)
                    eval_tokenizer = copy.deepcopy(intermediate_result.tokenizer)
                    eval_model.eval()

                    cfg = config_training.model
                    cfg.variant = f"_step_{intermediate_result.step}"
                    eval_results, json_path = engine_eval.evaluate_from_instances(
                        model_config=cfg,
                        model=eval_model,
                        tokenizer=eval_tokenizer,
                        overwrite=True,
                        device=engine_training.device,
                        output_json=config_training.eval_save_json,
                        json_suffix=suffix,
                    )

                    for k, v in eval_results.items():
                        label = f"train/{k}"
                        if self.config.env.report_wandb:
                            wandb.log(data={label: v}, step=intermediate_result.step)
                        if self.config.env.report_tensorboard:
                            self.tensorboard.add_scalar(
                                tag=label,
                                scalar_value=v,
                                global_step=intermediate_result.step,
                            )

                    if config_training.eval_save_json:
                        if json_path is None:
                            raise RuntimeError("json path not provided by evaluation engine")
                        art_json = wandb.Artifact(f"predictions{suffix}", type="data")
                        art_json.add_file(json_path)
                        self.wanb_run.log_artifact(art_json)

                    if (
                        config_training.eval_save_model
                        and intermediate_result.step != config_training.optimisation.steps_total
                    ):
                        utils.Save.to_hf_hub(
                            intermediate_result.model,
                            intermediate_result.tokenizer,
                            f"{config_training.model.output_path_save_to_hf_hub}{suffix}",
                        )

                    del eval_model
                    del eval_tokenizer
                    cleanup_memory()

            success = True

        except Exception as e:
            self.log.error(f"{training_kind} failed after {datetime.datetime.now() - start}", e)

            if hasattr(e, "__traceback__"):
                tb_frame = e.__traceback__.tb_frame
                line_no = e.__traceback__.tb_lineno
                filename = tb_frame.f_code.co_filename
                self.log.error(f"Error in {filename}:{line_no}")
            raise e

        finally:
            if (
                self.config.env.report_wandb
                and path_eval_results is not None
                and self.wanb_run is not None
            ):
                if os.path.isfile(path_eval_results):
                    artifact = wandb.Artifact("training_eval", type="data")
                    artifact.add_file(path_eval_results)
                    self.wanb_run.log_artifact(artifact)
                if os.path.isfile(self.log_file):
                    artifact = wandb.Artifact("logs", type="log")
                    artifact.add_file(self.log_file)
                    self.wanb_run.log_artifact(artifact)

            if self.config.env.report_wandb:
                exit_code = None if success else 1
                self.wanb_run.finish(exit_code=exit_code)

            self._cleanup()

            if success:
                self.log.info(f"{training_kind} finished in {datetime.datetime.now() - start}")
