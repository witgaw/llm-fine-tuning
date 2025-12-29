import logging
import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import finetuning.config as config
import finetuning.utils as utils
import peft
import torch
import torch.optim
import torch.utils.data as tud
import tqdm
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


@dataclass
class OptimisationMetrics:
    epoch: int
    step: int
    losses: List[float]
    learning_rates: List[float]
    model: Any
    tokenizer: Any


class NoOpGradScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass


class Trainer(ABC):
    @abstractmethod
    def __init__(self, config_env: config.Environment, config_training: config.Training):
        self.log = logging.getLogger(__name__)
        self.config_env = config_env
        self.config_training = config_training
        self.device = utils.Environment.get_device(config_env.preferred_device)
        self.kind = self.__class__.__name__
        self.dtype = self.get_prefered_dtype()

    def _log_trainable_params(self, config, model):
        trainable, total = 0, 0
        if isinstance(model, peft.PeftModel):
            trainable, total = model.get_nb_trainable_parameters()
            self.log.info("LoRA fine-tuning")
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log.info("Full fine-tuning")
        self.log.info(
            f"Training {trainable:,}/{total:,} ({100 * trainable / total:0.3g}%) parameters"
        )

    @abstractmethod
    def compute_loss(self, batch, model, tokenizer):
        pass

    @abstractmethod
    def _get_data_collator(self):
        pass

    @abstractmethod
    def set_reference_model(self, reference_model):
        pass

    def get_prefered_dtype(self):
        if torch.cuda.is_bf16_supported():
            self.log.info(f"using troch.bfloat16 for {__name__}")
            return torch.bfloat16

        self.log.info(
            f"using troch.float16 for {__name__} (bfloat16 not supported on this hardware)"
        )
        return torch.float16

    def _compute_logprobs(self, model, token_ids, n_query):
        outputs = model(input_ids=token_ids[:-1].unsqueeze(0))
        logits = outputs.logits[0, n_query - 1 :]
        response_tokens = token_ids[n_query:]
        all_logprobs = torch.log_softmax(logits, dim=-1)
        return all_logprobs[torch.arange(len(response_tokens)), response_tokens]

    def _compute_logprobs_mean(self, mask, logprobs):
        masked_logprobs = torch.where(mask.bool(), logprobs, torch.tensor(0.0, device=self.device))
        return torch.sum(masked_logprobs) / (torch.sum(mask) + 1e-8)

    def _try_compute_loss(self, batch, model, tokenizer):
        try:
            return self.compute_loss(batch, model, tokenizer)
        except Exception as e:
            self.log.error("Loss computation failed", e)
            if hasattr(e, "__traceback__"):
                tb_frame = e.__traceback__.tb_frame
                line_no = e.__traceback__.tb_lineno
                filename = tb_frame.f_code.co_filename
                self.log.error(f"Error in {filename}:{line_no}")
            raise e

    def train(
        self,
        report_every_n_steps: Optional[int] = None,
        yield_initial=False,
    ) -> Iterable[OptimisationMetrics]:
        utils.Environment.set_random_seed(self.config_env.random_number_seed, self.log)
        try:
            config = self.config_training

            model = utils.Load.model_causal_lm(
                config.model, cache_dir=self.config_env.model_cache
            ).to(self.device)

            if config.lora:
                if config.model.peft_model is not None:
                    msg = (
                        "attempting to get a PEFT model from a model which is already a PEFT model"
                    )
                    self.log.error(msg)
                    raise RuntimeError(msg)
                model = peft.get_peft_model(model, config.lora)  # type: ignore
                self.log.info("Using LoRA configuration")

            self._log_trainable_params(config=config, model=model)

            tokenizer = utils.Load.tokenizer(config.model, cache_dir=self.config_env.model_cache)
            if (
                config.data_training.handler.max_tokens_query is not None
                and config.data_training.handler.max_tokens_response is not None
            ):
                tokenizer.model_max_length = (
                    config.data_training.handler.max_tokens_query
                    + config.data_training.handler.max_tokens_response
                )

            cfg_opt = config.optimisation
            training_dataset = utils.Load.datasplit_from_disk(data_cfg=config.data_training)

            if self.config_env.torch_compile:
                model = torch.compile(model)

            self.set_reference_model(reference_model=model)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg_opt.learning_rate,
                weight_decay=cfg_opt.weight_decay,
            )

            schedulers = []
            milestones = []
            if cfg_opt.steps_warmup_fraction > 0:
                warmup_steps = int(cfg_opt.steps_warmup_fraction * cfg_opt.steps_total)

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )

                schedulers.append(warmup_scheduler)
                milestones.append(warmup_steps)

            if not cfg_opt.target_scheduler:
                schedulers.append(ConstantLR(optimizer, factor=1))
            elif cfg_opt.target_scheduler.lower() == "cosine":
                schedulers.append(
                    CosineAnnealingLR(
                        optimizer,
                        T_max=cfg_opt.steps_total - warmup_steps,
                        eta_min=cfg_opt.learning_rate * 0.1,
                    )
                )
            else:
                raise ValueError(f"unsported target scheduler '{cfg_opt.target_scheduler}'")

            scheduler = SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

            collator = self._get_data_collator()

            dataloader = tud.DataLoader(
                training_dataset,  # type: ignore
                batch_size=config.data_training.batch_size,
                collate_fn=lambda data: collator(tokenizer=tokenizer, data=data),  # type: ignore
                shuffle=config.data_training.shuffle,
            )

            total_batches = None
            if hasattr(training_dataset, "__len__"):
                total_datapoints = len(training_dataset)  # type:ignore
                total_batches = math.ceil(total_datapoints / config.data_training.batch_size)
                self.log.info(
                    f"{total_datapoints:,} datapoints resulting in {total_batches:,} batches (training_batch_size = '{config.data_training.batch_size}')"
                )

            precision = torch.float16
            autocast_supported_devices = "cuda"
            if self.device.type in autocast_supported_devices:
                if hasattr(torch, "cuda"):
                    precision = self.get_prefered_dtype()
                scaler = torch.GradScaler(device=self.device.type)
                self.log.info(f"using CUDA with autocast, dtype='{precision}'")
            else:
                model = model.to(torch.bfloat16)
                scaler = NoOpGradScaler()

            model.train()
            optimiser_step = 0
            epoch = -1
            losses = []
            learning_rates = []
            accumulated_loss = 0
            grad_accum_steps = cfg_opt.gradient_accumulation_sub_steps
            accumulated_steps = 0

            utils.Environment.set_random_seed(self.config_env.random_number_seed, self.log)
            if yield_initial:
                yield OptimisationMetrics(
                    epoch=epoch,
                    step=optimiser_step,
                    losses=losses,
                    learning_rates=learning_rates,
                    model=model,
                    tokenizer=tokenizer,
                )

            if cfg_opt.steps_total == 0:
                return

            if self.device.type in autocast_supported_devices:
                autocast = torch.autocast(device_type=self.device.type, dtype=precision)
            else:
                autocast = nullcontext()

            while optimiser_step <= cfg_opt.steps_total:
                epoch += 1

                train_loss = 0
                num_batches = 0

                with tqdm.tqdm(
                    total=cfg_opt.steps_total,
                    desc=f"train-epoch-{epoch}",
                    unit="step",
                    disable=False,
                    position=0,
                ) as pbar:
                    for data_step, batch in enumerate(dataloader):
                        if optimiser_step >= cfg_opt.steps_total:
                            optimiser_step += 1
                            break

                        utils.Environment.set_random_seed(self.config_env.random_number_seed)

                        with autocast:
                            loss = self._try_compute_loss(batch, model, tokenizer)

                        if loss.item() == 0:
                            self.log.warning("0 loss received, skipping")
                            continue

                        scaler.scale(loss).backward()
                        accumulated_steps += 1
                        accumulated_loss += loss.item()

                        epoch_end = data_step + 1 == len(dataloader)
                        if accumulated_steps % grad_accum_steps == 0 or epoch_end:
                            learning_rate = optimizer.param_groups[0]["lr"]

                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )  # TODO make this a parameter

                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                            optimizer.zero_grad()

                            optimiser_step += 1

                            pbar.update(1)
                            pbar.set_postfix(
                                loss=loss.item(),
                                lr=f"{learning_rate:.2e}",
                                optimiser_step=optimiser_step,
                            )

                            ls = accumulated_loss / accumulated_steps
                            losses.append(ls)
                            learning_rates.append(learning_rate)
                            accumulated_steps = 0
                            accumulated_loss = 0

                            if (
                                report_every_n_steps is not None
                                and optimiser_step % report_every_n_steps == 0
                            ):
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()

                                utils.Environment.set_random_seed(
                                    self.config_env.random_number_seed
                                )

                                yield OptimisationMetrics(
                                    epoch=epoch,
                                    step=optimiser_step,
                                    losses=losses,
                                    learning_rates=learning_rates,
                                    model=model,
                                    tokenizer=tokenizer,
                                )
                                losses = []
                                learning_rates = []
                                model.train()

                            train_loss += ls
                            num_batches += 1

                if num_batches > 1:
                    train_loss = train_loss / num_batches
                pbar.write(
                    f"Epoch: {epoch}, train loss: {train_loss:.3f}, optimiser steps from start: {optimiser_step}"
                )
                if (
                    config.optimisation.checkpoint_n_epochs is not None
                    and (epoch + 1) % config.optimisation.checkpoint_n_epochs == 0
                ):
                    dir = f"{config.model.outpath_save_locally}/epoch_{epoch + 1}"
                    self.log.info(f"saving model checkpoint to: '{dir}'")
                    utils.Save.to_disk(model, tokenizer, dir)

            utils.Save.to_disk(model, tokenizer, config.model.outpath_save_locally)
            utils.Save.to_hf_hub(model, tokenizer, config.model.output_path_save_to_hf_hub)
            if report_every_n_steps is not None and len(losses) > 0:
                return OptimisationMetrics(
                    epoch=epoch,
                    step=optimiser_step,
                    losses=losses,
                    learning_rates=learning_rates,
                    model=model,
                    tokenizer=tokenizer,
                )

        except Exception as e:
            self.log.error("Training failed", e)
            if hasattr(e, "__traceback__"):
                tb_frame = e.__traceback__.tb_frame
                line_no = e.__traceback__.tb_lineno
                filename = tb_frame.f_code.co_filename
                self.log.error(f"Error in {filename}:{line_no}")
