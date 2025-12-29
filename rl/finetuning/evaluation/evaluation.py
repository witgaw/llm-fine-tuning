import ast
import datetime
import math
import os
import shutil
import tempfile
from typing import Any, Optional

import finetuning.config as config
import finetuning.data_handlers as data_handlers
import finetuning.utils as utils
import numpy as np
import pandas as pd
import peft
import torch
import torch.utils.data as tud
import tqdm

COL_QUERY_ID = "dataset_query_id"
COL_INPUT = "dataset_input"
COL_TARGET = "dataset_target"
COL_PROMPT = "processed_prompt"
COL_PREFIX_RESPONSE = "generation_"
COL_PREFIX_ANSWER = "answer_"
COL_PREFIX_SCORE = "score_"


class Manager:
    def __init__(
        self,
        path_results: str,
        env_cfg: "config.Environment",
        gen_cfg: "config.Generation",
        data_cfg: "config.EvaluationData",
        preferred_device: Optional[str],
        purge_stored_results: bool = False,
        **kwargs,
    ) -> None:
        self.log = utils.Environment.setup_logger(__name__)
        if not path_results.lower().endswith(".csv"):
            msg = f"expecting a path to a .csv file (existing or target to be created) for 'results_path', got '{path_results}'"
            self.log.error(msg)
            raise ValueError(msg)

        self.path_results = path_results
        if purge_stored_results:
            if os.path.exists(path_results):
                self.log.info(f"purging results file: {path_results} as `reset_results`=`true`")
                os.remove(path_results)
        self.data_cfg = data_cfg
        self.env_cfg = env_cfg
        self.gen_cfg = gen_cfg
        self.preferred_device = preferred_device
        self._tqdm_position = 1
        self.target_col_ultrafeedback = None

    def evaluate_from_config(
        self,
        model_config: "config.Model",
        overwrite: bool = False,
        purge_stored_results: bool = False,
        device: Optional[Any] = None,
        output_json: bool = False,
    ):
        model, tokenizer = self._model_and_tokenizer(
            model_config=model_config, model=None, tokenizer=None, device=device
        )

        results = self.evaluate_from_instances(
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            overwrite=overwrite,
            purge_stored_results=purge_stored_results,
            device=device,
            output_json=output_json,
        )

        return results

    def evaluate_from_instances(
        self,
        model_config: "config.Model",
        model,
        tokenizer,
        overwrite: bool = False,
        purge_stored_results: bool = False,
        device: Optional[Any] = None,
        output_json: bool = False,
        json_suffix: str = "",
    ):
        if model is None:
            raise RuntimeError("model cannot be None")
        if tokenizer is None:
            raise RuntimeError("tokenizer cannot be None")

        start = datetime.datetime.now()

        df = self.generate(
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            overwrite=overwrite,
            purge_stored_results=purge_stored_results,
            device=device,
        )
        df = self.score(
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            overwrite=overwrite,
            df=df,
            device=device,
        )

        score = -1
        malformatted_fraction = -1

        col_score = self._get_score_column_name(model_config=model_config)
        if col_score in df.columns:
            score = df[col_score].mean()
            malformatted_fraction = (df[col_score] < 0.1).sum() / len(df)
        else:
            self.log.warning(f"'{col_score}' not found in results, returning {score}")

        def safe_ast(s):
            if isinstance(s, list) and len(s) > 0 and isinstance(s[0], int):
                return s
            try:
                return ast.literal_eval(s)  # type: ignore
            except:
                return s

        json_path = None
        if output_json:
            suffix_with_extension = f"{json_suffix}.json"
            json_path = self.path_results.replace(".csv", suffix_with_extension)
            col_answ = self._get_answer_column_name(model_config=model_config)
            if False:
                df_json = pd.DataFrame(
                    {
                        # "num": df[COL_INPUT].apply(ast.literal_eval),
                        "num": df[COL_INPUT].apply(safe_ast),
                        "response": df[col_answ],
                        "target": df[COL_TARGET],
                    }
                )

                df_json.to_json(json_path, lines=True, orient="records")
            elif isinstance(self.data_cfg.handler, data_handlers.UltraFeedback):
                col_gen = self._get_response_column_name(model_config=model_config)
                df_json = pd.DataFrame(
                    {
                        "prompt": df[COL_INPUT],
                        "response": df[col_gen],
                    }
                )

                df_json.to_json(json_path, lines=True, orient="records")

            else:
                raise NotImplementedError(
                    f"JSON output not implemented for '{type(self.data_cfg.handler)}' data handler"
                )

        elapsed = datetime.datetime.now() - start
        print(f"evaluation took {int(elapsed.total_seconds())} seconds\n")

        return {
            "score": score,
            "malformatted_fracton": malformatted_fraction,
        }, json_path

    def generate(
        self,
        model_config: "config.Model",
        model: Any = None,
        tokenizer: Any = None,
        overwrite: bool = False,
        purge_stored_results: bool = False,
        df: Optional[pd.DataFrame] = None,
        device: Optional[Any] = None,
    ) -> pd.DataFrame:
        if df is None:
            df = self._get_df(purge_stored_results)
        response_col_name = self._get_response_column_name(model_config=model_config)
        if response_col_name in df.columns and not overwrite:
            self.log.warning(
                f"using cached responses for model '{model_config}', rerun with 'overwrite' = True to regenerate the responses"
            )
            return df

        if self.env_cfg.use_vllm:
            return self.generate_vllm(
                model_config=model_config,
                df=df,
                response_col_name=response_col_name,
                model=model,
                tokenizer=tokenizer,
            )
        return self.generate_hf(
            model_config=model_config,
            df=df,
            response_col_name=response_col_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    def generate_vllm(
        self,
        model_config: "config.Model",
        df: pd.DataFrame,
        response_col_name: str,
        model: Any = None,
        tokenizer: Any = None,
    ) -> pd.DataFrame:
        import os

        import vllm

        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        try:
            model_path = model_config.base_model
            tmp_dir = tempfile.mkdtemp()

            if model_config.peft_model or isinstance(model, peft.PeftModel):
                if model is None:
                    model = utils.Load.model_causal_lm(model_config)

                if not isinstance(model, peft.PeftModel):
                    raise RuntimeError(
                        f"peft config specified, but loaded model is of type '{type(model)}'"
                    )

                merged_model = model.to(torch.float16).merge_and_unload()  # type: ignore
                # merged_model.config.save_pretrained(tmp_dir)
                utils.Save.to_disk(merged_model, tokenizer, local_path=tmp_dir)
                # merged_model.save_pretrained(tmp_dir)
                model_path = tmp_dir

                # lora_request =  vllm.lora.request.LoRARequest("lora adapter", 1, tmp_dir)
                # lora_enable = True
                # utils.Save.to_disk(model, tokenizer, local_path=tmp_dir)

            elif model is not None:
                utils.Save.to_disk(model, tokenizer, local_path=tmp_dir)
                model_path = tmp_dir

            if self.data_cfg.handler.max_tokens_response:
                max_tokens = self.data_cfg.handler.max_tokens_response
            else:
                max_tokens = 25
                self.log.warning(
                    f"{self.data_cfg.handler.max_tokens_response} not specified, using hardcoded fallback: '{max_tokens}'"
                )

            llm = vllm.LLM(
                model=model_path,
                tokenizer=model_path,
                dtype="auto",
                enforce_eager=True,
                seed=self.env_cfg.random_number_seed,
                disable_custom_all_reduce=True,
                disable_log_stats=True,
            )

            nums = None
            targets = None
            is_countdown = False
            if is_countdown:
                nums = df[COL_INPUT]
                targets = df[COL_TARGET]

            if self.gen_cfg.beam_search:
                params = vllm.sampling_params.BeamSearchParams(
                    beam_width=self.gen_cfg.beam_n,
                    temperature=self.gen_cfg.temperature,
                    max_tokens=max_tokens,
                )
                prompts = [{"prompt": p} for p in df[COL_PROMPT]]
                outputs = llm.beam_search(prompts=prompts, params=params)
                responses = [
                    self._extract_best(
                        is_countdown=is_countdown,
                        i=i,
                        nums=nums,
                        targets=targets,
                        generations=[s.text for s in o.sequences],
                        score_threshold=self.gen_cfg.verifier_threshold,
                        secondary_threshold=self.gen_cfg.verifier_threshold_secondary,
                    )
                    for i, o in enumerate(outputs)
                ]

            else:
                params = vllm.SamplingParams(
                    seed=self.env_cfg.random_number_seed,
                    max_tokens=max_tokens,
                    temperature=self.gen_cfg.temperature,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                    skip_special_tokens=False,
                    top_k=self.gen_cfg.top_k,
                    top_p=self.gen_cfg.top_p,
                    repetition_penalty=self.gen_cfg.repetition_penalty,
                    n=self.gen_cfg.n_generations,
                )
                prompts = df[COL_PROMPT].tolist()
                outputs = llm.generate(prompts, params)

                responses = [
                    self._extract_best(
                        is_countdown=is_countdown,
                        i=i,
                        nums=nums,
                        targets=targets,
                        generations=[s.text for s in o.outputs],
                        score_threshold=self.gen_cfg.verifier_threshold,
                        secondary_threshold=self.gen_cfg.verifier_threshold_secondary,
                    )
                    for i, o in enumerate(outputs)
                ]

            response_col_name = self._get_response_column_name(model_config)
            df[response_col_name] = responses
            self._save_df(df)
        except Exception as e:
            self.log.error("vLLM generation failed", e)
            raise e
        finally:
            shutil.rmtree(tmp_dir)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        return df

    def _extract_best(
        self,
        is_countdown,
        i,
        nums,
        targets,
        generations,
        score_threshold: float = 0,
        secondary_threshold: float = 0,
    ):
        if not is_countdown or score_threshold == 0:
            return generations[0]

        ground_truth = {
            "target": targets[i],
            "numbers": nums[i],
        }
        rewards = []
        for g in generations:
            reward = 0  # Reward computation removed (was Countdown-specific)
            rewards.append(reward)

        rewards = np.array(rewards)
        valid_indices = np.where(rewards >= score_threshold)[0]
        if len(valid_indices) > 0:
            return generations[valid_indices[0]]
        elif secondary_threshold > 0:
            valid_indices_2 = np.where(rewards >= secondary_threshold)[0]
            if len(valid_indices_2) > 0:
                return generations[valid_indices_2[0]]

        return generations[0]

    def generate_hf(
        self,
        model_config: "config.Model",
        df: pd.DataFrame,
        response_col_name: str,
        model: Any = None,
        tokenizer: Any = None,
        device: Optional[Any] = None,
    ) -> pd.DataFrame:
        if device is None:
            device = utils.Environment.get_device(self.preferred_device)

        model, tokenizer = self._model_and_tokenizer(
            model_config=model_config, model=model, tokenizer=tokenizer, device=device
        )

        dataloader = tud.DataLoader(
            df[COL_PROMPT],  # type: ignore
            batch_size=self.data_cfg.batch_size,
            collate_fn=lambda data: self.data_cfg.handler.collate_fn_eval(
                tokenizer=tokenizer, data=data
            ),  # type: ignore
            shuffle=False,
        )

        responses = []
        model.eval()
        with tqdm.tqdm(
            dataloader,
            desc="evaluation (generating answers)",
            unit="batch",
            total=math.ceil(len(df) / self.data_cfg.batch_size),
            disable=False,
            position=self._tqdm_position,
        ) as pbar:
            with (
                torch.no_grad(),
                torch.amp.autocast_mode.autocast(device.type, dtype=torch.bfloat16),
            ):
                for batch in pbar:
                    output = model.generate(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        max_new_tokens=self.data_cfg.handler.max_tokens_response,
                        pad_token_id=tokenizer.pad_token_id,
                        num_beams=1,
                        do_sample=False,
                    )

                    for i, row in enumerate(output):
                        # Skipping special tokens needs to be strictly false as reawrd function counts on these
                        generated = tokenizer.decode(row, skip_special_tokens=False)
                        # generated = generated.replace(batch["queries"][i], "")
                        responses.append(generated)
                        self.log.info(generated)

        df[response_col_name] = responses

        self._save_df(df)

        return df

    def score(
        self,
        model_config: "config.Model",
        model: Any = None,
        tokenizer: Any = None,
        overwrite: bool = False,
        df: Optional[pd.DataFrame] = None,
        device: Optional[Any] = None,
    ) -> pd.DataFrame:
        if df is None:
            df = self._get_df()

        col_response = self._get_response_column_name(model_config=model_config)
        if col_response not in df.columns:
            msg = f"expected response column name '{col_response}' not found for model '{model_config}', make sure generation step complets successfully first"
            self.log.error(msg)
            raise RuntimeError(msg)

        col_score = self._get_score_column_name(model_config=model_config)
        if col_score in df.columns and not overwrite:
            self.log.warning(
                f"using cached scores for model '{model_config}', rerun with 'overwrite' = True to regenerate the scores"
            )
            return df

        col_answer = self._get_answer_column_name(model_config=model_config)

        if device is None:
            device = utils.Environment.get_device(self.preferred_device)
        model, tokenizer = self._model_and_tokenizer(
            model_config=model_config, model=model, tokenizer=tokenizer, device=device
        )

        col_target = COL_TARGET

        # HACK
        if isinstance(self.data_cfg.handler, data_handlers.UltraFeedback):
            if self.target_col_ultrafeedback is None:
                # set to first evaluation, then we will score against that in subsequent steps
                # this is super hacky and depends on the fact that we do a preliminary evaluation
                # from the experiment manager before any optimiser steps are taken
                self.target_col_ultrafeedback = col_answer
                self.log.info(f"Using '{col_answer}' as a baseline for win-rate calculation")
            col_target = self.target_col_ultrafeedback

        self.data_cfg.handler.add_scores_and_answers(
            data=df,
            col_response=col_response,
            col_input=COL_INPUT,
            col_target=col_target,
            col_answer=col_answer,
            col_score=col_score,
        )

        self.log.info(f"average score: {df[col_score].mean()}")
        self._save_df(df)

        return df

    def _get_df(self, purge_stored_results: bool = False) -> pd.DataFrame:
        if not purge_stored_results and os.path.isfile(self.path_results):
            return pd.read_csv(self.path_results)
        parent_dir = os.path.dirname(os.path.abspath(self.path_results))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        data = utils.Load.datasplit_from_disk(self.data_cfg)

        df = pd.DataFrame(
            {
                COL_QUERY_ID: self.data_cfg.handler.get_query_id(dataset=data),  # type: ignore
                COL_INPUT: self.data_cfg.handler.get_query(
                    dataset=data,  # type: ignore
                ),
                COL_TARGET: self.data_cfg.handler.get_response(dataset=data),  # type: ignore
                COL_PROMPT: self.data_cfg.handler.get_prompt_plaintext(dataset=data),  # type: ignore
            }
        )

        self._save_df(df)

        return df

    def _get_response_column_name(self, model_config: "config.Model") -> str:
        return f"{COL_PREFIX_RESPONSE}{self._get_model_suffix(model_config)}"

    def _get_answer_column_name(self, model_config: "config.Model") -> str:
        return f"{COL_PREFIX_ANSWER}{self._get_model_suffix(model_config)}"

    def _get_score_column_name(self, model_config: "config.Model") -> str:
        return f"{COL_PREFIX_SCORE}{self._get_model_suffix(model_config)}"

    def _get_model_suffix(self, model_config: "config.Model"):
        suffix = (
            model_config.peft_model
            if model_config.peft_model is not None
            else model_config.base_model
        )
        return f"{suffix} ({model_config.variant})"

    def _save_df(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path_results, index=False)

    def _get_model(self, model_config: "config.Model"):
        model = utils.Load.model_causal_lm(model_config)
        return model

    def _get_tokenizer(self, config: "config.Model"):
        return utils.Load.tokenizer(config)

    def _model_and_tokenizer(
        self, model_config: "config.Model", model=None, tokenizer=None, device=None
    ):
        if model is None:
            model = self._get_model(model_config=model_config)
        if tokenizer is None:
            tokenizer = utils.Load.tokenizer(model_config)
        model.eval()
        if device is not None:
            model = model.to(device)
        return model, tokenizer
