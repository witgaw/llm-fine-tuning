import copy
import datetime

import finetuning.config as config
import finetuning.data_handlers as data_handlers
import pandas as pd
import torch
from finetuning.training.base import Trainer


class RLOO(Trainer):
    def __init__(
        self,
        config_env: config.Environment,
        config_training: config.RLOO,
    ):
        super().__init__(config_env, config_training)
        self.max_tokens_response = config_training.data_training.handler.max_tokens_response
        self.k = config_training.num_generations
        self.kl_penalty = torch.tensor(
            config_training.kl_penalty, device=self.device, dtype=torch.float32
        )
        self.reference_model = None
        self.add_length_penalty = config_training.add_length_penalty

    def __del__(self):
        del self.reference_model

    def set_reference_model(self, reference_model):
        self.reference_model = copy.deepcopy(reference_model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def get_prefered_dtype(self):
        self.log.info("using torch.float16 for RLOO")
        return torch.float16

    def _get_data_collator(self):
        if not isinstance(
            self.config_training.data_training.handler,
            data_handlers.DataHandlerEvaluation,
        ):
            raise RuntimeError(
                f"Expecting data collator for RFLOO trainer to be of type ' data_handlers.DataHandlerEvaluation', got '{type(self.config_training.data_training.handler)}'"
            )
        return self.config_training.data_training.handler.collate_fn_eval_from_dataset

    def compute_loss(self, batch, model, tokenizer):
        if self.kl_penalty.item() > 0 and self.reference_model is None:
            raise RuntimeError(
                f"Reference model needs to be set before begining training when kl_coefficient is greater than 0 (current value: '{self.kl_penalty}')"
            )

        start = datetime.datetime.now()

        # stop_tokens = [tokenizer.encode("</answer>")[0]]
        n_examples, n_query = batch["input_ids"].shape

        batch_outputs = model.generate(
            batch["input_ids"].to(device=self.device),
            attention_mask=batch["attention_mask"].to(device=self.device),
            max_new_tokens=self.max_tokens_response,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=self.k,
            do_sample=True,
            return_dict_in_generate=True,
            # output_scores=True, # fast execution, but OOMs for too large k
            use_cache=self.k < 8,
            # use_cache=False,
            num_beams=1,
        )

        batch_logprobs = []
        batch_logprobs_ref = []

        for seq in batch_outputs.sequences:
            batch_logprobs.append(self._compute_logprobs(model, seq, n_query))
            if self.kl_penalty > 0:
                with torch.no_grad():
                    batch_logprobs_ref.append(
                        self._compute_logprobs(self.reference_model, seq, n_query)
                    )

        batch_padding_mask = (batch_outputs.sequences != tokenizer.pad_token_id).detach().clone()

        col_resp = "response"
        col_num = "nums"
        col_target = "target"
        batch_nums = batch[col_num]
        batch_targets = batch[col_target]

        self.log.debug(
            f"Generation (max_tokens={self.max_tokens_response}) as part of RLOO training took: {datetime.datetime.now() - start}s"
        )

        batch_generations = []

        for i_batch in range(n_examples):
            per_prompt_generations = []
            for j_generated_sequence in range(self.k):
                idx = self.k * i_batch + j_generated_sequence
                per_gen_tokens = batch_outputs.sequences[idx]
                response = per_gen_tokens[n_query:]
                per_gen_text = tokenizer.decode(response, skip_special_tokens=False).replace(
                    tokenizer.pad_token, ""
                )

                col_score = "score"

                df = pd.DataFrame(
                    {
                        col_resp: [per_gen_text],
                        col_num: [batch_nums[i_batch]],
                        col_target: [batch_targets[i_batch]],
                    }
                )

                self.config_training.data_training.handler.add_scores_and_answers(  # type: ignore
                    data=df,
                    col_response=col_resp,
                    col_input=col_num,
                    col_target=col_target,
                    col_score=col_score,
                    col_answer="answer",
                )

                per_gen_scores = df[col_score]
                assert len(per_gen_scores) == 1
                per_gen_reward = per_gen_scores[0]

                length_reward = 0
                if self.add_length_penalty:
                    target_length = 404.0  # mean response length from WarmStart
                    len_resp = torch.sum(
                        batch_padding_mask[idx][n_query:]
                    )  # this will skip padding tokens
                    deviation_ratio = abs(len_resp - target_length) / target_length
                    length_reward = torch.exp(-5 * deviation_ratio**2) * 2 - 1  # [-1, 1]
                    length_reward *= 0.05  # half the max format score

                per_gen_logprob_mean = self._compute_logprobs_mean(
                    batch_padding_mask[idx][n_query:], batch_logprobs[idx]
                )

                kl_div = 0
                if self.kl_penalty > 0:
                    per_gen_logprob_mean_ref = self._compute_logprobs_mean(
                        batch_padding_mask[idx][n_query:], batch_logprobs_ref[idx]
                    )
                    kl_div = per_gen_logprob_mean - per_gen_logprob_mean_ref.detach()

                self.log.info(
                    f"Example: {i_batch + 1}, generation: {j_generated_sequence + 1}, base reward: {per_gen_reward}, length reward: {length_reward}, KL div: {kl_div}"
                )

                per_gen_reward += length_reward

                per_prompt_generations.append(
                    {
                        "prompt_and_reply_tokens": per_gen_tokens,
                        "reply_plaintext": per_gen_text,
                        "mean_logprob": per_gen_logprob_mean,
                        "reward": per_gen_reward,
                        "kl_divergence": kl_div,
                    }
                )
            batch_generations.append(per_prompt_generations)

        batch_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        valid_examples = n_examples
        for example in batch_generations:
            assert len(example) == self.k
            rewards = torch.tensor(
                [x["reward"] for x in example], device=self.device, dtype=torch.float32
            )
            # if self.kl_penalty > 0:
            #     kl_divs = torch.tensor(
            #         [x["kl_divergence"] for x in example],
            #         device=self.device,
            #         dtype=torch.float32,
            #     )
            #     rewards += self.kl_penalty * kl_divs
            #     self.log.debug(f"KL divergence (mean): {kl_divs.mean():.3g}")

            if rewards.sum() == 0:
                self.log.warning(f"Rewards for all k={self.k} generations are 0, skipping example")
                valid_examples -= 1
                continue

            sum_rewards = torch.sum(rewards)
            advantages = torch.zeros(len(example), device=self.device, dtype=torch.float32)
            for i, generation in enumerate(example):
                adv = rewards[i] - 1 / (self.k - 1) * (sum_rewards - rewards[i])
                advantages[i] = adv

            # if advantages.std() > 1e-8:
            #     advantages = (advantages - advantages.mean()) / (
            #         advantages.std() + 1e-8
            #     )
            #     advantages = torch.clamp(advantages, -2.0, 2.0)

            example_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            for i, generation in enumerate(example):
                example_loss += advantages[i] * generation["mean_logprob"]
                if self.kl_penalty > 0:
                    example_loss += self.kl_penalty * generation["kl_divergence"]

            batch_loss += example_loss / self.k

        if valid_examples > 0:
            batch_loss /= valid_examples
        else:
            self.log.warning("None of the RLOO examples considered in a batch were valid!")

        del batch_outputs
        del batch_logprobs
        del batch_logprobs_ref
        del batch_padding_mask

        torch.cuda.empty_cache()

        return batch_loss
