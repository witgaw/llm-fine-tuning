import copy

import finetuning.config as config
import finetuning.data_handlers as data_handlers
import torch
from finetuning.training.base import Trainer


class DPO(Trainer):
    def __init__(
        self,
        config_env: config.Environment,
        config_training: config.DPO,
    ):
        super().__init__(config_env, config_training)
        self.max_tokens_response = config_training.data_training.handler.max_tokens_response

        self.beta = torch.tensor(config_training.beta, device=self.device)

    def set_reference_model(self, reference_model):
        self.reference_model = copy.deepcopy(reference_model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def _get_data_handler(self):
        if not isinstance(
            self.config_training.data_training.handler,
            data_handlers.DataHandlerEvaluation,
        ):
            raise RuntimeError(
                f"Expecting data collator for DPO trainer to be of type ' data_handlers.DataHandlerEvaluation', got '{type(self.config_training.data_training.handler)}'"
            )
        return self.config_training.data_training.handler

    def _get_data_collator(self):
        return self._get_data_handler().collate_fn_eval_from_dataset

    def compute_loss(self, batch, model, tokenizer):
        n_examples = batch["input_ids"].shape[0]

        prefs = []
        for i in range(n_examples):
            chosen_plain, chosen_tokens, chosen_n_query = self.extract_with_check(
                batch["chosen"][i],
                tokenizer=tokenizer,
            )
            rejected_plain, rejected_tokens, rejected_n_query = self.extract_with_check(
                batch["rejected"][i],
                tokenizer=tokenizer,
            )

            chosen_logprobs = self._compute_logprobs(
                model=model, token_ids=chosen_tokens, n_query=chosen_n_query
            )

            rejected_logprobs = self._compute_logprobs(
                model=model, token_ids=rejected_tokens, n_query=rejected_n_query
            )

            with torch.no_grad():
                chosen_logprobs_ref = self._compute_logprobs(
                    model=self.reference_model,
                    token_ids=chosen_tokens,
                    n_query=chosen_n_query,
                )

                rejected_logprobs_ref = self._compute_logprobs(
                    model=self.reference_model,
                    token_ids=rejected_tokens,
                    n_query=rejected_n_query,
                )

            pref = self.beta * (
                (chosen_logprobs.sum() - chosen_logprobs_ref.sum())
                - (rejected_logprobs.sum() - rejected_logprobs_ref.sum())
            )

            prefs.append(pref.float())

        loss = -torch.nn.functional.logsigmoid(torch.stack(prefs)).mean()

        return loss

    def extract_with_check(self, arr, tokenizer):
        if len(arr) != 2:
            raise RuntimeError(
                f"Assuming two entries in arrays with chosen and rejected responses, got: '{len(arr)}'"
            )
        dh = self._get_data_handler()

        plain_query = ""
        if dh.apply_chat_template:
            plain_query = tokenizer.apply_chat_template(
                [arr[0]],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=dh.chat_template,
            )
        else:
            plain_query = arr[0]["content"] + " "

        plain_response = ""
        if dh.apply_chat_template:
            plain_response = tokenizer.apply_chat_template(
                [arr[1]],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=dh.chat_template,
            )
        else:
            plain_response = arr[1]["content"]

        tokens_query = tokenizer.encode(
            plain_query,
            add_special_tokens=False,
            return_tensors="pt",
        ).squeeze(0)

        tokens_response = tokenizer.encode(
            plain_response,
            add_special_tokens=False,
            return_tensors="pt",
        ).squeeze(0)

        plaintext = plain_query + plain_response
        tokens = torch.cat([tokens_query, tokens_response])

        return plaintext, tokens.to(self.device), len(tokens_query)
