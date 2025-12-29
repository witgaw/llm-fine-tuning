import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
import torch
import torch.utils.data as tud
import transformers.tokenization_utils as ttu
from openai import OpenAI

IGNORE_INDEX = -100
ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
# HACK: implementation of chat template in WarmStartHandler assumes this form, if this were to change, so would need that implementation
CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


class DataFrameDataset(tud.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Return as dict or tuple as needed
        return self.df.iloc[idx].to_dict()


class DataHandler(ABC):
    @abstractmethod
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> None:
        self.log = logging.getLogger(__name__)
        self.max_tokens_query = max_tokens_query
        self.max_tokens_response = max_tokens_response
        self.apply_chat_template = apply_chat_template
        self.chat_template = CHAT_TEMPLATE
        self.add_generation_prompt = add_generation_prompt

    def add_right_padding(self, elements: List[List[Any]], max_length: int, padding_element: Any):
        for row in elements:
            diff = max_length - len(row)
            row.extend([padding_element] * diff)

    def _annotate_query_with_roles(self, query: str, role: str, eval: bool):
        if not eval:
            return f"[{role.upper()}]: {query}\n"
        return f"[{role.upper()}]: {query}\n[{ROLE_ASSISTANT.upper()}]: "

    def _validate_key(self, key, dataset):
        if key not in dataset.column_names:
            msg = f"key: '{key}' not found in the provided dataset ('{dataset.info}'), available columns '{dataset.data.column_names}'"
            logging.getLogger(__name__).error(msg)
            raise RuntimeError(msg)


class DataHandlerTraining(DataHandler, ABC):
    @abstractmethod
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            max_tokens_query=max_tokens_query,
            max_tokens_response=max_tokens_response,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=False,
            **kwargs,
        )

    @abstractmethod
    def tokenize_datapoint(self, tokenizer, datapoint, tokens, attn, labels):
        pass

    def collate_fn_train_sft(
        self, tokenizer: ttu.PreTrainedTokenizer, data: List[Any]
    ) -> Dict[str, Any]:
        max_n = 0
        batch_tokens = []
        batch_attn = []
        batch_labels = []
        for datapoint in data:
            tokens = []
            attn = []
            labels = []

            # Since we're ignoring special tokens in tokenizer this might be useful for some models
            if tokenizer.bos_token_id is not None:
                tokens = [tokenizer.bos_token_id]
                attn = [1]
                labels = [IGNORE_INDEX]

            self.tokenize_datapoint(
                tokenizer=tokenizer,
                datapoint=datapoint,
                tokens=tokens,
                attn=attn,
                labels=labels,
            )

            # Since we're ignoring special tokens in tokenizer this might be useful for some models
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
                attn.append(1)
                labels.append(tokenizer.eos_token_id)  # type: ignore

            max_n = max(max_n, len(tokens))
            batch_tokens.append(tokens)
            batch_attn.append(attn)
            batch_labels.append(labels)

        self.add_right_padding(
            elements=batch_tokens,
            max_length=max_n,
            padding_element=tokenizer.pad_token_id,
        )
        self.add_right_padding(
            elements=batch_attn,
            max_length=max_n,
            padding_element=0,
        )
        self.add_right_padding(
            elements=batch_labels,
            max_length=max_n,
            padding_element=IGNORE_INDEX,
        )

        batched_data = {
            "input_ids": torch.tensor(batch_tokens),
            "attention_mask": torch.tensor(batch_attn),
            "labels": torch.tensor(batch_labels),
        }

        return batched_data


class DataHandlerEvaluation(DataHandler, ABC):
    @abstractmethod
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            max_tokens_query=max_tokens_query,
            max_tokens_response=max_tokens_response,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=True,
            **kwargs,
        )

    def collate_fn_eval_from_dataset(self, data, tokenizer):
        batch = self.collate_fn_eval(tokenizer=tokenizer, data=self.get_prompt_plaintext(data))
        additional_data = self.extend_collate(data)
        if additional_data:
            batch.update(additional_data)

        return batch

    def extend_collate(self, data):
        return None

    @abstractmethod
    def collate_fn_eval(
        self, tokenizer: ttu.PreTrainedTokenizer, data: List[Any]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_query_id(self, dataset: datasets.Dataset) -> List[str]:
        pass

    @abstractmethod
    def get_prompt_plaintext(self, dataset: datasets.Dataset) -> List[str]:
        pass

    @abstractmethod
    def get_query(self, dataset: datasets.Dataset) -> List[str]:
        pass

    @abstractmethod
    def get_response(self, dataset: datasets.Dataset) -> List[str]:
        pass

    @abstractmethod
    def add_scores_and_answers(
        self,
        data: pd.DataFrame,
        col_response: str,
        col_input: str,
        col_target: str,
        col_answer: str,
        col_score: str,
    ):
        pass


class SmolTalk(DataHandlerTraining):
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        skip_system_messages: bool,
        max_assistant_messages: Optional[int],
        **kwargs,
    ) -> None:
        super().__init__(
            max_tokens_query=max_tokens_query,
            max_tokens_response=max_tokens_response,
            apply_chat_template=apply_chat_template,
            **kwargs,
        )
        if max_assistant_messages is not None and max_assistant_messages < 1:
            raise ValueError(
                f"'max_assistant_messages' parameter must be positive, got '{max_assistant_messages}'"
            )
        self.max_assistant_responses = max_assistant_messages
        self.skip_system_messages = skip_system_messages

    def tokenize_datapoint(self, tokenizer, datapoint, tokens, attn, labels) -> None:
        n_assistant_responses = 0
        for message in datapoint["messages"]:
            role = message["role"].lower()
            if self.skip_system_messages and role == "system":
                continue

            if self.apply_chat_template:
                per_message_plaintext = tokenizer.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=self.add_generation_prompt,
                    chat_template=self.chat_template,
                )
            else:
                per_message_plaintext = message["content"]

            per_message_encoding = tokenizer(
                per_message_plaintext,  # type: ignore
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=self.max_tokens_query if role == ROLE_USER else self.max_tokens_response,
            )

            tokens.extend(per_message_encoding["input_ids"])  # type: ignore
            attn.extend(per_message_encoding["attention_mask"])  # type: ignore

            if role in (ROLE_SYSTEM, ROLE_USER):
                labels.extend([IGNORE_INDEX] * len(per_message_encoding["input_ids"]))
            elif role == ROLE_ASSISTANT:
                n_assistant_responses += 1
                labels.extend(per_message_encoding["input_ids"])  # type: ignore
                if (
                    self.max_assistant_responses is not None
                    and n_assistant_responses == self.max_assistant_responses
                ):
                    break
            else:
                raise RuntimeError(
                    f"unexpected role '{role}', expecting 'system', 'user' or 'assistant'"
                )


class WarmStart(DataHandlerTraining):
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        reduce_response_to_answer: bool,
        **kwargs,
    ) -> None:
        super().__init__(
            max_tokens_query=max_tokens_query,
            max_tokens_response=max_tokens_response,
            apply_chat_template=apply_chat_template,
            **kwargs,
        )
        self.reduce_response_to_answer = reduce_response_to_answer

    def tokenize_datapoint(self, tokenizer, datapoint, tokens, attn, labels) -> None:
        query = datapoint["query"]
        completion = datapoint["completion"]
        if self.apply_chat_template:
            query = query.replace(
                f"{ROLE_USER.capitalize()}: ",
                f"<|im_end|>\n<|im_start|>{ROLE_USER}\n",
            )
            query = query.replace(
                f"{ROLE_ASSISTANT.capitalize()}: ",
                f"<|im_end|>\n<|im_start|>{ROLE_ASSISTANT}\n",
            )
            query = f"<|im_start|>{ROLE_SYSTEM}\n" + query
            completion += "<|im_end|>\n"

        tokenizer.truncation_side = "right"
        query_encoding = tokenizer(
            query,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_tokens_query,
        )
        tokenizer.truncation_side = "left"

        if self.reduce_response_to_answer:
            start_tag = "<answer>"
            end_tag = "</answer>"

            start_pos = completion.find(start_tag)
            if start_pos == -1:
                start_pos = None
            end = completion.find(end_tag, start_pos)
            if end == -1:
                end_pos = None
            else:
                end_pos = end + len(end_tag)

            completion = completion[start_pos:end_pos]

        completion_encoding = tokenizer(
            completion,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_tokens_response,
        )

        # recovered = tokenizer.decode(completion_encoding["input_ids"])
        # assert recovered == completion

        tokens.extend(
            query_encoding["input_ids"] + completion_encoding["input_ids"]  # type: ignore
        )
        attn.extend(
            query_encoding["attention_mask"] + completion_encoding["attention_mask"]  # type: ignore
        )
        labels.extend(
            [IGNORE_INDEX] * len(query_encoding["input_ids"]) + completion_encoding["input_ids"]
        )  # type: ignore


class UltraFeedback(DataHandlerEvaluation):
    def __init__(
        self,
        max_tokens_query: Optional[int],
        max_tokens_response: Optional[int],
        apply_chat_template: bool,
        col_prompt_id: str = "prompt_id",
        col_response: Optional[str] = "chosen",
        **kwargs,
    ) -> None:
        super().__init__(
            max_tokens_query=max_tokens_query,
            max_tokens_response=max_tokens_response,
            apply_chat_template=apply_chat_template,
            **kwargs,
        )
        self.col_query = "prompt"
        self.col_query_id = col_prompt_id
        self.col_response = col_response

    def collate_fn_eval(
        self, data: List[str], tokenizer: ttu.PreTrainedTokenizer
    ) -> Dict[str, Any]:
        batch_queries = []
        for query in data:
            if self.apply_chat_template:
                query = tokenizer.apply_chat_template(
                    [{"role": ROLE_USER, "content": query}],
                    tokenize=False,
                    add_generation_prompt=self.add_generation_prompt,
                    chat_template=self.chat_template,
                )
            batch_queries.append(query)

        encoded = tokenizer(
            batch_queries,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            padding_side="left",
            max_length=self.max_tokens_query,
        )
        return {**encoded, "queries": batch_queries}

    def extend_collate(self, data):
        chosen = []
        rejected = []
        for d in data:
            chosen.append(d["chosen"])
            rejected.append(d["rejected"])
        return {"chosen": chosen, "rejected": rejected}

    def get_query_id(self, dataset: datasets.Dataset) -> List[str]:
        self._validate_key(key=self.col_query_id, dataset=dataset)
        return dataset[self.col_query_id]

    def get_prompt_plaintext(self, dataset: datasets.Dataset) -> List[str]:
        return self.get_query(dataset=dataset)

    def get_query(self, dataset: datasets.Dataset) -> List[str]:
        # self._validate_key(key=self.col_query, dataset=dataset)
        return [d[self.col_query] for d in dataset]  # type: ignore

    def get_response(self, dataset: datasets.Dataset) -> List[str]:
        if self.col_response is None:
            return ["N/A"] * len(dataset)
        self._validate_key(key=self.col_response, dataset=dataset)

        assistant_contents = [
            next(
                (msg["content"] for msg in entry if msg.get("role") == ROLE_ASSISTANT),
                None,
            )
            for entry in dataset[self.col_response]
        ]
        return assistant_contents  # type: ignore

    def add_scores_and_answers(
        self,
        data: pd.DataFrame,
        col_response: str,
        col_input: str,
        col_target: str,
        col_answer: str,
        col_score: str,
    ):
        import os

        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY environment variable not set. "
                "Please set it to use the reward model."
            )

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
        )

        data[col_answer] = None
        data[col_score] = None

        for index, row in data.iterrows():
            query = row[col_input]
            response = row[col_response]

            response = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-reward",
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ],
            )

            response_content = response.choices[0].message.content
            reward_pairs = [pair.split(":") for pair in response_content.split(",")]
            reward_dict = {attribute: float(score) for attribute, score in reward_pairs}
            reward = reward_dict["reward"]
            data.at[index, col_answer] = reward
            ref = reward if col_target == col_answer else row[col_target]
            win = int(reward > ref)
            data.at[index, col_score] = win
