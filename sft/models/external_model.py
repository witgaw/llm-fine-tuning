import json
from enum import Enum
from os import path
from pprint import pp
from typing import Any, Dict, Sequence

import evaluate
import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from pyreft import (
    ReftConfig,
    ReftTrainerForCausalLM,
    ReftTrainerForSequenceClassification,
    get_reft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    metrics = ["accuracy", "recall", "precision", "f1"]  # List of metrics to return
    metric = {}
    for met in metrics:
        metric[met] = evaluate.load(met)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_res = {}
    for met in metrics:
        metric_res[met] = metric[met].compute(predictions=predictions, references=labels)[met]
    return metric_res


class BaseModelType(Enum):
    CausalLM = "causal_lm"
    SequenceClassification = "sequence_classification"


class InternventionConfig:
    def __init__(self, position="l1", nonstop=False, share_weights=False):
        self.position = position
        self.nonstop = nonstop
        self.share_weights = share_weights

    def set_num_interventions(self, reft_config: ReftConfig):
        if reft_config is not None:
            self.num_interventions = len(reft_config.representations)


class FineTuningConfig:
    def __init__(
        self,
        directory: str,
        base_model_type: BaseModelType = BaseModelType.SequenceClassification,
        base_model_name: str = "gpt2",
        saved_base_model_directory: str = None,
        dtype: torch.dtype = torch.float32,
        peft_config: LoraConfig = None,
        reft_config: ReftConfig = None,
        intervention_config: InternventionConfig = InternventionConfig(),
        optimise_all_base_model_layers=False,
        keep_optimising_additional_model_layers=None,
        num_labels: int = 2,
    ):
        self.base_model_name = base_model_name
        self.base_model_type = base_model_type
        self.saved_base_model_directory = saved_base_model_directory
        self.dtype = dtype
        self.peft_config = peft_config
        self.reft_config = reft_config
        self.intervention_config = intervention_config
        self.num_labels = num_labels
        if intervention_config is not None:
            self.intervention_config.set_num_interventions(self.reft_config)
        self.directory = directory
        self.reft = self.reft_config is not None
        self.peft = self.peft_config is not None
        self.additional_model_layers_trained = self.saved_base_model_directory is not None
        self.optimise_all_base_model_layers = optimise_all_base_model_layers
        if keep_optimising_additional_model_layers is not None:
            self.additional_model_layers_trained = not keep_optimising_additional_model_layers

        self.seq_cls_base = self.base_model_type == BaseModelType.SequenceClassification
        self.causal_lm_base = self.base_model_type == BaseModelType.CausalLM


def _get_base_model(config: FineTuningConfig, device):
    base_model = None
    load_from = (
        config.saved_base_model_directory
        if config.additional_model_layers_trained
        else config.base_model_name
    )
    if config.seq_cls_base:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            load_from, torch_dtype=config.dtype, device_map=device, num_labels=config.num_labels
        )
    elif config.causal_lm_base:
        base_model = AutoModelForCausalLM.from_pretrained(
            load_from, torch_dtype=config.dtype, device_map=device
        )
    else:
        raise ValueError("Unsupported model type (supporting SequenceClassification and CausalLM)")

    base_model.config.pad_token_id = base_model.config.eos_token_id

    for param in base_model.parameters():
        param.requires_grad = config.optimise_all_base_model_layers
    if config.seq_cls_base:
        for param in base_model.score.parameters():
            param.requires_grad = not config.additional_model_layers_trained
    if config.causal_lm_base:
        for param in base_model.lm_head.parameters():
            param.requires_grad = not config.additional_model_layers_trained

    return base_model


def get_fine_tuned_external_gpt2(config: FineTuningConfig, training_data, args, device) -> Any:
    base_model = _get_base_model(config=config, device=device)
    model = base_model
    peft_model = None
    reft_model = None
    if config.peft:
        peft_config = config.peft_config
        peft_model = get_peft_model(model, peft_config)
        model = peft_model
        if config.seq_cls_base:
            for param in peft_model.model.score.parameters():
                param.requires_grad = not config.additional_model_layers_trained
        if config.causal_lm_base:
            for param in peft_model.model.lm_head.parameters():
                param.requires_grad = not config.additional_model_layers_trained

    if config.reft:
        reft_model = get_reft_model(model, config.reft_config, disable_model_grads=True)
        reft_model.train(True)
        model = reft_model
        if not config.additional_model_layers_trained:
            if config.seq_cls_base:
                for param in reft_model.model.score.parameters():
                    param.requires_grad = True
            if config.causal_lm_base:
                for param in reft_model.model.lm_head.parameters():
                    param.requires_grad = True
        # you need to call this to re-enable lora grads!
        if config.peft:
            model.model.enable_adapter_layers()

    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        output_dir=config.directory,
        run_name="prototyping",
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_strategy="epoch",
        use_cpu=device.type == "cpu",
        save_strategy="no",
        report_to="none",
        # evaluation_strategy = "epoch", #To calculate metrics per epoch
    )

    # data_collator_fn = training_data.collate_fn_regular
    data_collator_fn = None
    if config.seq_cls_base:
        data_collator_fn = training_data.collate_fn_sequence_classification
    if config.causal_lm_base:
        if config.reft:
            data_collator_fn = DataCollatorForSeq2Seq(
                tokenizer=training_data.tokenizer,
                model=model,
                label_pad_token_id=-100,
                padding="longest",
            )
        else:
            data_collator_fn = training_data.collate_fn_causal_lm

    trainer_fn = Trainer
    if config.reft:
        if config.seq_cls_base:
            trainer_fn = ReftTrainerForSequenceClassification
        elif config.causal_lm_base:
            trainer_fn = ReftTrainerForCausalLM
        else:
            raise ValueError(f"Unrecognized base model type '{config.base_model_type}'")

    # TODO: explore if reft doesn't need it's special dataset type to function properly
    trainer = trainer_fn(
        model=model,
        args=training_args,
        train_dataset=training_data,
        # eval_dataset={"train" : training_data, "dev": dev_data},
        # compute_metrics=compute_metrics,
        data_collator=data_collator_fn,
    )

    modules = dict(model.named_modules())
    if args.verbose:
        pp(modules)

    trainable_parameters = sum(
        [p.shape.numel() if p.requires_grad else 0 for p in model.parameters()]
    )
    if config.peft or config.reft:
        model.print_trainable_parameters()
    else:
        print(f"trainable parameters: {trainable_parameters:,}")

    with open(f"{config.directory}/info.json", "w") as fp:
        dump = vars(args)
        dump["trainable_parameters"] = trainable_parameters
        dump["total_parameters"] = sum([p.shape.numel() for p in model.parameters()])
        dump["modules"] = list(modules.keys())
        json.dump(dump, fp, indent=4)

    results = trainer.train()

    if config.peft:
        peft_model.save_pretrained(f"{config.directory}/peft")
    if config.reft:
        reft_model.eval()
        reft_model.train(False)
        reft_model.set_device("cpu")
        reft_model.save(save_directory=f"{config.directory}/reft", include_model=True)
        reft_model.set_device(device)
    base_model.eval()
    model.eval()
    if not config.reft and not config.peft:
        trainer.save_model(config.directory)
    trainer.save_state()

    return FineTunedExternalGPT2(model=model, config=config)


def get_pretuned_external_gpt2(config: FineTuningConfig, args, device) -> Any:
    if not config.peft and not config.reft and config.saved_base_model_directory is None:
        if not path.isfile(f"{config.directory}/model.safetensors"):
            raise ValueError(
                f"Aborting: fine tuning configs not specified, 'saved_base_model_directory' not specified and no 'model.safetensors' file present in {args.directory}"
            )
        config.saved_base_model_directory = config.directory
    base_model = _get_base_model(config=config, device=device)
    base_model.eval()
    model = base_model
    peft_model = None
    reft_model = None
    if args.verbose:
        print(f"Loading model form '{config.directory}'")
    if config.peft:
        peft_model = PeftModel.from_pretrained(model, f"{config.directory}/peft", device=device)
        model = peft_model
        if args.verbose:
            print("Loaded saved Peft model")
    if config.reft:
        reft_model = get_reft_model(model, config.reft_config, disable_model_grads=False)
        reft_model.load_intervention(f"{config.directory}/reft", include_model=True)
        reft_model.set_device(device)
        reft_model.eval()
        reft_model.train(False)
        model = reft_model
        if args.verbose:
            print("Loaded saved Reft model")

    return FineTunedExternalGPT2(model, config)


class FineTunedExternalGPT2:
    def __init__(self, model, config: FineTuningConfig):
        self.model = model
        self.config: FineTuningConfig = config
        self.output_last_token = False

    def forward(self, input_ids, attention_mask):
        output = None
        if self.config.reft:
            # output = self.model({"input_ids": input_ids, "attention_mask": attention_mask}, output_hidden_states=self.output_last_token)[1]
            output = self.model({"input_ids": input_ids, "attention_mask": attention_mask})[1]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.output_last_token,
            )

        if self.output_last_token:
            return self._get_last_token(output["hidden_states"], attention_mask)

        return output["logits"]

    def generate(self, encoding, generation_config: GenerationConfig, device):
        if self.config.reft:
            internvention_locations = encoding["intervention_locations"]
            output = self.model.generate(
                {
                    "input_ids": encoding["input_ids"][None, :].to(device),
                    "attention_mask": encoding["attention_mask"][None, :].to(device),
                },
                unit_locations={"sources->base": (None, [[x] for x in internvention_locations])},
                intervene_on_prompt=True,
                generation_config=generation_config,
            )
            return output[1][0]
        output = self.model.generate(**encoding, generation_config=generation_config)
        return output[0]

    def hidden_state_to_token(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        last_hidden_state = hidden_states[-1]
        # Get the hidden state of the final token.
        last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
        last_token_representation = last_hidden_state[
            torch.arange(last_hidden_state.shape[0]), last_non_pad_idx
        ]

        embedding_weights = self.model.base_model.wte.weight

        last_token = torch.matmul(last_token_representation, embedding_weights.t())

        return last_token

    def __call__(self, *args, **kwds):
        return self.forward(*args)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()


class AdaptorReftDataCollator(object):
    """Collate examples for ReFT."""

    def __init__(self, tokenizer, data_collator):
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        return batch_inputs


class ReftDataCollator(object):
    """Collate examples for ReFT."""

    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.data_collator(instances)
        max_seq_length = batch_inputs["input_ids"].shape[-1]
        batch_inputs["intervention_locations"] = batch_inputs["intervention_locations"][
            ..., :max_seq_length
        ]
        return batch_inputs


# def make_all_positions_unsupervised_data_module(
#     tokenizer, model, inputs, low_rank,
#     num_interventions=1, nonstop=False,
# ):
#     """Make dataset and collator for un-supervised (or really, semi-supervised) fine-tuning."""
#     FULL_SUBSPACE = list(range(low_rank))
#     all_base_input_ids, all_intervention_locations, all_output_ids, all_subspaces = [], [], [], []
#     for i in range(len(inputs)):
#         _input = inputs[i]
#         # print(_input)

#         base_input = _input["text"]
#         if not nonstop:
#             base_input += tokenizer.eos_token

#         base_input_ids = tokenizer(
#             base_input, padding="max_length",max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
#         output_ids = base_input_ids.clone().detach()

#         all_base_input_ids.append(base_input_ids)
#         all_output_ids.append(output_ids)
#         all_subspaces.append([FULL_SUBSPACE] * num_interventions)
#         # all_intervention_locations.append([[0]] * num_interventions)

#     train_dataset = Dataset.from_dict({
#         "input_ids": all_base_input_ids,
#         "labels": all_output_ids,
#         # "intervention_locations": all_intervention_locations,
#         "subspaces": all_subspaces,
#     })

#     data_collator_fn = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         label_pad_token_id=-100,
#         padding="longest"
#     )
#     max_train_samples = 2000

#     if max_train_samples is not None:
#         max_train_samples = min(len(train_dataset), max_train_samples)
#         train_dataset = train_dataset.shuffle(seed=123)
#         train_dataset = train_dataset.select(range(max_train_samples))

#     data_collator = AdaptorReftDataCollator(tokenizer=tokenizer, data_collator=data_collator_fn)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
