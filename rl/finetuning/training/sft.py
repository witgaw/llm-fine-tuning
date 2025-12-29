import finetuning.config as config
import finetuning.data_handlers as data_handlers
import torch
from finetuning.training.base import Trainer


class SFT(Trainer):
    def __init__(self, config_env: config.Environment, config_training: config.Training):
        super().__init__(config_env=config_env, config_training=config_training)

    def set_reference_model(self, reference_model):
        pass

    def compute_loss(self, batch, model, tokenizer):
        tokens, attn, labels = (
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["labels"].to(self.device),
        )

        output = model(input_ids=tokens, attention_mask=attn)

        shift_logits = output.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = output.logits.shape[-1]

        loss = torch.nn.functional.cross_entropy(
            input=shift_logits.view(-1, vocab_size),
            target=shift_labels.view(-1),
            ignore_index=data_handlers.IGNORE_INDEX,
            reduction="mean",
        )

        return loss

    def _get_data_collator(self):
        if not isinstance(
            self.config_training.data_training.handler,
            data_handlers.DataHandlerTraining,
        ):
            raise RuntimeError(
                f"Expecting data collator for SFT trainer to be of type ' data_handlers.DataHandlerTraining', got '{type(config.data_training.handler)}'"
            )
        return self.config_training.data_training.handler.collate_fn_train_sft
