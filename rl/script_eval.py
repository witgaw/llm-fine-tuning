import finetuning.config as config
import finetuning.data_handlers as data_handlers
import finetuning.evaluation as evaluation
import huggingface_hub
import settings

DEV_MODEL = config.Model(
    base_model="EleutherAI/pythia-70m", peft_model="witgaw/dev_pythia_verifier_sft"
)
MODEL = config.Model(base_model="Qwen/Qwen2.5-0.5B")
MODEL_SFT_FULL = config.Model(base_model="witgaw/qwen_verifier_sft_full", variant="SFT full")
# MODEL_SFT_LORA_1 = config.Model(base_model="Qwen/Qwen2.5-0.5B", peft_model="witgaw/qwen_verifier_sft_lora_1", variant="SFT LORA 1")
MODEL_SFT_LORA_1 = config.Model(base_model="witgaw/qwen_verifier_sft_lora_1", variant="SFT LORA 1")
MODEL_SFT_LORA_2 = config.Model(
    base_model="Qwen/Qwen2.5-0.5B",
    peft_model="witgaw/qwen_verifier_sft_lora_2",
    variant="SFT LORA 2",
)
MODEL_SFT_LORA_test = config.Model(
    base_model="Qwen/Qwen2.5-0.5B",
    peft_model="witgaw/qwen_verifier_sft_lora_1_steps_1000",
    variant="SFT LORA 1",
)

MODEL_SFT_FROM_TRL = config.Model(
    # base_model="./models/trl_output",
    base_model="witgaw/trl_output",
    variant="SFT TRL trained)",
)


def ultrafeedback():
    ultrafeedback = evaluation.Manager(
        env_cfg=config.Environment(),
        gen_cfg=config.Generation(),
        path_results="./experiments/{experiment_name}/predictions/ultrafeedback_dev_reduced.csv".format(
            experiment_name="dev_preferences"
        ),
        data_cfg=config.EvaluationData(
            path="./data/ultrafeedback",
            split=config.DataSplit.DEV,
            batch_size=1,
            reduce_to_n_examples=5,
            stream=False,
            shuffle=False,
            handler=data_handlers.UltraFeedback(
                max_tokens_query=settings.DEV_PER_TASK["preferences"]["max_query"],
                max_tokens_response=settings.DEV_PER_TASK["preferences"]["max_response"],
                apply_chat_template=True,
            ),
        ),
        use_vllm=False,
        role_annotated_query=False,
        preferred_device="cpu",
    )

    ultrafeedback.evaluate_from_config(model_config=DEV_MODEL, overwrite=True)


if __name__ == "__main__":
    huggingface_hub.login()  # Uses HF_TOKEN from environment
    # ultrafeedback()
