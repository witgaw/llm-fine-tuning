import copy

import finetuning.config as config
import finetuning.data_handlers as data_handlers
import peft

MODEL = "Qwen/Qwen2.5-0.5B"
DEV_MODEL = "EleutherAI/pythia-70m"

MODEL_POST_SFT = "witgaw/qwen_verifier_sft_full"

OUTPUT_DIR = "./experiments"


PROD_PER_TASK = {
    "preferences": {
        "max_query": 260,
        "max_response": 426,
        "chat_template": True,
        "skip_system_messages": False,
        "max_assistant_messages": 1,
    },
    "verifier": {
        "max_query": 153,
        "max_response": 730,
        "chat_template": False,
        "answer_only": False,
    },
}

DEV_PER_TASK = {
    "preferences": {
        "max_query": 260,
        "max_response": 426,
        "chat_template": True,
        "skip_system_messages": False,
        "max_assistant_messages": 1,
    },
    # "verifier": {"max_query": 153, "max_response": 730, "chat_template": True},
    "verifier": {"max_query": 153, "max_response": 730, "chat_template": False},
}

DEV_ENV = config.Environment(preferred_device="cuda", report_wandb=False, use_vllm=True)
PROD_ENV = config.Environment(preferred_device="cuda", report_wandb=True, use_vllm=True)

PROD_GEN_OPT = config.Optimisation(
    steps_total=0,
    report_n_steps=1,
    eval_n_steps=1,
)

PROD_SFT_OPT = config.Optimisation(
    steps_total=1000,
    steps_warmup_fraction=0.1,
    target_scheduler="cosine",
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_accumulation_sub_steps=5,
    report_n_steps=10,
    eval_n_steps=200,
)

PROD_SFT_LORA = copy.deepcopy(PROD_SFT_OPT)
PROD_SFT_LORA.learning_rate = 5e-5


DEV_SFT_OPT = config.Optimisation(
    steps_total=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_n_steps=1,
    eval_n_steps=5,
)

DEV_RL_OPT = config.Optimisation(
    steps_total=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_n_steps=1,
    eval_n_steps=5,
)

PROD_WARMSTART = config.TrainingData(
    path="./data/warmstart",
    split=config.DataSplit.TRAIN,
    batch_size=4,
    stream=False,
    shuffle=False,
    handler=data_handlers.WarmStart(
        max_tokens_query=PROD_PER_TASK["verifier"]["max_query"],
        max_tokens_response=PROD_PER_TASK["verifier"]["max_response"],
        apply_chat_template=PROD_PER_TASK["verifier"]["chat_template"],
        reduce_response_to_answer=PROD_PER_TASK["verifier"]["answer_only"],
    ),
)

PROD_SMALTALK = config.TrainingData(
    path="./data/smoltalk",
    split=config.DataSplit.TRAIN,
    batch_size=2,
    stream=False,
    shuffle=False,
    handler=data_handlers.SmolTalk(
        max_tokens_query=PROD_PER_TASK["preferences"]["max_query"],
        max_tokens_response=PROD_PER_TASK["preferences"]["max_response"],
        apply_chat_template=PROD_PER_TASK["preferences"]["chat_template"],
        skip_system_messages=PROD_PER_TASK["preferences"]["skip_system_messages"],
        max_assistant_messages=PROD_PER_TASK["preferences"]["max_assistant_messages"],
    ),
)

PROD_ULTRAFREEDBACK_EVAL = config.EvaluationData(
    path="./data/ultrafeedback",
    split=config.DataSplit.DEV,
    reduce_to_n_examples=200,
    batch_size=1,
    stream=False,
    shuffle=False,
    handler=data_handlers.UltraFeedback(
        max_tokens_query=DEV_PER_TASK["preferences"]["max_query"],
        max_tokens_response=DEV_PER_TASK["preferences"]["max_response"],
        apply_chat_template=DEV_PER_TASK["preferences"]["chat_template"],
    ),
)

SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]

PREF_LEADERBOARD = config.EvaluationData(
    path="./data/ultrafeedback/ultrafeedback_heldout_prompt.json",
    split=None,
    batch_size=1,
    stream=False,
    shuffle=False,
    # reduce_to_n_examples=4,
    handler=data_handlers.UltraFeedback(
        max_tokens_query=PROD_PER_TASK["preferences"]["max_query"],
        max_tokens_response=PROD_PER_TASK["preferences"]["max_response"],
        apply_chat_template=PROD_PER_TASK["preferences"]["chat_template"],
        col_prompt_id="id",
        col_response=None,
    ),
)

PROD_LORA_1 = peft.LoraConfig(
    r=8,  # type: ignore
    lora_alpha=16,  # type: ignore
    lora_dropout=0.05,  # type: ignore
    task_type=peft.TaskType.CAUSAL_LM,
    target_modules=[  # type: ignore
        "q_proj",
        "v_proj",
    ],
)

PROD_LORA_2 = peft.LoraConfig(
    r=2,  # type: ignore
    lora_alpha=16,  # type: ignore
    lora_dropout=0.05,  # type: ignore
    task_type=peft.TaskType.CAUSAL_LM,
    target_modules=[  # type: ignore
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

verifier_lora_1 = config.Experiment(
    name="verifier",
    comment="LORA_1",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_LORA,
        # optimisation=DEV_SFT_OPT,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            outpath_save_locally="./models/qwen_verifier_sft_lora_1",
            output_path_save_to_hf_hub="witgaw/qwen_verifier_sft_lora_1",
        ),
        lora=PROD_LORA_1,
        data_training=PROD_WARMSTART,
        data_evaluation=None,  # TODO: Add FEVER/GSM8K evaluation
        eval_save_json=False,
        eval_save_model=False,
    ),
    rl=None,
)

verifier_lora_2 = config.Experiment(
    name="verifier",
    comment="LORA_2",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_LORA,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            outpath_save_locally="./models/qwen_verifier_sft_lora_2",
            output_path_save_to_hf_hub="witgaw/qwen_verifier_sft_lora_2",
        ),
        lora=PROD_LORA_2,
        data_training=PROD_WARMSTART,
        data_evaluation=None,  # TODO: Add FEVER/GSM8K evaluation
        eval_save_json=False,
        eval_save_model=False,
    ),
    rl=None,
)

verifier_full = config.Experiment(
    name="verifier",
    comment="FULL",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_OPT,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            outpath_save_locally="./models/qwen_verifier_sft_full",
            output_path_save_to_hf_hub=MODEL_POST_SFT,
        ),
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None,  # TODO: Add FEVER/GSM8K evaluation
        eval_save_json=False,
        eval_save_model=False,
    ),
    rl=None,
)


prod_verifier_gen_1 = config.Experiment(
    name="verifier",
    comment="basic, test-time",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=config.Model(
            base_model=MODEL_POST_SFT,
            special_tokens=SPECIAL_TOKENS,
        ),
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
    ),
    rl=None,
)

prod_verifier_rloo_1 = config.Experiment(
    name="verifier",
    comment="batch_size=2, accumulation_steps=2, k=4, kl_penalty=0, lr=1e-5",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=None,
    rl=config.RLOO(
        num_generations=4,
        add_length_penalty=False,
        optimisation=config.Optimisation(
            steps_total=50,
            gradient_accumulation_sub_steps=2,
            steps_warmup_fraction=0,
            learning_rate=1e-5,
            weight_decay=0.01,
            report_n_steps=10,
            eval_n_steps=10,
        ),
        model=config.Model(
            base_model=MODEL_POST_SFT,
            special_tokens=SPECIAL_TOKENS,
            # outpath_save_locally="./models/qwen_verifier_rloo_basic",
            # output_path_save_to_hf_hub="witgaw/qwen_verifier_rloo_basic",
        ),
        lora=None,
        # TODO: Replace with FEVER or GSM8K training data
        data_training=None,
        data_evaluation=None,  # TODO: Add FEVER/GSM8K evaluation
    ),
)

prod_verifier_rloo_2 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_2.rl.kl_penalty = 0.01  # type: ignore
prod_verifier_rloo_2.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0.01, lr=1e-5"

prod_verifier_rloo_3 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_3.rl.kl_penalty = 0.1  # type: ignore
prod_verifier_rloo_3.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0.1, lr=1e-5"

prod_verifier_rloo_4 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_4.rl.kl_penalty = 0.2  # type: ignore
prod_verifier_rloo_4.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0.2, lr=1e-5"


prod_verifier_rloo_5 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_5.rl.kl_penalty = 0.05  # type: ignore
prod_verifier_rloo_5.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0.05, lr=1e-5"

prod_verifier_rloo_6 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_6.rl.add_length_penalty = True  # type: ignore
prod_verifier_rloo_6.comment = (
    "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0, with length penalty"
)

prod_verifier_rloo_7 = copy.deepcopy(prod_verifier_rloo_3)
prod_verifier_rloo_7.rl.add_length_penalty = True  # type: ignore
prod_verifier_rloo_7.comment = (
    "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0.1, lr=1e-5, with length penalty"
)

prod_verifier_rloo_8 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_8.rl.optimisation.learning_rate = 2e-5
prod_verifier_rloo_8.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0, lr=2e-5"


prod_verifier_rloo_9 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_9.rl.optimisation.learning_rate = 5e-6
prod_verifier_rloo_9.comment = "batch_size=2, accumulation_steps=2, k=4, kl_penalty=0, lr=1e-6"

prod_verifier_rloo_10 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_10.rl.num_generations = 6
prod_verifier_rloo_10.comment = "batch_size=2, accumulation_steps=2, k=6, kl_penalty=0, lr=1e-5"

prod_verifier_rloo_11 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_11.rl.num_generations = 8
prod_verifier_rloo_11.comment = "batch_size=2, accumulation_steps=2, k=8, kl_penalty=0, lr=1e-5"

prod_verifier_rloo_target1 = copy.deepcopy(prod_verifier_rloo_1)
prod_verifier_rloo_target1.rl.model.outpath_save_locally = "./models/qwen_verifier_rloo"
prod_verifier_rloo_target1.rl.model.output_path_save_to_hf_hub = "witgaw/qwen_verifier_rloo"
prod_verifier_rloo_target1.rl.eval_save_model = True


# PREFERENCE

preferences_full = config.Experiment(
    name="preferences",
    comment="FULL",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_OPT,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            # outpath_save_locally="./models/qwen_preferences_sft",
            output_path_save_to_hf_hub="witgaw/qwen_preferences_sft_full",
        ),
        lora=None,
        data_training=PROD_SMALTALK,
        data_evaluation=PROD_ULTRAFREEDBACK_EVAL,
    ),
    rl=None,
)

preferences_lora_1 = config.Experiment(
    name="preferences",
    comment="LORA_1",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_LORA,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            # outpath_save_locally="./models/qwen_preferences_sft",
            output_path_save_to_hf_hub="witgaw/qwen_preferences_sft_lora_1",
        ),
        lora=PROD_LORA_1,
        data_training=PROD_SMALTALK,
        data_evaluation=PROD_ULTRAFREEDBACK_EVAL,
    ),
    rl=None,
)

preferences_lora_2 = config.Experiment(
    name="preferences",
    comment="LORA_2",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_SFT_LORA,
        model=config.Model(
            base_model=MODEL,
            special_tokens=SPECIAL_TOKENS,
            # outpath_save_locally="./models/qwen_preferences_sft",
            output_path_save_to_hf_hub="witgaw/qwen_preferences_sft_lora_2",
        ),
        lora=PROD_LORA_2,
        data_training=PROD_SMALTALK,
        data_evaluation=PROD_ULTRAFREEDBACK_EVAL,
    ),
    rl=None,
)

PREFERENCES_POST_SFT_MODEL = "witgaw/qwen_preferences_sft_lora_2"
PREFERENCES_POST_DPO_MODEL = config.Model(
    base_model=MODEL,
    peft_model="witgaw/dev_pythia_preferences_dpo_1",
    special_tokens=SPECIAL_TOKENS,
)


PROD_ULTRAFREEDBACK_TRAIN = config.TrainingData(
    path="./data/ultrafeedback",
    split=config.DataSplit.TRAIN,
    batch_size=2,
    stream=False,
    shuffle=False,
    handler=data_handlers.UltraFeedback(
        max_tokens_query=DEV_PER_TASK["preferences"]["max_query"],
        max_tokens_response=DEV_PER_TASK["preferences"]["max_response"],
        apply_chat_template=DEV_PER_TASK["preferences"]["chat_template"],
    ),
)

preferences_dpo_1b = config.Experiment(
    name="preferences",
    comment="beta=0.01 (from full model)",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=None,
    rl=config.DPO(
        beta=0.01,
        optimisation=config.Optimisation(
            steps_total=200,
            gradient_accumulation_sub_steps=2,
            steps_warmup_fraction=0,
            learning_rate=1e-5,
            weight_decay=0.01,
            report_n_steps=10,
            eval_n_steps=10,
        ),
        model=config.Model(
            base_model="witgaw/qwen_preferences_sft_full",
            # base_model=MODEL,
            # peft_model=PREFERENCES_POST_SFT_MODEL,
            special_tokens=SPECIAL_TOKENS,
            # outpath_save_locally="./models/dev_pythia_preferences_dpo_1_v2",
            output_path_save_to_hf_hub="witgaw/dev_pythia_preferences_dpo_1_v2",
        ),
        lora=None,
        data_training=PROD_ULTRAFREEDBACK_TRAIN,
        data_evaluation=PROD_ULTRAFREEDBACK_EVAL,
        eval_save_model=True,
    ),
)

preferences_dpo_1 = config.Experiment(
    name="preferences",
    comment="beta=0.01",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=None,
    rl=config.DPO(
        beta=0.01,
        optimisation=config.Optimisation(
            steps_total=50,
            gradient_accumulation_sub_steps=2,
            steps_warmup_fraction=0,
            learning_rate=1e-5,
            weight_decay=0.01,
            report_n_steps=10,
            eval_n_steps=10,
        ),
        model=config.Model(
            base_model=MODEL,
            peft_model=PREFERENCES_POST_SFT_MODEL,
            special_tokens=SPECIAL_TOKENS,
            outpath_save_locally="./models/dev_pythia_preferences_dpo_1",
            output_path_save_to_hf_hub="witgaw/dev_pythia_preferences_dpo_1",
        ),
        lora=None,
        data_training=PROD_ULTRAFREEDBACK_TRAIN,
        data_evaluation=PROD_ULTRAFREEDBACK_EVAL,
    ),
)

preferences_dpo_2 = copy.deepcopy(preferences_dpo_1)
preferences_dpo_2.comment = "beta=0.1"
preferences_dpo_2.rl.beta = 0.1  # type: ignore
preferences_dpo_2.rl.model.outpath_save_locally = "./models/dev_pythia_preferences_dpo_2"  # type: ignore
preferences_dpo_2.rl.model.output_path_save_to_hf_hub = "witgaw/dev_pythia_preferences_dpo_2"  # type: ignore
preferences_dpo_2.rl.eval_save_model = True  # type: ignore

preferences_dpo_3 = copy.deepcopy(preferences_dpo_1)
preferences_dpo_3.comment = "beta=0.5"
preferences_dpo_3.rl.beta = 0.5  # type: ignore
preferences_dpo_3.rl.model.outpath_save_locally = "./models/dev_pythia_preferences_dpo_3"  # type: ignore
preferences_dpo_3.rl.model.output_path_save_to_hf_hub = "witgaw/dev_pythia_preferences_dpo_3"  # type: ignore
preferences_dpo_3.rl.eval_save_model = True  # type: ignore

# Leaderboard submissions

preferences_gen_leaderboard_post_dpo = config.Experiment(
    name="preferences_inference",
    comment="leaderboard_dpo",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=PREFERENCES_POST_DPO_MODEL,
        lora=None,
        data_training=PROD_SMALTALK,
        data_evaluation=PREF_LEADERBOARD,
        eval_save_json=True,
    ),
    rl=None,
)

VERIFIER_POST_RLOO_MODEL = config.Model(base_model="witgaw/qwen_verifier_rloo")

verifier_gen_leaderboard_post_dpo = config.Experiment(
    name="verifier_inference",
    comment="leaderboard_rloo",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=VERIFIER_LEADERBOARD,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_temp_1 = config.Experiment(
    name="verifier_inference",
    comment="temp=0.00",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_temp_2 = config.Experiment(
    name="verifier_inference",
    comment="temp=0.50",
    env=PROD_ENV,
    gen=config.Generation(temperature=0.5),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_temp_3 = config.Experiment(
    name="verifier_inference",
    comment="temp=0.75",
    env=PROD_ENV,
    gen=config.Generation(temperature=0.75),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_temp_4 = config.Experiment(
    name="verifier_inference",
    comment="temp=0.90",
    env=PROD_ENV,
    gen=config.Generation(temperature=0.9),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)


verifier_extension_temp_5 = config.Experiment(
    name="verifier_inference",
    comment="temp=1.00",
    env=PROD_ENV,
    gen=config.Generation(temperature=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_temp_6 = config.Experiment(
    name="verifier_inference",
    comment="temp=1.10",
    env=PROD_ENV,
    gen=config.Generation(temperature=1.1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_beam_1 = config.Experiment(
    name="verifier_inference",
    comment="beams=1",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)


verifier_extension_beam_2 = config.Experiment(
    name="verifier_inference",
    comment="beams=2",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=2),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_beam_3 = config.Experiment(
    name="verifier_inference",
    comment="beams=4",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=4),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_beam_4 = config.Experiment(
    name="verifier_inference",
    comment="beams=8",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=8),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)


verifier_extension_generation_length_1 = config.Experiment(
    name="verifier_inference",
    comment="max response = 730",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

half_response = copy.deepcopy(None  # TODO: Add FEVER/GSM8K evaluation)
half_response.handler.max_tokens_response = int(half_response.handler.max_tokens_response / 2)

verifier_extension_generation_length_2 = config.Experiment(
    name="verifier_inference",
    comment="max response = 365",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=half_response,
        eval_save_json=True,
    ),
    rl=None,
)

double_response = copy.deepcopy(None  # TODO: Add FEVER/GSM8K evaluation)
double_response.handler.max_tokens_response = half_response.handler.max_tokens_response * 2

verifier_extension_generation_length_3 = config.Experiment(
    name="verifier_inference",
    comment="max response = 1460",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=double_response,
        eval_save_json=True,
    ),
    rl=None,
)

quatro_response = copy.deepcopy(None  # TODO: Add FEVER/GSM8K evaluation)
quatro_response.handler.max_tokens_response = half_response.handler.max_tokens_response * 4

verifier_extension_generation_length_4 = config.Experiment(
    name="verifier_inference",
    comment="max response = 2920",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=quatro_response,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_k_1 = config.Experiment(
    name="verifier_inference",
    comment="top_k=0 (disabled)",
    env=PROD_ENV,
    gen=config.Generation(top_k=0),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_k_2 = config.Experiment(
    name="verifier_inference",
    comment="top_k=10",
    env=PROD_ENV,
    gen=config.Generation(top_k=10),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_k_3 = config.Experiment(
    name="verifier_inference",
    comment="top_k=50",
    env=PROD_ENV,
    gen=config.Generation(top_k=50),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_k_4 = config.Experiment(
    name="verifier_inference",
    comment="top_k=100",
    env=PROD_ENV,
    gen=config.Generation(top_k=100),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_p_1 = config.Experiment(
    name="verifier_inference",
    comment="top_p=1.0 (disabled)",
    env=PROD_ENV,
    gen=config.Generation(),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_p_2 = config.Experiment(
    name="verifier_inference",
    comment="top_p=0.80",
    env=PROD_ENV,
    gen=config.Generation(top_p=0.8),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_p_3 = config.Experiment(
    name="verifier_inference",
    comment="top_p=0.90",
    env=PROD_ENV,
    gen=config.Generation(top_p=0.9),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_top_p_4 = config.Experiment(
    name="verifier_inference",
    comment="top_p=0.95",
    env=PROD_ENV,
    gen=config.Generation(top_p=0.95),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_repetition_1 = config.Experiment(
    name="verifier_inference",
    comment="rep. pen. = 1 (disabled)",
    env=PROD_ENV,
    gen=config.Generation(repetition_penalty=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_repetition_2 = config.Experiment(
    name="verifier_inference",
    comment="rep. pen. = 1.05",
    env=PROD_ENV,
    gen=config.Generation(repetition_penalty=1.05),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_repetition_3 = config.Experiment(
    name="verifier_inference",
    comment="rep. pen. = 1.1",
    env=PROD_ENV,
    gen=config.Generation(repetition_penalty=1.1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_repetition_4 = config.Experiment(
    name="verifier_inference",
    comment="rep. pen. = 1.2",
    env=PROD_ENV,
    gen=config.Generation(repetition_penalty=1.2),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_combined_1 = config.Experiment(
    name="verifier_inference",
    comment="beams=8",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=8),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=double_response,
        eval_save_json=True,
    ),
    rl=None,
)

# verifier use
verifier_extension_verifier_1 = config.Experiment(
    name="verifier_inference",
    comment="beams=8, verifier threshold=0.1",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=8, verifier_threshold=0.1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_verifier_2 = config.Experiment(
    name="verifier_inference",
    comment="beams=8, verifier threshold=1",
    env=PROD_ENV,
    gen=config.Generation(beam_search=True, beam_n=8, verifier_threshold=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_verifier_3 = config.Experiment(
    name="verifier_inference",
    comment="samples=8, verifier threshold=0.1",
    env=PROD_ENV,
    gen=config.Generation(temperature=1, n_generations=8, verifier_threshold=0.1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_verifier_4 = config.Experiment(
    name="verifier_inference",
    comment="samples=8, verifier threshold=1",
    env=PROD_ENV,
    gen=config.Generation(temperature=1, n_generations=8, verifier_threshold=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_verifier_4b = config.Experiment(
    name="verifier_inference",
    comment="samples=8, verifier threshold=1, secondary threshold=0.1",
    env=PROD_ENV,
    gen=config.Generation(
        temperature=1, n_generations=8, verifier_threshold=1, verifier_threshold_secondary=0.1
    ),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)


verifier_extension_verifier_5 = config.Experiment(
    name="verifier_inference",
    comment="samples=16, verifier threshold=1, secondary threshold=0",
    env=PROD_ENV,
    gen=config.Generation(
        temperature=1, n_generations=16, verifier_threshold=1, verifier_threshold_secondary=0
    ),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_verifier_6 = config.Experiment(
    name="verifier_inference",
    comment="samples=16, verifier threshold=1, secondary threshold=0.1",
    env=PROD_ENV,
    gen=config.Generation(
        temperature=1, n_generations=16, verifier_threshold=1, verifier_threshold_secondary=0.1
    ),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=None  # TODO: Add FEVER/GSM8K evaluation,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_leaderboard = config.Experiment(
    name="verifier_inference",
    comment="samples=8, verifier threshold=1",
    env=PROD_ENV,
    gen=config.Generation(temperature=1, n_generations=8, verifier_threshold=1),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=VERIFIER_LEADERBOARD,
        eval_save_json=True,
    ),
    rl=None,
)

verifier_extension_leaderboard_2 = config.Experiment(
    name="verifier_inference",
    comment="samples=16, verifier threshold=1, secondary threshold=0.1",
    env=PROD_ENV,
    gen=config.Generation(
        temperature=1, n_generations=16, verifier_threshold=1, verifier_threshold_secondary=0.1
    ),
    sft=config.SFT(
        optimisation=PROD_GEN_OPT,
        model=VERIFIER_POST_RLOO_MODEL,
        lora=None,
        data_training=PROD_WARMSTART,
        data_evaluation=VERIFIER_LEADERBOARD,
        eval_save_json=True,
    ),
    rl=None,
)
