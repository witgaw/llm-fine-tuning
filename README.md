# LLM Fine-Tuning Experiments

A comprehensive repository demonstrating advanced fine-tuning techniques for large language models, including parameter-efficient methods (LoRA, ReFT) and reinforcement learning approaches (RLOO, DPO).

## Overview

This project showcases two main categories of fine-tuning experiments:

### Supervised Fine-Tuning (SFT)

Located in `sft/`, these experiments demonstrate:

1. **Propaganda Technique Detection** (SemEval-2020 Task 11)
   - Multi-class classification of propaganda techniques in text
   - Techniques: LoRA, ReFT (Representation Fine-Tuning)

2. **Claim Verification**
   - **LIAR**: 6-class fact-checking (pants-fire to true)
   - **FEVER**: 3-class claim verification (supports/refutes/not enough info)

3. **Fact-Check Generation**
   - Generate explanations for claim verdicts
   - Autoregressive generation with GPT-2

### Reinforcement Learning (RL)

Located in `rl/`, these experiments demonstrate:

1. **RLOO (Reinforcement Learning with Leave-One-Out)**
   - **FEVER**: Generate reasoning chains for claim verification
   - **GSM8K**: Solve grade school math word problems
   - Custom reward functions for task-specific objectives

2. **DPO (Direct Preference Optimization)**
   - **UltraFeedback**: Learn from preference pairs (chosen vs rejected)
   - **SmolTalk**: Warm-start with conversational data

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for package management

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-fine-tuning

# Install dependencies with uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   - WANDB_API_KEY
#   - HF_TOKEN
#   - NVIDIA_API_KEY (for DPO reward model)
```

### Download Datasets

```bash
# Download all datasets
uv run python download_all_datasets.py

# Or download specific subsets
uv run python download_all_datasets.py --sft-only
uv run python download_all_datasets.py --rl-only
```

## Project Structure

```
llm-fine-tuning/
├── sft/                          # Supervised fine-tuning
│   ├── models/                   # GPT-2 model implementations
│   ├── modules/                  # Attention, transformer layers
│   ├── datasets/                 # Dataset downloaders
│   ├── classifier.py             # Sentiment classification (reference)
│   ├── paraphrase_detection.py   # Paraphrase detection (to be adapted)
│   ├── sonnet_generation.py      # Generation task (to be adapted)
│   ├── config.py                 # Model configurations
│   ├── utils.py                  # Utilities
│   ├── lang_utils.py             # Language metrics
│   └── evaluation.py             # Evaluation utilities
├── rl/                           # Reinforcement learning
│   ├── finetuning/
│   │   ├── training/             # RLOO, DPO, SFT trainers
│   │   ├── evaluation/           # Reward functions
│   │   ├── data_handlers.py      # Dataset handling
│   │   └── config.py             # Training configurations
│   ├── script_train.py           # Main training script
│   ├── script_eval.py            # Evaluation script
│   ├── settings.py               # Experiment settings
│   └── tasks.py                  # Data preparation tasks
├── data/                         # Downloaded datasets
├── download_all_datasets.py      # Master download script
├── pyproject.toml                # Project dependencies
└── README.md                     # This file
```

## Usage

### Supervised Fine-Tuning

**Note**: The SFT tasks are currently being adapted for the new datasets. The code structure is in place but requires configuration updates.

```bash
# Example: Train propaganda detection model
cd sft
uv run python paraphrase_detection.py --use_gpu --ft_config=lora_baseline

# Example: Train claim verification on LIAR
uv run python classifier.py --train_on=liar --use_gpu
```

### Reinforcement Learning

```bash
# Train with RLOO on GSM8K
cd rl
uv run python script_train.py --dev-mode

# Evaluate on leaderboard
uv run python script_eval.py --checkpoint <path-to-checkpoint>
```

## Key Techniques

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)
- Adds trainable low-rank matrices to attention layers
- Drastically reduces trainable parameters
- Config examples: `r=2, alpha=16, dropout=0.1`

#### ReFT (Representation Fine-Tuning)
- Novel approach: intervene at specific representation layers
- Targets: Query/Key/Value outputs, block outputs, MLP layers
- Position strategies: first token (f1), last token (l1), etc.

### Reinforcement Learning Methods

#### RLOO (Leave-One-Out RL)
- Generate K completions per prompt
- Compute advantages using leave-one-out baseline
- Optimize: `advantage * log_prob + KL_penalty`
- Custom reward functions per task

#### DPO (Direct Preference Optimization)
- Train on paired (chosen, rejected) responses
- No explicit reward model needed during training
- Optimizes implicit reward via preference likelihood

## Datasets

All datasets are automatically downloaded from HuggingFace:

- **SemEval-2020 Task 11**: Propaganda technique detection
- **LIAR**: Political statement fact-checking (6 classes)
- **FEVER**: Claim verification with evidence (3 classes)
- **GSM8K**: Grade school math problems with reasoning
- **UltraFeedback**: Preference pairs for helpful responses
- **SmolTalk**: Conversational data for warm-start

## Evaluation Metrics

### Classification Tasks
- Accuracy, F1, Precision, Recall

### Generation Tasks
- CHRF score (character n-gram F-score)
- Syllables per line (for poetry)
- Line count and structure

### RL Tasks
- Task-specific rewards (e.g., correctness for math)
- KL divergence from reference model
- Length penalties (optional)

## TODO / Work in Progress

- [ ] Adapt SFT task scripts for new datasets (propaganda, LIAR, FEVER)
- [ ] Update configs for new classification tasks
- [ ] Implement fact-check generation task
- [ ] Add evaluation scripts for new tasks
- [ ] Extend RL experiments to FEVER reasoning

## Citation

This repository combines techniques from:
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- ReFT: [Wu et al., 2024](https://arxiv.org/abs/2404.03592)
- RLOO: [Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)
- DPO: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)

## License

See LICENSE file for details.
