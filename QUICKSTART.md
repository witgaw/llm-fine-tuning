# Quick Start Guide

This guide helps you get started with the LLM fine-tuning experiments.

## Setup

### 1. Install Dependencies

```bash
cd llm-fine-tuning
uv sync
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# - WANDB_API_KEY (for experiment tracking)
# - HF_TOKEN (for HuggingFace models/datasets)
# - NVIDIA_API_KEY (for DPO reward model, optional)
```

### 3. Download Datasets

```bash
# Download all datasets
uv run python download_all_datasets.py

# Or download specific subsets
uv run python download_all_datasets.py --sft-only
uv run python download_all_datasets.py --rl-only
```

## Testing Dataset Downloaders

Test that datasets download correctly:

```bash
# Test SemEval download
cd sft/datasets
uv run python download_semeval.py

# Test LIAR download
uv run python download_liar.py

# Test FEVER download
uv run python download_fever.py

# Test GSM8K download
uv run python download_gsm8k.py
```

Expected output:
- CSV/TSV files in `data/{dataset_name}/`
- train.csv, dev.csv, test.csv for each dataset
- Summary statistics printed to console

## Running Experiments

### Supervised Fine-Tuning (SFT)

#### Propaganda Detection

```bash
cd sft
uv run python propaganda_detection.py --use_gpu
```

**Note**: This script is a starter template. Full implementation requires:
- Model initialization with LoRA/ReFT
- Complete training loop
- Evaluation metrics

#### Claim Verification

```bash
# LIAR (6-class fact-checking)
cd sft
uv run python claim_verification.py --dataset liar --use_gpu

# FEVER (3-class claim verification)
uv run python claim_verification.py --dataset fever --use_gpu
```

#### Fact-Check Generation

```bash
cd sft
uv run python factcheck_generation.py --dataset liar --use_gpu
```

### Reinforcement Learning (RL)

#### RLOO on GSM8K

```bash
cd rl
uv run python script_train.py --dev-mode
```

**Note**: You'll need to update `settings.py` to configure experiments for:
- GSM8K math reasoning
- FEVER claim reasoning

#### DPO on UltraFeedback

The existing DPO experiments should work as-is:

```bash
cd rl
# Edit settings.py to uncomment DPO experiments
uv run python script_train.py
```

## Project Status

### ✅ Completed

- [x] Repository structure
- [x] Dataset downloaders (SemEval, LIAR, FEVER, GSM8K)
- [x] Starter scripts for SFT tasks
- [x] Reward functions for RL tasks (FEVER, GSM8K)
- [x] API key environment variables
- [x] Documentation

### ⚠️ Requires Additional Work

The following tasks need completion for full functionality:

#### SFT Tasks

1. **Model Initialization**
   - Create model classes for multi-class classification tasks
   - Configure tokenizers for each task
   - Set up LoRA/ReFT configurations

2. **Training Loops**
   - Complete training implementation in task scripts
   - Add checkpoint saving/loading
   - Implement evaluation on dev/test sets

3. **Dataset Classes**
   - Create PyTorch Dataset classes in `datasets_internal.py`
   - Add proper preprocessing for each dataset
   - Implement collate functions for batching

#### RL Tasks

1. **Data Handlers**
   - Update `rl/finetuning/data_handlers.py` for FEVER
   - Add GSM8K data handler
   - Configure chat templates for reasoning tasks

2. **Experiment Settings**
   - Add FEVER/GSM8K experiments to `rl/settings.py`
   - Configure hyperparameters (k, KL penalty, etc.)
   - Set up warmstart experiments

3. **Evaluation**
   - Integrate `rewards_fever.py` and `rewards_gsm8k.py`
   - Add leaderboard evaluation scripts
   - Implement metric tracking

## Troubleshooting

### Dependency Conflicts

If you encounter dependency conflicts:
- `pyreft==0.1.0` requires `transformers==4.45.1`
- `vllm` has been removed to avoid conflicts
- For inference optimization, consider adding vllm as optional dependency

### Dataset Download Issues

If datasets fail to download:
- Check internet connection
- Verify HuggingFace access (some datasets require agreement to terms)
- Try downloading individual datasets instead of all at once

### Import Errors

If you get import errors:
```bash
# Make sure you're running via uv
uv run python script.py

# Not just:
python script.py
```

## Next Steps

1. **Test dataset downloads** - Verify all datasets download correctly
2. **Complete one SFT task** - Start with propaganda detection or LIAR
3. **Configure RL experiments** - Set up GSM8K or FEVER RLOO
4. **Add evaluation metrics** - Implement proper eval for each task
5. **Run experiments** - Train models and track with wandb

## Getting Help

For implementation details, refer to:
- SFT: See task scripts (`propaganda_detection.py`, `claim_verification.py`, `factcheck_generation.py`)
- RL: See `script_train.py`, `settings.py`, `data_handlers.py`
