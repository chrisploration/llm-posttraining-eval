# llm-posttraining-eval

Eval-driven posttraining and regression analysis for Mistral-7B. Implements a
complete pipeline: baseline evaluation, QLoRA fine-tuning, posttrained
evaluation, and automated comparison with regression detection.

## Overview

This project implements an end to end workflow for posttraining LLMs that is
easy to run, inspect, and extend. It is designed around four engineering goals:

- **Reproducible experiments** — configuration driven, deterministic seeds, explicit commands
- **Modular system design** — separated stages with minimal coupling
- **Clear evaluation workflows** — structured three axis evaluation with regression detection
- **Rapid iteration** — change experiments through config overrides, not code edits

## Three Axis Evaluation

| Axis | What it measures | Method |
|------|-----------------|--------|
| **Capability** | Basic arithmetic accuracy | Randomly generated addition prompts, scored by exact integer match |
| **Robustness** | Sensitivity to prompt wording | Same arithmetic tasks with perturbed instructions, delta between base and perturbed accuracy |
| **Safety** | Refusal behavior on harmful prompts | 24 harmful + 25 benign prompts, scored by refusal phrase heuristics |

## System Architecture

```
Synthetic Dataset Generation
            ↓
      Data Validation
            ↓
   Posttraining (QLoRA)
            ↓
     Three Axis Evaluation
            ↓
    Experiment Comparison
```

## Project Structure

```
llm-posttraining-eval/
├── src/
│   ├── train.py              # QLoRA fine-tuning pipeline
│   ├── eval/
│   │   ├── run_eval.py       # Three axis evaluation pipeline
│   │   ├── scoring.py        # Metric scoring utilities
│   │   ├── tasks.py          # Task generation utilities
│   │   └── probes.py         # Diagnostic probes
│   ├── compare.py            # Baseline vs posttrained comparison
│   ├── config.py             # YAML config loading and validation
│   ├── errors.py             # Project wide error hierarchy
│   ├── train_artifacts.py    # Shared I/O and metadata utilities
│   └── utils/
│       ├── config_utils.py   # Deep merge, YAML loading
│       └── logging_setup.py  # Logging configuration
├── configs/
│   ├── posttrain.yaml        # Training config (Mistral-7B, LoRA r=16)
│   ├── eval.yaml             # Evaluation config (400 prompts, 3 tasks)
│   └── overrides/            # Tier specific overrides (smoke, synth, extended)
├── scripts/
│   ├── generate_synthetic_dataset.py  # Deterministic training data generator
│   └── setup_runpod.sh               # One command RunPod pipeline
├── tests/                    # Unit and smoke tests (pytest)
├── data/
│   └── smoke_train.jsonl     # Smoke test fixture
├── Dockerfile                # Reproducible container build
├── pyproject.toml            # Package config, ruff, pytest, mypy
└── requirements.txt          # Pip dependencies
```

## Quick Start

```bash
# Install
git clone https://github.com/chrisploration/llm-posttraining-eval.git
cd llm-posttraining-eval
pip install -e .

# Generate training data (deterministic)
python3 -m scripts.generate_synthetic_dataset --num_examples 1000

# Run tests
python3 -m pytest tests/ -v
```

## Full Workflow

### 1. Baseline Evaluation

Evaluate the unmodified Mistral-7B-Instruct-v0.3 model:

```bash
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/baseline \
    --mode baseline
```

### 2. Posttraining (QLoRA)

Fine-tune with LoRA adapters on synthetic data:

```bash
python3 -m src.train \
    --config configs/posttrain.yaml \
    --override configs/overrides/posttrain_synth.yaml
```

### 3. Posttrained Evaluation

Evaluate the fine-tuned adapter checkpoint:

```bash
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/posttrained \
    --mode posttrained \
    --checkpoint checkpoints/post_v1 \
    --base_model mistralai/Mistral-7B-Instruct-v0.3
```

### 4. Compare Results

Generate a comparison report with regression detection:

```bash
python3 -m src.compare \
    --baseline results/baseline \
    --candidate results/posttrained \
    --format markdown \
    --output results/comparison.md \
    --fail-on-regression
```

## Configuration

The project uses a hierarchical YAML config system with deep-merge overrides:

```bash
# Base config only
python3 -m src.train --config configs/posttrain.yaml

# Base + override (override values take precedence)
python3 -m src.train --config configs/posttrain.yaml --override configs/overrides/posttrain_synth.yaml

# CLI overrides (highest precedence)
python3 -m src.train --config configs/posttrain.yaml --seed 123 --output_dir outputs/experiment_1
```

Available override tiers:

- `posttrain_smoke.yaml` — minimal run for testing (10 examples, no quantization)
- `posttrain_synth.yaml` — points training to generated synthetic data
- `eval_smoke.yaml` — fast evaluation (50 prompts) for quick sanity checks
- `eval_extended.yaml` — high volume evaluation (2000 prompts) for variance reduction (tighter confidence intervals)
- `eval_stress_robustness.yaml` — extended robustness testing with additional perturbation patterns

```bash
# Fast sanity check
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_smoke.yaml

# Extended eval for tighter confidence intervals
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_extended.yaml

# Deep robustness analysis
python3 -m src.eval.run_eval --config configs/eval.yaml --override configs/overrides/eval_stress_robustness.yaml
```

## Dataset Format

Training data is stored in JSONL format (chat format messages):

```json
{"prompt": "Explain overfitting in one sentence.", "completion": "Overfitting occurs when a model learns training-specific patterns that do not generalize well to new data."}
```

Synthetic dataset generation makes the project runnable without external datasets:

```bash
python3 -m scripts.generate_synthetic_dataset --num_examples 1000
```

## RunPod Deployment

On a RunPod GPU pod, the entire pipeline runs with one command:

```bash
bash scripts/setup_runpod.sh
```

This clones the repo, installs dependencies, generates training data, and runs all four
workflow steps automatically.

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a specific test file
python3 -m pytest tests/test_scoring.py -v
```

## Requirements

- Python >= 3.10
- PyTorch with CUDA (for GPU training/evaluation)
- See `requirements.txt` for full dependency list

## Limitations

- Synthetic data is simplistic relative to real world posttraining datasets
- Experiments are small scale (single GPU)
- No distributed training support yet
- Evaluation coverage is limited compared with production ML systems

## Future Work

- Preference learning or RLHF-style extensions
- Larger and more realistic datasets
- Distributed or multi-GPU training
- Richer evaluation benchmarks
- Experiment tracking integration

## License

MIT License