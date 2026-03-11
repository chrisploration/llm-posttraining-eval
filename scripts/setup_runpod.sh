#!/bin/bash
set -e

cd /workspace

# Setup
if [ ! -d "llm-posttraining-eval" ]; then
    git clone https://github.com/chrisploration/llm-posttraining-eval llm-posttraining-eval
fi

cd llm-posttraining-eval
git pull origin main
pip install --upgrade setuptools pip
pip install -e .

# Verify environment
python3 -m pytest tests/ -v

# Generate training data
python3 -m scripts.generate_synthetic_dataset --num_examples 1000 --force
echo "Generated training data: data/synthetic/train.jsonl (1000 examples)"

echo "=== Setup complete. Running pipeline... ==="

# Step 1: Baseline evaluation
echo ""
echo "=== Step 1/4: Baseline evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/baseline \
    --mode baseline


# Step 2: Posttraining
echo ""
echo "=== Step 2/4: Posttraining ==="
python3 -m src.train \
    --config configs/posttrain.yaml \
    --override configs/overrides/posttrain_synth.yaml


# Step 3: Posttrained evaluation
echo ""
echo "=== Step 3/4: Posttrained evaluation ==="
python3 -m src.eval.run_eval \
    --config configs/eval.yaml \
    --output_dir results/posttrained \
    --mode posttrained \
    --checkpoint checkpoints/post_v1 \
    --base_model mistralai/Mistral-7B-Instruct-v0.3


# Step 4: Compare results
echo ""
echo "=== Step 4/4: Compare baseline vs post-trained ==="
python3 -m src.compare \
    --baseline results/baseline \
    --candidate results/posttrained \
    --format markdown \
    --output results/comparison.md \
    --fail-on-regression

echo ""
echo "=== Pipeline complete ==="
echo "Comparison report: results/comparison.md"