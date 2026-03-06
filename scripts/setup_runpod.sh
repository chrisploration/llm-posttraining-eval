#!/bin/bash
set -e

cd /workspace

if [ ! -d "llm-posttraining-eval" ]; then
    git clone https://github.com/chrisploration/llm-posttraining-eval llm-posttraining-eval
fi

cd llm-posttraining-eval
git pull origin main
pip install -e .

echo "Setup complete. Run training with:"
echo "Still to do: Insert CLI instructions for running this project..."