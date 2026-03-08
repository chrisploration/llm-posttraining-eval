FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/llm-posttraining-eval

# Install dependencies first
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY scripts/ scripts/
COPY tests/ tests/

# Install project in editable mode
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import src; print('Package import OK')" && \
    python -c "import torch; print(f'CUDA build: {torch.cuda.is_available()}')" && \
    python -c "import bitsandbytes; print('bitsandbytes OK')"

CMD ["bash"]