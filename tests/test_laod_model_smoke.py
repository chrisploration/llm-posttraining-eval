import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_artifacts import require_accelerate_if_needed

MODEL_ID ="mistralai/Mistral-7B-Instruct-v0.3"

def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_load_model_and_generate() -> None:
    _set_seeds(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None

    require_accelerate_if_needed(device_map)

    model_kwargs = {"device_map": device_map} if device_map is not None else {}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **model_kwargs
    )
    model.eval()

    prompt = "Write one sentence explaining what evaluation driven posttraining means."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_params = {
        "max_new_tokens": 60,
        # Deterministic decoding:
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_beams": 1,
        "pad_token_id": int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params)

    text = tokenizer.decode(outputs[0], skip_special_tokens = True)
    assert len(text) > len(prompt), "Model should generate text beyond the prompt"