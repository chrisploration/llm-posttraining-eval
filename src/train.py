import argparse
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from src.config import PostTrainConfig, load_config, load_config_from_dict
from src.errors import CheckpointError, ConfigError, DataError
from src.train_artifacts import (
    append_jsonl,
    build_train_meta,
    guard_output_dir_empty,
    require_accelerate_if_needed,
    write_json,
    write_yaml,
)
from src.utils.config_utils import deep_merge
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

def build_model(cfg: PostTrainConfig) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load the base model with optional 4-bit quantization and apply LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_id, use_fast=True)

    added_tokens = 0

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device_map = "auto" if torch.cuda.is_available() else None
    require_accelerate_if_needed(device_map)
    model_kwargs = {"device_map": device_map} if device_map is not None else {}

    if cfg.load_in_4bit:
        # Fail fast with a clear error if bitsandbytes is missing.
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise ConfigError(
                "load_in_4bit=true requires bitsandbytes. Install with: pip install bitsandbytes"
            ) from e

        model_kwargs.update({"load_in_4bit": True, "torch_dtype": torch.float16})
    else:
        model_kwargs.update(
            {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.base_id, **model_kwargs)

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "config") and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    logger.info("Trainable params: %s | All params: %s | Trainable%%: %.4f", trainable_params, all_params, trainable_pct)

    return model, tokenizer


def load_data(cfg: PostTrainConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Load chat format JSONL training data and convert to plain text using the tokenizer's chat template."""
    if not os.path.exists(cfg.train_file):
        raise DataError(f"Training file not found: {cfg.train_file}")

    ds = load_dataset("json", data_files={"train": cfg.train_file})["train"]
    if "messages" not in ds.column_names:
        raise DataError("Training JSONL must contain a 'messages' field per line for v1.")

    def messages_to_text(messages):
        # Normalize
        norm = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if role and content:
                norm.append({"role": role, "content": content})
        if not norm:
            raise DataError("Empty/invalid messages in one training example")

        # Prefer model's chat template if available
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(norm, tokenize=False)

        # Fallback: explicit role tags
        out = []
        for m in norm:
            r = m["role"].lower()
            if r == "user":
                out.append(f"User: {m['content']}")
            elif r == "assistant":
                out.append(f"Assistant: {m['content']}")
            else:
                out.append(f"{m['role']}: {m['content']}")
        return "\n".join(out).strip() + "\n"

    def map_fn(ex):
        return {"text": messages_to_text(ex["messages"])}

    ds = ds.map(map_fn, remove_columns=ds.column_names)
    return ds




class JsonlLoggerCallback(TrainerCallback):
    """Trainer callback that appends training metrics to a JSONL log file after each logging step."""
    def __init__(self, path: str):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": int(state.global_step)}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
        append_jsonl(self.path, row)



def run_training(
    cfg: PostTrainConfig,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    resume_from: str | None
) -> None:
    """Run the SFTTrainer loop, save the adapter checkpoint, and validate artifacts."""

    os.makedirs(cfg.output_dir, exist_ok=True)

    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to=[]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=args,
        packing=False,  # v1: predictable behavior
        callbacks=[JsonlLoggerCallback(log_path)]
    )

    logger.info("Starting training...")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save adapter weights (not full base model) + tokenizer.
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    logger.info("Saved checkpoint to %s", cfg.output_dir)


    # Validate the saved adapter checkpoint with a lightweight artifact check
    adapter_config_path = os.path.join(cfg.output_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise CheckpointError(f"Missing adapter_config.json after save: {adapter_config_path}")

    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    if not isinstance(adapter_cfg, dict):
        raise CheckpointError(f"adapter_config.json must contain a JSON object: {adapter_config_path}")

    required_keys = ["peft_type", "base_model_name_or_path"]
    missing = [k for k in required_keys if k not in adapter_cfg]
    if missing:
        raise CheckpointError(f"adapter_config.json missing required keys {missing}: {adapter_config_path}")

    logger.info("Checkpoint validation passed: %s", cfg.output_dir)



def run_training_entry(*, config_path: str, override_paths: Sequence[str], output_dir: str | None, seed: int | None, resume_from: str | None) -> None:
    """End to end training entry point: load config, build model, and run training."""

    cfg, cfg_snapshot_raw = load_config(config_path, override_paths=override_paths)

    cli_ov: dict[str, Any] = {}
    if output_dir is not None:
        cli_ov.setdefault("output", {})["output_dir"] = output_dir
    if seed is not None:
        cli_ov["seed"] = int(seed)
    if cli_ov:
        cfg_snapshot_raw = dict(cfg_snapshot_raw)
        cfg_snapshot_raw = deep_merge(cfg_snapshot_raw, cli_ov)
        cfg = load_config_from_dict(cfg_snapshot_raw)

    set_seed(cfg.seed)

    if resume_from is None:
        guard_output_dir_empty(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    write_yaml(os.path.join(cfg.output_dir, "config_snapshot.yaml"), cfg_snapshot_raw)

    resolved_view = dict(cfg_snapshot_raw)
    resolved_view["override_paths"] = list(override_paths)
    resolved_view["resolved"] = cfg.to_dict()
    write_yaml(os.path.join(cfg.output_dir, "config_resolved.yaml"), resolved_view)

    model, tokenizer = build_model(cfg)

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mem_gb = free_bytes / 1e9
        total_mem_gb = total_bytes / 1e9

        logger.info("GPU memory available after model load: %.1fGB free / %.1fGB total", free_mem_gb, total_mem_gb)

        if free_mem_gb < 2.0:
            raise ConfigError(
                f"Only {free_mem_gb:.1f}GB free GPU memory after model load. Need at least 2.0GB."
            )

    train_dataset = load_data(cfg, tokenizer)

    meta = build_train_meta(
        output_dir=cfg.output_dir,
        cfg_dict=cfg_snapshot_raw,
        dataset_size_used=len(train_dataset),
        config_path=config_path,
        override_paths=override_paths
    )
    write_json(os.path.join(cfg.output_dir, "meta.json"), meta)

    run_training(cfg, model, tokenizer, train_dataset, resume_from)



def main() -> None:
    """CLI entry point for posttraining."""
    setup_logging()
    ap = argparse.ArgumentParser(description="Run post-training from a YAML config.")
    ap.add_argument("--config",
                    default="configs/posttrain.yaml",
                    help="Path to post-training YAML."
                    )
    ap.add_argument("--override",
                    action="append",
                    default=[],
                    help="Override YAML (repeatable). Applied in order."
                    )
    ap.add_argument("--output_dir",
                    default=None,
                    help="Override output.output_dir from config.")
    ap.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Override seed from config.")
    ap.add_argument("--resume",
                    default=None,
                    help="Path to a HF Trainer checkpoint dir to resume from.")
    args = ap.parse_args()

    run_training_entry(
        config_path=args.config,
        override_paths=args.override,
        output_dir=args.output_dir,
        seed=args.seed,
        resume_from=args.resume
    )




if __name__ == "__main__":
    main()
