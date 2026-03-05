from src.config import load_config, PostTrainConfig, load_config_from_dict

import torch
import os
import argparse

from typing import Tuple, Sequence, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset

from src.train_artifacts import guard_output_dir_empty, write_yaml, write_json, build_train_meta, append_jsonl, require_accelerate_if_needed
from src.utils.config_utils import deep_merge



def build_model(cfg: PostTrainConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
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
            raise RuntimeError(
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

    model.print_trainable_parameters()
    return model, tokenizer


def load_data(cfg: PostTrainConfig, tokenizer: AutoTokenizer) -> Dataset:
    if not os.path.exists(cfg.train_file):
        raise FileNotFoundError(f"Training file not found: {cfg.train_file}")

    ds = load_dataset("json", data_files={"train": cfg.train_file})["train"]
    if "messages" not in ds.column_names:
        raise ValueError("Training JSONL must contain a 'messages' field per line for v1.")

    def messages_to_text(messages):
        # Normalize
        norm = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if role and content:
                norm.append({"role": role, "content": content})
        if not norm:
            raise ValueError("Empty/invalid messages in one training example")

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
    train_dataset: Dataset
) -> None:
    
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
        save_strategy="no",
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

    trainer.train()

    # Save adapter weights (not full base model) + tokenizer.
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)





def run_training_entry(*, config_path: str, override_paths: Sequence[str], output_dir: Optional[str], seed: Optional[int]) -> None:
    

    cfg, cfg_snapshot_raw = load_config(config_path, override_paths=override_paths)

    cli_ov: Dict[str, Any] = {}
    if output_dir is not None:
        cli_ov.setdefault("output", {})["output_dir"] = output_dir
    if seed is not None:
        cli_ov["seed"] = int(seed)
    if cli_ov:
        cfg_snapshot_raw = dict(cfg_snapshot_raw)
        deep_merge(cfg_snapshot_raw, cli_ov)
        cfg = load_config_from_dict(cfg_snapshot_raw)

    set_seed(cfg.seed)

    guard_output_dir_empty(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    write_yaml(os.path.join(cfg.output_dir, "config_snapshot.yaml"), cfg_snapshot_raw)

    resolved_view = dict(cfg_snapshot_raw)
    resolved_view["override_paths"] = list(override_paths)
    resolved_view["resolved"] = cfg.to_dict()
    write_yaml(os.path.join(cfg.output_dir, "config_resolved.yaml"), resolved_view)

    model, tokenizer = build_model(cfg)
    train_dataset = load_data(cfg, tokenizer)

    meta = build_train_meta(
        output_dir=cfg.output_dir,
        cfg_dict=cfg_snapshot_raw,
        dataset_size_used=len(train_dataset),
        config_path=config_path,
        override_paths=override_paths
    )
    write_json(os.path.join(cfg.output_dir, "meta.json"), meta)

    run_training(cfg, model, tokenizer, train_dataset)



def main():
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
    args = ap.parse_args()

    run_training_entry(
        config_path=args.config,
        override_paths=args.override,
        output_dir=args.output_dir,
        seed=args.seed
    )




if __name__ == "__main__":
    main()
