from config import load_config, PostTrainConfig

import torch
import os

from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer




def build_model(cfg: PostTrainConfig) -> Tuple[torch.nn.Module, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto"}

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


def load_data(cfg: PostTrainConfig, tokenizer: AutoTokenizer):
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
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
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


def run_training(
    cfg: PostTrainConfig,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_dataset
) -> None:
    
    os.makedirs(cfg.output_dir, exist_ok=True)

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
        packing=False  # v1: predictable behavior
    )

    trainer.train()

    # Save adapter weights (not full base model) + tokenizer.
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)





def main():
    cfg = load_config("configs/posttrain.yaml")
    set_seed(cfg.seed)

    # Post-training pipeline
    model, tokenizer = build_model(cfg)
    train_dataset = load_data(cfg, tokenizer)
    run_training(cfg, model, tokenizer, train_dataset)




if __name__ == "__main__":
    main()
