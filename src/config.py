from typing import List, Dict, Any

import yaml

class PostTrainConfig:
    def __init__(
        self,
        *,
        seed: int,
        base_id: str,
        load_in_4bit: bool,
        train_file: str,
        max_seq_length: int,
        output_dir: str,
        # training (v1: SFT only)
        num_epochs: int,
        learning_rate: float,
        warmup_ratio: float,
        max_grad_norm: float,
        # batching
        micro_batch_size: int,
        grad_accum_steps: int,
        gradient_checkpointing: bool,
        # LoRA
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str]
    ) -> None:
        # Assume load_config() already casts types; validate only the expensive-failure cases.
        if not base_id:
            raise ValueError("base_id must be non-empty")
        if not train_file:
            raise ValueError("train_file must be non-empty")
        if not output_dir:
            raise ValueError("output_dir must be non-empty")

        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be > 0")
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be > 0")
        if grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not lora_target_modules:
            raise ValueError("lora_target_modules must be a non-empty list")

        self.seed = seed
        self.base_id = base_id
        self.load_in_4bit = load_in_4bit

        self.train_file = train_file
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm

        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.gradient_checkpointing = gradient_checkpointing

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = list(lora_target_modules)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "base_id": self.base_id,
            "load_in_4bit": self.load_in_4bit,
            "train_file": self.train_file,
            "max_seq_length": self.max_seq_length,
            "output_dir": self.output_dir,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "micro_batch_size": self.micro_batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.lora_target_modules
            }
        }

def _get(d: dict, path: str):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"Missing required config key: {path}")
        cur = cur[key]
    return cur

def load_config(path: str) -> PostTrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model = _get(raw, "model")
    data = _get(raw, "data")
    training = _get(raw, "training")
    batching = _get(raw, "batching")
    lora = _get(raw, "lora")
    output = _get(raw, "output")

    method = str(training["method"]).strip().lower()
    if method != "sft":
        raise ValueError("v1 only supports training.method: sft")

    data_format = str(data["format"]).strip().lower()
    if data_format != "chat":
        raise ValueError("v1 expects data.format: chat and JSONL with a 'messages' field.")

    lora_targets = list(lora["target_modules"])
    if not lora_targets:
        raise ValueError("lora.target_modules must be a non-empty list")

    return PostTrainConfig(
        seed=int(raw.get("seed", 0)),
        base_id=str(model["base_id"]),
        load_in_4bit=bool(model["load_in_4bit"]),
        train_file=str(data["train_file"]),
        max_seq_length=int(data["max_seq_length"]),
        output_dir=str(output["output_dir"]),
        num_epochs=int(training["num_epochs"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        max_grad_norm=float(training["max_grad_norm"]),
        micro_batch_size=int(batching["micro_batch_size"]),
        grad_accum_steps=int(batching["grad_accum_steps"]),
        gradient_checkpointing=bool(batching["gradient_checkpointing"]),
        lora_r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        lora_target_modules=lora_targets
    )