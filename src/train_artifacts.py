
import subprocess
import os
import yaml
import json
import torch
import sys
import platform
import transformers

from typing import Optional, Mapping, Any, Sequence, Dict
from datetime import datetime, timezone


def require_accelerate_if_needed(device_map: Optional[str]) -> None:
    if device_map != "auto":
        return
    try:
        import accelerate  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "device_map='auto' requires the 'accelerate' package. "
            "Install accelerate or set device_map=None."
        ) from e


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    

def guard_output_dir_empty(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        return
    leftovers = os.listdir(output_dir)
    if leftovers:
        raise FileExistsError(
            f"Refusing to use non-empty output_dir: {output_dir}. "
            "Choose a fresh directory or delete its contents."
        )
    

def write_yaml(path: str, obj: Mapping[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    text = yaml.safe_dump(obj, sort_keys=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_json(path: str, obj: Mapping[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def append_jsonl(path: str, row: Mapping[str, Any]) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def build_train_meta(*, output_dir: str, cfg_dict: Mapping[str, Any], dataset_size_used: int, config_path: Optional[str] = None, override_paths: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "output_dir": output_dir,
        "config_path": config_path,
        "override_paths": list(override_paths) if override_paths else [],
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": {"available": torch.cuda.is_available(), "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None},
        "seed": int(cfg_dict.get("seed", 0)),
        "model_base_id": (cfg_dict.get("model") or {}).get("base_id"),
        "dataset_size_used": int(dataset_size_used),
        "training": cfg_dict.get("training", {}),
        "batching": cfg_dict.get("batching", {}),
        "lora": cfg_dict.get("lora", {}),
        "data": cfg_dict.get("data", {})
    }