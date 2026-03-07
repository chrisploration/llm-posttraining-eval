from typing import Any, Dict, Mapping
import yaml

from src.errors import ConfigError


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(base, dict):
        raise TypeError(f"deep_merge base must be dict, got {type(base).__name__}")

    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            child = base.get(k)
            if not isinstance(child, dict):
                raise TypeError(
                    f"deep_merge expected dict at key '{k}', got {type(child).__name__}"
                )
            deep_merge(child, v)
        else:
            base[k] = v
    return base


def load_yaml_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise ConfigError(f"Override YAML must be a mapping/dict at top-level: {path}")
    return dict(obj)