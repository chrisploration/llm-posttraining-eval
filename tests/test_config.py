import unittest

from src.errors import ConfigError
from src.config import PostTrainConfig, load_config_from_dict



def _valid_raw_config() -> dict:
    return {
        "seed": 42,
        "model": {
            "base_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "load_in_4bit": True
        },
        "data": {
            "train_file": "data/train.jsonl",
            "format": "chat",
            "max_seq_length": 2048
        },
        "training": {
            "method": "sft",
            "num_epochs": 1,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0
        },
        "batching": {
            "micro_batch_size": 1,
            "grad_accum_steps": 16,
            "gradient_checkpointing": True
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj"]
        },
        "output": {
            "output_dir": "checkpoints/test"
        }
    }


def _valid_posttrain_kwargs() -> dict:
    return {
        "seed": 42,
        "base_id": "test-model",
        "load_in_4bit": False,
        "train_file": "data/train.jsonl",
        "max_seq_length": 2048,
        "output_dir": "checkpoints/test",
        "num_epochs": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "micro_batch_size": 1,
        "grad_accum_steps": 16,
        "gradient_checkpointing": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"]
    }


class TestPostTrainConfig(unittest.TestCase):
    def test_valid_construction(self) -> None:
        cfg = PostTrainConfig(**_valid_posttrain_kwargs())

        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.base_id, "test-model")
        self.assertEqual(cfg.lora_target_modules, ["q_proj", "v_proj"])


    def test_empty_base_id_raises(self) -> None:
        kwargs = _valid_posttrain_kwargs()
        kwargs["base_id"] = ""

        with self.assertRaises(ConfigError):
            PostTrainConfig(**kwargs)


    def test_max_seq_length_zero_raises(self) -> None:
        kwargs = _valid_posttrain_kwargs()
        kwargs["max_seq_length"] = 0

        with self.assertRaises(ConfigError):
            PostTrainConfig(**kwargs)


    def test_negative_max_seq_length_raises(self) -> None:
        kwargs = _valid_posttrain_kwargs()
        kwargs["max_seq_length"] = -1

        with self.assertRaises(ConfigError):
            PostTrainConfig(**kwargs)


    def test_empty_lora_target_modules_raises(self) -> None:
        kwargs = _valid_posttrain_kwargs()
        kwargs["lora_target_modules"] = []

        with self.assertRaises(ConfigError):
            PostTrainConfig(**kwargs)


    def test_empty_output_dir_raises(self) -> None:
        kwargs = _valid_posttrain_kwargs()
        kwargs["output_dir"] = ""

        with self.assertRaises(ConfigError):
            PostTrainConfig(**kwargs)


class TestLoadConfigFromDict(unittest.TestCase):
    def test_valid_config_loads(self) -> None:
        raw = _valid_raw_config()

        cfg = load_config_from_dict(raw)

        self.assertEqual(cfg.base_id, "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.learning_rate, 2e-4)


    def test_non_sft_method_raises(self) -> None:
        raw = _valid_raw_config()
        raw["training"]["method"] = "dpo"

        with self.assertRaises(ConfigError):
            load_config_from_dict(raw)


    def test_rlhf_method_raises(self) -> None:
        raw = _valid_raw_config()
        raw["training"]["method"] = "rlhf"

        with self.assertRaises(ConfigError):
            load_config_from_dict(raw)


    def test_non_chat_format_raises(self) -> None:
        raw = _valid_raw_config()
        raw["data"]["format"] = "completion"

        with self.assertRaises(ConfigError):
            load_config_from_dict(raw)


    def test_missing_model_section_raises(self) -> None:
        raw = _valid_raw_config()
        del raw["model"]

        with self.assertRaises(ConfigError):
            load_config_from_dict(raw)


    def test_missing_lora_target_modules_raises(self) -> None:
        raw = _valid_raw_config()
        raw["lora"]["target_modules"] = []

        with self.assertRaises(ConfigError):
            load_config_from_dict(raw)


    def test_non_mapping_config_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config_from_dict(["not", "a", "dict"])


if __name__ == "__main__":
    unittest.main()