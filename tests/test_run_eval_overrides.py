import src.eval.run_eval as run_eval

import unittest
import tempfile
import yaml
import os

class TestEvalOverrides(unittest.TestCase):
    def test_override_updates_num_prompts_keeps_batch_size(self) -> None:
        base_cfg = {
            "model": {"id": "dummy-model"},
            "seed": 0,
            "eval": {
                "num_prompts": 400,
                "batch_size": 8,
                "mix": {"capability": 0.6, "robustness": 0.25, "safety": 0.15},
                "tasks": ["basic_capability", "robustness", "safety"]
            }
        }
        override_cfg = {"eval": {"num_prompts": 50}}

        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "eval.yaml")
            ov_path = os.path.join(td, "eval_smoke.yaml")

            with open(base_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(base_cfg, f, sort_keys=False)
            with open(ov_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(override_cfg, f, sort_keys=False)

            cfg, _ = run_eval.load_config(base_path, override_paths=[ov_path])

            self.assertEqual(int(cfg.eval_cfg["num_prompts"]), 50)
            self.assertEqual(int(cfg.eval_cfg["batch_size"]), 8)

    def test_override_must_be_mapping(self) -> None:
        base_cfg = {
            "model": {"id": "dummy-model"},
            "eval": {"num_prompts": 10, "tasks": ["basic_capability"]}
        }

        with tempfile.TemporaryDirectory() as td:
            base_path = os.path.join(td, "eval.yaml")
            bad_ov_path = os.path.join(td, "bad.yaml")

            with open(base_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(base_cfg, f, sort_keys=False)
            with open(bad_ov_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(["not-a-mapping"], f, sort_keys=False)

            with self.assertRaises(ValueError):
                run_eval.load_config(base_path, override_paths=[bad_ov_path])

    def test_is_peft_adapter_dir_detects_adapter_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(run_eval._is_peft_adapter_dir(td))

            adapter_cfg_path = os.path.join(td, "adapter_config.json")
            with open(adapter_cfg_path, "w", encoding="utf-8") as f:
                f.write("{}\n")

            self.assertTrue(run_eval._is_peft_adapter_dir(td))

            self.assertFalse(run_eval._is_peft_adapter_dir(adapter_cfg_path))

            self.assertFalse(run_eval._is_peft_adapter_dir(None))


if __name__ == "__main__":
    unittest.main()