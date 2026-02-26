import os
import unittest
import tempfile
import subprocess
import json
import sys



class TestEvalSmokeArtifacts(unittest.TestCase):
    def test_eval_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "eval_smoke_out")

            cmd = [
                sys.executable,
                "-m",
                "src.eval.run_eval",
                "--config",
                "configs/eval.yaml",
                "--override",
                "configs/overrides/eval_smoke.yaml",
                "--mode",
                "smoke",
                "--output_dir",
                out_dir
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, msg=f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}")

            for name in [
                "meta.json",
                "config_snapshot.yaml",
                "config_resolved.yaml",
                "results.json",
                "metrics.json",
                "samples.jsonl",
                "failures.jsonl"
            ]:
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), msg=f"missing {name}")

            with open(os.path.join(out_dir, "results.json"), "r", encoding="utf-8") as f:
                results = json.load(f)

            tasks = results["tasks"]
            metrics = results["metrics"]
            for t in tasks:
                self.assertIn(t, metrics, msg=f"missing metrics for task {t}")

    def test_eval_unknown_task_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "eval_bad_task_out")

            cmd = [
                sys.executable,
                "-m",
                "src.eval.run_eval",
                "--config",
                "configs/eval.yaml",
                "--override",
                "configs/overrides/eval_bad_task.yaml",
                "--mode",
                "smoke",
                "--output_dir",
                out_dir
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)

            self.assertNotEqual(r.returncode, 0, msg=f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}")
            self.assertIn("Unsupported eval tasks", r.stderr)


if __name__ == "__main__":
    unittest.main()