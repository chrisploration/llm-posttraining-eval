import os
import sys
import subprocess
import tempfile
import unittest


class TestTrainSmokeArtifacts(unittest.TestCase):
    def test_train_smoke_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = tempfile.mkdtemp(prefix="train_smoke_out_", dir=td)

            cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--config",
                "configs/posttrain.yaml",
                "--override",
                "configs/overrides/posttrain_smoke.yaml",
                "--output_dir",
                out_dir
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, msg=f"stderr:\n{r.stderr}\nstdout:\n{r.stdout}")


            for name in [
                "meta.json",
                "config_snapshot.yaml",
                "config_resolved.yaml",
                "train_log.jsonl"
            ]:
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), msg=f"missing {name}")


if __name__ == "__main__":
    unittest.main()