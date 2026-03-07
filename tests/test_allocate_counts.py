import unittest

from src.errors import ConfigError
from src.eval.run_eval import allocate_counts



class TestAllocateCounts(unittest.TestCase):
    def test_counts_sum_to_total_for_mixed_weights(self) -> None:
        tasks = ["basic_capability", "robustness", "safety"]
        mix = {"capability": 0.6, "robustness": 0.25, "safety": 0.15}
        total = 100

        counts = allocate_counts(total=total, tasks=tasks, mix=mix)

        self.assertEqual(sum(counts.values()), total)
        self.assertEqual(set(counts.keys()), set(tasks))
        self.assertGreater(counts["basic_capability"], counts["safety"])


    def test_total_one_assigns_exactly_one_prompt(self) -> None:
        tasks = ["basic_capability", "robustness", "safety"]
        mix = {"capability": 0.6, "robustness": 0.25, "safety": 0.15}

        counts = allocate_counts(total=1, tasks=tasks, mix=mix)

        self.assertEqual(sum(counts.values()), 1)
        non_zero_tasks = [task for task, count in counts.items() if count > 0]
        self.assertEqual(len(non_zero_tasks), 1)


    def test_zero_weight_bucket_gets_zero_allocation(self) -> None:
        tasks = ["basic_capability", "robustness", "safety"]
        mix = {"capability": 0.7, "robustness": 0.3, "safety": 0.0}

        counts = allocate_counts(total=100, tasks=tasks, mix=mix)

        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(counts["safety"], 0)
        self.assertGreater(counts["basic_capability"], 0)
        self.assertGreater(counts["robustness"], 0)


    def test_all_zero_weights_raise_config_error(self) -> None:
        tasks = ["basic_capability", "robustness", "safety"]
        mix = {"capability": 0.0, "robustness": 0.0, "safety": 0.0}

        with self.assertRaises(ConfigError) as ctx:
            allocate_counts(total=100, tasks=tasks, mix=mix)

        self.assertIn("zero weight", str(ctx.exception).lower())


    def test_missing_bucket_key_raises_config_error(self) -> None:
        tasks = ["basic_capability", "robustness", "safety"]
        mix = {"capability": 0.6, "robustness": 0.4}

        with self.assertRaises(ConfigError) as ctx:
            allocate_counts(total=100, tasks=tasks, mix=mix)

        self.assertIn("missing", str(ctx.exception).lower())




if __name__ == "__main__":
    unittest.main()