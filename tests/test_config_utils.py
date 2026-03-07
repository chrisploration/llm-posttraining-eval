import unittest
import tempfile
import os

from src.errors import ConfigError
from src.utils.config_utils import deep_merge, load_yaml_mapping




class TestDeepMerge(unittest.TestCase):
    def test_preserves_non_overlapping_keys(self) -> None:
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"b": 2, "nested": {"z": 30}}

        result = deep_merge(base, override)

        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["nested"]["x"], 10)
        self.assertEqual(result["nested"]["y"], 20)
        self.assertEqual(result["nested"]["z"], 30)

    def test_override_replaces_leaf_values(self) -> None:
        base = {"a": 1, "nested": {"x": 10}}
        override = {"a": 99, "nested": {"x": 99}}

        result = deep_merge(base, override)

        self.assertEqual(result["a"], 99)
        self.assertEqual(result["nested"]["x"], 99)

    def test_override_replaces_list_entirely(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = deep_merge(base, override)

        self.assertEqual(result["items"], [4, 5])

    def test_non_dict_base_raises(self) -> None:
        with self.assertRaises(TypeError):
            deep_merge("not a dict", {"a": 1})

    def test_nested_type_conflict_raises(self) -> None:
        base = {"nested": 123}
        override = {"nested": {"x": 1}}

        with self.assertRaises(TypeError):
            deep_merge(base, override)

    def test_mutates_base_in_place(self) -> None:
        base = {"a": 1}
        override = {"b": 2}

        result = deep_merge(base, override)

        self.assertIs(result, base)
        self.assertEqual(base["b"], 2)


class TestLoadYamlMapping(unittest.TestCase):
    def _write_temp_yaml(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(content)
            return f.name

    def test_empty_yaml_returns_empty_dict(self) -> None:
        path = self._write_temp_yaml("")

        try:
            result = load_yaml_mapping(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)

    def test_comment_only_yaml_returns_empty_dict(self) -> None:
        path = self._write_temp_yaml("# comment only\n")

        try:
            result = load_yaml_mapping(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)

    def test_list_yaml_raises(self) -> None:
        path = self._write_temp_yaml("- item1\n- item2\n")

        try:
            with self.assertRaises(ConfigError) as ctx:
                load_yaml_mapping(path)
            self.assertIn("mapping", str(ctx.exception).lower())
        finally:
            os.unlink(path)

    def test_scalar_yaml_raises(self) -> None:
        path = self._write_temp_yaml("just a string\n")

        try:
            with self.assertRaises(ConfigError):
                load_yaml_mapping(path)
        finally:
            os.unlink(path)

    def test_valid_mapping(self) -> None:
        path = self._write_temp_yaml("key1: value1\nkey2: 42\n")

        try:
            result = load_yaml_mapping(path)
            self.assertEqual(result["key1"], "value1")
            self.assertEqual(result["key2"], 42)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()