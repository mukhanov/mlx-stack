"""Launch settings: what the app's form may save, and the clamps that keep
a model inside this machine's RAM."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_llm import hf, widget
from local_llm.config import launch_settings_from

GB = 1024**3
VALID = {"max_ctx": 110_000, "prompt_cache_bytes": 16 * GB, "kv_bits": 0,
         "kv_group_size": 64, "load_timeout": 900, "client_timeout": 1800,
         "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
         "max_tokens": 32768}


class LaunchSettingsTests(unittest.TestCase):
    def test_the_forms_values_round_trip(self):
        self.assertEqual(launch_settings_from(VALID), VALID)
        self.assertEqual(launch_settings_from({**VALID, "top_k": "40"})["top_k"], 40)
        self.assertIsInstance(launch_settings_from(VALID)["top_p"], float)

    def test_each_limit_names_itself(self):
        cases = [
            ("temperature", 1.5, "temperature must be between 0 and 1"),
            ("top_p", -0.1, "top-p and min-p must be between 0 and 1"),
            ("min_p", 2, "top-p and min-p must be between 0 and 1"),
            ("top_k", 4096, "top-k must be between 0 and 1024"),
            ("max_tokens", 128, "max tokens must be between 256 and 262144"),
            ("kv_bits", 7, "KV bits must be 0, 4 or 8"),
            ("kv_group_size", 8, "KV group size must be between 16 and 1024"),
            ("load_timeout", 0, "timeouts must be positive"),
            ("client_timeout", -5, "timeouts must be positive"),
            ("max_ctx", 1024, "context must be between 4K and 262K tokens"),
            ("prompt_cache_bytes", -1, "prompt cache must be between 0 and 512 GB"),
        ]
        for name, value, message in cases:
            with self.subTest(setting=name):
                with self.assertRaises(ValueError) as caught:
                    launch_settings_from({**VALID, name: value})
                self.assertEqual(str(caught.exception), message)

    def test_a_missing_or_non_numeric_value_is_reported(self):
        for payload in ({**VALID, "top_k": "many"}, {k: v for k, v in VALID.items()
                                                     if k != "max_tokens"}, None):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, "expected all model launch settings"):
                    launch_settings_from(payload)


class ClampTests(unittest.TestCase):
    def setUp(self):
        self.kv = 1024                      # bytes per token
        self.available = 100 * GB
        self.weights = 50 * GB
        patches = [
            patch.object(hf.psutil, "virtual_memory",
                         lambda: SimpleNamespace(available=self.available)),
            patch.object(hf, "model_disk_bytes", lambda *_: self.weights),
            patch.object(hf, "model_kv_per_token", lambda *_: self.kv),
            patch.object(hf, "model_ctx", lambda *_: 262_144),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.cfg = SimpleNamespace(hf_hub=Path("hub"))

    def test_saving_refuses_a_model_without_room_for_a_4k_context(self):
        # 52G available - a 10% reserve - 50G weights: no room for a cache
        self.available = 52 * GB
        clamped = widget._safe_model_settings(self.cfg, "org/model", VALID)
        self.assertEqual((clamped["prompt_cache_bytes"], clamped["max_ctx"]), (0, 4096))
        with self.assertRaisesRegex(ValueError, "not enough available RAM"):
            widget._safe_model_settings(self.cfg, "org/model", VALID, strict=True)

    def test_evicting_another_model_frees_its_weights_a_reload_only_lends_its_own(self):
        with patch.object(widget, "_reclaimable_model_bytes", return_value=70 * GB), \
                patch.object(hf, "model_disk_bytes", return_value=20 * GB):
            self.assertEqual(widget._unload_headroom(self.cfg, "org/next", "org/loaded"),
                             (70 * GB, 70 * GB))
            self.assertEqual(widget._unload_headroom(self.cfg, "org/loaded", "org/loaded"),
                             (0, 20 * GB))


if __name__ == "__main__":
    unittest.main()
