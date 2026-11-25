import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_llm import hf


class MemoryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.available = 100 * 1024**3
        self.weights = 50 * 1024**3
        self.kv = 1024
        self.model_ctx = 262_144

        def available():
            return SimpleNamespace(available=self.available)

        patches = [
            patch.object(hf.psutil, "virtual_memory", available),
            patch.object(hf, "model_disk_bytes", lambda *_: self.weights),
            patch.object(hf, "model_kv_per_token", lambda *_: self.kv),
            patch.object(hf, "model_ctx", lambda *_: self.model_ctx),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_cache_limit_uses_ten_percent_reserve(self):
        self.assertEqual(hf.memory_cache_limit(Path("hub"), "org/model"), 40 * 1024**3)

    def test_cache_limit_uses_two_gb_minimum_reserve(self):
        self.available = 10 * 1024**3
        self.weights = 1024**3
        self.assertEqual(hf.memory_cache_limit(Path("hub"), "org/model"), 7 * 1024**3)

    def test_reclaimable_ram_increases_cache_budget(self):
        self.assertEqual(hf.memory_cache_limit(Path("hub"), "org/model", 50 * 1024**3),
                         85 * 1024**3)

    def test_prompt_cache_subtracts_from_shared_context_budget(self):
        self.assertEqual(hf.memory_context_limit(Path("hub"), "org/model", 0, 10 * 1024**3),
                         30 * 1024**2)

    def test_safe_context_has_no_double_headroom(self):
        self.model_ctx = 1_000_000_000
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 20 * 1024**3, 0),
                         20 * 1024**2)

    def test_kv8_doubles_context_budget(self):
        self.assertEqual(hf.memory_context_limit(Path("hub"), "org/model", 0, 10 * 1024**3, 8),
                         60 * 1024**2)

    def test_kv4_quadruples_context_budget(self):
        self.assertEqual(hf.memory_context_limit(Path("hub"), "org/model", 0, 10 * 1024**3, 4),
                         120 * 1024**2)

    def test_model_context_wins_when_smaller_than_ram(self):
        self.model_ctx = 12_345
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 0, 0), 12_345)

    def test_hard_cap_wins_when_smaller_than_ram(self):
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 0, 9_999), 9_999)

    def test_cache_cap_wins_when_smaller_than_ram(self):
        self.model_ctx = 1_000_000_000
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 1024**3, 0), 1024**2)

    def test_ram_cap_wins_when_smaller_than_cache_cap(self):
        self.model_ctx = 1_000_000_000
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 30 * 1024**3, 0),
                         10 * 1024**2)

    def test_zero_budget_returns_zero_instead_of_forcing_context(self):
        self.available = 40 * 1024**3
        self.assertEqual(hf.memory_context_limit(Path("hub"), "org/model"), 0)
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 0, 0), 0)

    def test_small_native_context_is_not_forced_up(self):
        self.model_ctx = 2048
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 0, 0), 2048)

    def test_zero_prompt_cache_keeps_ram_limit(self):
        self.model_ctx = 1_000_000_000
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 0, 0), 40 * 1024**2)

    def test_missing_kv_metadata_falls_back_to_model_and_hard_caps(self):
        self.kv = 0
        self.assertIsNone(hf.memory_context_limit(Path("hub"), "org/model"))
        self.assertEqual(hf.safe_context(Path("hub"), "org/model", 10 * 1024**3, 12_345), 12_345)

    def test_cache_limit_never_goes_negative(self):
        self.available = 40 * 1024**3
        self.assertEqual(hf.memory_cache_limit(Path("hub"), "org/model"), 0)

    def test_safe_model_settings_clamps_legacy_values(self):
        cfg = SimpleNamespace(hf_hub=Path("hub"))
        values = {
            "max_ctx": 999_999_999,
            "prompt_cache_bytes": 20 * 1024**3,
            "kv_bits": 4,
            "kv_group_size": 64,
            "load_timeout": 900,
            "client_timeout": 1800,
            "temperature": 2.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "max_tokens": 999_999_999,
        }
        from local_llm.widget import _safe_model_settings
        self.model_ctx = 1_000_000_000
        safe = _safe_model_settings(cfg, "org/model", values)
        self.assertEqual(safe["prompt_cache_bytes"], 20 * 1024**3)
        self.assertEqual(safe["max_tokens"], 80 * 1024**2)
        self.assertEqual(safe["max_ctx"], 80 * 1024**2)
        self.assertEqual(safe["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
