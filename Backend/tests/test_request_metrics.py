import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_llm import mlxwrap


class RequestMetricsTests(unittest.TestCase):
    def setUp(self):
        self.emit = patch("local_llm.mlxwrap._emit").start()
        self.tok = patch("local_llm.mlxwrap._tok_bump").start()
        self.active = patch("local_llm.mlxwrap._active_bump").start()
        self.addCleanup(patch.stopall)

    def test_only_watchdog_and_warmup_probes_are_internal(self):
        def request(text, max_tokens):
            return (SimpleNamespace(messages=[{"role": "user", "content": text}]),
                    SimpleNamespace(max_tokens=max_tokens))

        self.assertTrue(mlxwrap._internal_request(*request("ping", 1)))
        self.assertTrue(mlxwrap._internal_request(*request("Say OK", 8)))
        self.assertFalse(mlxwrap._internal_request(*request("ping", 2)))
        self.assertFalse(mlxwrap._internal_request(*request("Explain this", 1)))
        self.assertFalse(mlxwrap._internal_request(*request("A real prompt", 128)))

    def test_internal_completion_emits_no_speed_or_request_activity(self):
        ctx = SimpleNamespace(prompt=(), prompt_cache_count=0)
        out = list(mlxwrap._counting(ctx, iter(range(3)), internal=True))
        self.assertEqual(out, [0, 1, 2])
        self.emit.assert_not_called()
        self.tok.assert_not_called()
        self.active.assert_not_called()

    def test_real_completion_counts_tokens_and_one_request(self):
        ctx = SimpleNamespace(prompt=(), prompt_cache_count=0)
        out = list(mlxwrap._counting(ctx, iter(range(3))))
        self.assertEqual(out, [0, 1, 2])
        self.assertEqual(self.active.call_args_list, [((1,),), ((-1,),)])
        self.assertEqual(self.tok.call_count, 3)
        self.emit.assert_called_once()

    def test_internal_prefill_keeps_server_progress_but_emits_no_metrics(self):
        calls = []
        callback = mlxwrap._prefill_cb(lambda done, total: calls.append((done, total)),
                                       enabled=False)
        callback(10, 20)
        self.assertEqual(calls, [(10, 20)])
        self.emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
