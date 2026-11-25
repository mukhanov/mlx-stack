"""Which processes the stack may kill, and when."""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from local_llm import servers

MLX = ("/venv/bin/python -m local_llm.mlxwrap --model org/model --host 127.0.0.1"
       " --port 19797 --max-tokens 4096")
RAW_MLX = "/venv/bin/python -m mlx_lm.server --model org/model --port 19797"
LLAMA = ("/Users/x/.mlx-stack/llama.cpp/llama-b1/llama-server -m m.gguf"
         " --host 127.0.0.1 --port 19797 -ngl 999")
OTHERS = [
    "/venv/bin/python -m local_llm.mlxwrap --model org/model --port 197970",
    "/venv/bin/python -m local_llm.mlxwrap --model mlx-community/small --port 8081",
    "/venv/bin/python -m mlx_lm.server --model org/model --port 11434",
    "/opt/homebrew/bin/llama-server -m other.gguf --port 8080",
    "/venv/bin/python -m local_llm serve org/model",
    "/venv/bin/python -m local_llm.widget",
]


class StaleServerTests(unittest.TestCase):
    def matches(self, port, cmdline):
        return re.search(servers.stale_server_pattern(port), cmdline) is not None

    def test_the_pattern_names_the_stacks_servers_on_one_port_and_nothing_else(self):
        for ours in (MLX, RAW_MLX, LLAMA):
            with self.subTest(ours=ours):
                self.assertTrue(self.matches(19797, ours))
                self.assertFalse(self.matches(19798, ours))
        for other in OTHERS:
            with self.subTest(other=other):
                self.assertFalse(self.matches(19797, other))

    def test_stop_sweeps_the_model_ports_and_waits_only_when_it_killed(self):
        cfg = SimpleNamespace(mlx_port=19797, mlx_small_port=8081)
        swept = []

        def pkill(pattern):
            swept.append(pattern)
            return pattern == servers.stale_server_pattern(19797)

        with patch.object(servers, "_pkill", pkill), \
                patch.object(servers.time, "sleep") as sleep:
            self.assertTrue(servers.stop_stale(cfg))
        self.assertEqual(swept, [servers.stale_server_pattern(19797),
                                 servers.stale_server_pattern(8081)])
        sleep.assert_called_once_with(1)   # the port is released as the process dies

        with patch.object(servers, "_pkill", return_value=False), \
                patch.object(servers.time, "sleep") as sleep:
            self.assertFalse(servers.stop_stale(cfg))
        sleep.assert_not_called()

    def test_a_child_that_ignores_sigterm_is_killed(self):
        class Stubborn:
            poll_calls = 0
            killed = False

            def poll(self):
                self.poll_calls += 1
                return None          # alive until SIGKILL

            def terminate(self):
                pass

            def wait(self, timeout=None):
                if not self.killed:
                    raise servers.subprocess.TimeoutExpired("stubborn", timeout)

            def kill(self):
                self.killed = True

        proc = Stubborn()
        servers._stop(proc)
        self.assertTrue(proc.killed)

        class Gone:
            def poll(self):
                return 0

            def terminate(self):
                raise AssertionError("a dead server needs no signal")

        servers._stop(Gone())


if __name__ == "__main__":
    unittest.main()
