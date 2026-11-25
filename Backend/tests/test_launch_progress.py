"""What the window is told while a model loads, and why a load stopped."""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psutil

from local_llm import cli, servers, widget

GB = 1024**3
MODEL = "org/model"
CONTROLLER = f"{sys.executable} -m local_llm serve {MODEL}"


class Controller:
    """Stands in for the process that loads a model while its record is checked."""

    def __init__(self, _pid, cmdline=CONTROLLER):
        self._cmdline = cmdline

    def cmdline(self):
        return self._cmdline.split()


def write_record(path: Path, **fields) -> None:
    path.write_text(json.dumps({"model": MODEL, "pid": os.getpid(),
                                "started": time.time(), **fields}))


class LaunchProgressTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="mlx-stack-launch-test-"))
        self.addCleanup(lambda: os.system(f"rm -rf {self.home}"))
        self.cfg = SimpleNamespace(home=self.home)
        self.path = self.home / "launch.json"

    def test_each_stage_is_published_and_ready_removes_the_record(self):
        progress = cli.LaunchProgress(self.cfg, MODEL)
        for stage in ("prepare", "weights", "helper", "warmup"):
            progress.stage(stage)
            record = json.loads(self.path.read_text())
            self.assertEqual(record["stage"], stage)
            self.assertEqual((record["model"], record["pid"]), (MODEL, os.getpid()))
            self.assertLess(abs(record["started"] - time.time()), 5)
        progress.done()
        self.assertFalse(self.path.exists())

    def test_a_failure_stays_with_its_reason(self):
        cli.LaunchProgress(self.cfg, MODEL).failed("mlx_lm.server died (code 1)", 1)
        record = json.loads(self.path.read_text())
        self.assertEqual(record["stage"], "failed")
        self.assertEqual((record["status"], record["error"]),
                         (1, "mlx_lm.server died (code 1)"))

    def test_an_unwritable_home_does_not_break_a_launch(self):
        cfg = SimpleNamespace(home=self.home / "file")
        cfg.home.write_text("not a directory")
        progress = cli.LaunchProgress(cfg, MODEL)
        progress.stage("weights")
        progress.done()
        progress.failed("x")

    def test_the_reason_of_a_failure_is_one_line(self):
        died = servers.StartupFailed("mlx_lm.server died (code -6)")
        self.assertIsInstance(died, SystemExit)
        self.assertEqual(cli._failure(died), "mlx_lm.server died (code -6)")
        self.assertEqual(cli._failure(SystemExit("   port collision: …")),
                         "port collision: …")
        self.assertEqual(cli._failure(SystemExit(3)), "exit status 3")
        self.assertEqual(cli._failure(OSError("no space left on device")),
                         "OSError: no space left on device")


class LaunchRecordTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="mlx-stack-launch-test-"))
        self.addCleanup(lambda: os.system(f"rm -rf {self.home}"))
        self.cfg = SimpleNamespace(home=self.home, hf_hub=self.home / "hub")
        self.path = self.home / "launch.json"

    def test_without_a_record_nothing_is_loading(self):
        self.assertIsNone(widget._launch_in_progress(self.cfg, 0))
        self.assertIsNone(widget._launch_failure(self.cfg, main_up=False))
        self.path.write_text("{not json")
        self.assertIsNone(widget._launch_in_progress(self.cfg, 0))
        self.assertIsNone(widget._launch_failure(self.cfg, main_up=False))

    def test_the_weights_count_as_loaded_in_step_with_the_servers_memory(self):
        weights = patch.object(widget.hf, "model_disk_bytes", return_value=10 * GB)
        with weights, patch.object(widget.psutil, "Process", Controller):
            write_record(self.path, stage="weights")
            load = lambda rss: widget._launch_in_progress(self.cfg, rss)
            self.assertEqual(load(0)["fraction"], 0.0)
            self.assertAlmostEqual(load(6 * GB)["fraction"], 0.55)
            self.assertEqual(load(40 * GB)["fraction"], 1.0)     # never past the end
            self.assertEqual((load(6 * GB)["stage"], load(6 * GB)["model"]),
                             ("weights", MODEL))
            self.assertLess(load(6 * GB)["elapsed"], 5)
            for stage in ("prepare", "helper", "warmup"):
                write_record(self.path, stage=stage)
                self.assertIsNone(load(6 * GB)["fraction"])     # nothing to measure

    def test_a_record_nobody_is_driving_is_ignored(self):
        write_record(self.path, stage="weights")
        dead = psutil.NoSuchProcess(os.getpid())
        with patch.object(widget.psutil, "Process", side_effect=dead):
            self.assertIsNone(widget._launch_in_progress(self.cfg, 0))   # controller killed
        with patch.object(widget.psutil, "Process",
                          lambda _pid: Controller(_pid, "/bin/sleep 9")):
            self.assertIsNone(widget._launch_in_progress(self.cfg, 0))   # pid reused
        self.path.write_text("[]")
        self.assertIsNone(widget._launch_in_progress(self.cfg, 0))
        write_record(self.path, stage="weights", started=time.time() - 2 * 3600)
        with patch.object(widget.psutil, "Process", Controller):
            self.assertIsNone(widget._launch_in_progress(self.cfg, 0))   # long stale

    def test_a_failure_is_reported_until_a_stack_serves(self):
        write_record(self.path, stage="failed", status=1, error="port busy")
        self.assertEqual(widget._launch_failure(self.cfg, main_up=False),
                         {"model": MODEL, "status": 1, "reason": "port busy"})
        self.assertTrue(self.path.exists())
        self.assertIsNone(widget._launch_failure(self.cfg, main_up=True))
        self.assertFalse(self.path.exists())


class StackControllerTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="mlx-stack-launch-test-"))
        self.addCleanup(lambda: os.system(f"rm -rf {self.home}"))
        cfg = SimpleNamespace(home=self.home, hf_hub=self.home / "hub",
                              small_model="org/small")
        self.path = self.home / "launch.json"
        self.start_api = lambda: widget.StackController(cfg)

    def test_a_new_app_session_does_not_resurface_an_old_failure(self):
        write_record(self.path, stage="failed")
        self.start_api()
        self.assertFalse(self.path.exists())

    def test_a_load_someone_else_started_is_still_reported(self):
        write_record(self.path, stage="weights")
        self.start_api()
        self.assertTrue(self.path.exists())


class ServeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="mlx-stack-launch-test-"))
        self.addCleanup(lambda: os.system(f"rm -rf {self.home}"))
        self.cfg = SimpleNamespace(home=self.home)
        self.path = self.home / "launch.json"

    def serve(self, raised):
        """How run_serve ends when the boot raises `raised`: the exit status
        it returns, or the exception it passes on to the interpreter."""
        with patch.object(cli, "_boot", side_effect=raised), \
                patch.object(cli.servers, "stop_watchdog"), \
                patch.object(cli.servers, "stop_children"):
            try:
                return cli.run_serve(self.cfg, MODEL)
            except BaseException as exc:
                return exc

    def test_a_failed_start_leaves_its_reason_for_the_window(self):
        message = SystemExit("   port collision: mlx=1, small=2")
        cases = [   # what the boot raises, what run_serve ends with, the recorded reason
            (servers.StartupFailed("mlx_lm.server died (code -6)"), 1,
             "mlx_lm.server died (code -6)"),
            (message, message, "port collision: mlx=1, small=2"),
            (SystemExit(3), 3, "exit status 3"),
            (OSError("no space left"), OSError("no space left"), "OSError: no space left"),
        ]
        for raised, outcome, reason in cases:
            with self.subTest(reason=reason):
                ended = self.serve(raised)
                if isinstance(outcome, int):
                    self.assertEqual(ended, outcome)
                else:
                    self.assertIs(ended, raised)
                record = json.loads(self.path.read_text())
                self.assertEqual((record["stage"], record["status"], record["error"]),
                                 ("failed", outcome if isinstance(outcome, int) else 1,
                                  reason))

    def test_a_declined_start_leaves_no_record(self):
        write_record(self.path, stage="prepare")
        with patch.object(cli, "_boot", return_value=False):
            self.assertEqual(cli.run_serve(self.cfg, MODEL), 0)
        self.assertFalse(self.path.exists())


class BootStagesTests(unittest.TestCase):
    """The whole boot with every server and download swapped for stubs: what
    matters is the order of the stages told to the window."""

    def test_the_window_hears_every_stage_in_order_until_the_stack_serves(self):
        home = Path(tempfile.mkdtemp(prefix="mlx-stack-launch-test-"))
        self.addCleanup(lambda: os.system(f"rm -rf {home}"))
        cfg = SimpleNamespace(home=home, hf_hub=home / "hub", hf_token=None,
                              mlx_port=19797, mlx_small_port=8081, proxy_port=4000,
                              prompt_cache_bytes=0, max_ctx=8192, kv_bits=0,
                              small_model="org/small", claude_cfg="claude.json")

        class Recorder(cli.LaunchProgress):
            heard = []

            def stage(self, name):
                Recorder.heard.append(name)
                super().stage(name)

        nothing = lambda *_, **__: None
        stubs = [
            patch.object(cli.servers, "stop_stale", nothing),
            patch.object(cli.servers, "start_proxy", nothing),
            patch.object(cli.servers, "start_mlx", lambda *_a, **_k: "main"),
            patch.object(cli.servers, "start_mlx_small", lambda *_a, **_k: "small"),
            patch.object(cli.servers, "warmup", nothing),
            patch.object(cli.servers, "supervise", nothing),
            patch.object(cli.clients, "write_client_configs", nothing),
            patch.object(cli.hf, "local_gguf_files", lambda *_: []),
            patch.object(cli.hf, "repo_gguf_files", lambda *_: None),
            patch.object(cli.hf, "missing_files", lambda *_: []),
            patch.object(cli.hf, "is_cached", lambda *_: True),
            patch.object(cli.hf, "model_disk_bytes", lambda *_: 0),
            patch.object(cli.hf, "packed_format", lambda *_: None),
            patch.object(cli.hf, "safe_context", lambda *_a, **_k: 8192),
            patch.object(cli.hf, "model_ctx", lambda *_: 8192),
            patch.object(cli.psutil, "virtual_memory",
                         lambda: SimpleNamespace(available=100 * GB)),
        ]
        Recorder.heard = []
        with contextlib.ExitStack() as stack:
            for stub in stubs:
                stack.enter_context(stub)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            booted = cli._boot(cfg, MODEL, None, Recorder(cfg, MODEL))
        self.assertTrue(booted)
        self.assertEqual(Recorder.heard, ["prepare", "weights", "helper", "warmup"])
        self.assertFalse((home / "launch.json").exists())   # the stack serves


class StartFailureTests(unittest.TestCase):
    def test_a_server_that_dies_names_itself_and_its_exit_code(self):
        class Exited:
            returncode = 3

            def poll(self):
                return 3

        with patch.object(servers, "_spawn", return_value=Exited()), \
                patch.object(servers, "http_ok", return_value=False):
            with self.assertRaises(servers.StartupFailed) as caught:
                servers._start([sys.executable, "-c", "raise SystemExit(3)"], None,
                               "/nonexistent/server.log",
                               "http://127.0.0.1:1/v1/models", 60, "test server")
        self.assertEqual(caught.exception.reason, "test server died (code 3)")
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
