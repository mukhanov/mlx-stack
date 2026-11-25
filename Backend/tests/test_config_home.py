import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_llm import clients
from local_llm.config import Config


class ConfigHomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlx-stack-config-test-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.stack = self.home / ".mlx-stack"
        self.env = patch.dict(os.environ, {
            "HOME": str(self.home),
            "MLX_STACK_HOME": str(self.stack),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
        }, clear=False).start()
        self.addCleanup(patch.stopall)

    def test_launcher_migrates_legacy_home_and_keeps_compat_link(self):
        old = self.home / ".ollmlx"
        old.mkdir()
        (old / "model-settings.json").write_text('{"model": {"max_ctx": 12345}}')
        script = Path(__file__).parents[1] / "mlx-stack-home.sh"
        result = subprocess.run(
            ["sh", "-c", '. "$1"; mlx_stack_home; printf "%s" "$OLLMLX"',
             "test", str(script)],
            env={**os.environ, "MLX_STACK_HOME": ""},
            text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout, str(self.stack))
        self.assertTrue(old.is_symlink())
        self.assertEqual((self.stack / "model-settings.json").read_text(),
                         '{"model": {"max_ctx": 12345}}')

    def test_harness_configs_are_canonical_and_external_paths_are_links(self):
        opencode = self.home / ".config" / "opencode" / "opencode.json"
        opencode.parent.mkdir(parents=True)
        opencode.write_text(json.dumps({"theme": "dark",
                                        "provider": {"other": {"enabled": True}}}))
        cfg = Config.from_env()
        clients.write_client_configs(cfg, "org/model", small=False, max_input=8192)

        expected = {
            "pi-models.json": self.home / ".pi" / "agent" / "models.json",
            "omp-models.json": self.home / ".omp" / "agent" / "models.json",
            "opencode.json": opencode,
        }
        for name, external in expected.items():
            canonical = self.stack / "configs" / name
            self.assertTrue(canonical.is_file(), canonical)
            self.assertTrue(external.is_symlink(), external)
            self.assertEqual(external.resolve(), canonical.resolve())
            data = json.loads(canonical.read_text())
            self.assertIn("ollmlx", data["providers" if name != "opencode.json" else "provider"])

        provider = json.loads((self.stack / "configs" / "opencode.json").read_text())
        self.assertEqual(provider["theme"], "dark")
        self.assertTrue(provider["provider"]["other"]["enabled"])
        self.assertEqual(provider["provider"]["ollmlx"]["options"]["baseURL"],
                         "http://127.0.0.1:4000/v1")
        self.assertEqual(cfg.claude_cfg, self.stack / "configs" / "claude.json")
        self.assertEqual(json.loads(cfg.claude_cfg.read_text())["env"]["ANTHROPIC_MODEL"],
                         "local")

    def test_late_legacy_writers_are_merged_without_overwriting(self):
        old = self.home / ".ollmlx"
        self.stack.mkdir()
        old_logs = old / "logs"
        old_logs.mkdir(parents=True)
        (old_logs / "old.log").write_text("old")
        new_logs = self.stack / "logs"
        new_logs.mkdir()
        (new_logs / "new.log").write_text("new")
        script = Path(__file__).parents[1] / "mlx-stack-home.sh"
        subprocess.run(
            ["sh", "-c", '. "$1"; mlx_stack_home', "test", str(script)],
            env={**os.environ, "MLX_STACK_HOME": ""},
            check=True)
        self.assertTrue(old.is_symlink())
        self.assertEqual((new_logs / "new.log").read_text(), "new")
        self.assertEqual((self.stack / "logs.legacy" / "old.log").read_text(), "old")

    def test_snapshot_client_settings_include_connection_parameters(self):
        cfg = Config.from_env()
        settings = clients.client_configs(cfg)
        self.assertEqual({item["name"] for item in settings},
                         {"pi", "omp", "opencode", "claude"})
        for item in settings:
            self.assertTrue(item["config_path"].startswith(str(self.stack)))
            self.assertIn("127.0.0.1", item["api_base"])
            self.assertTrue(item["command"])
            self.assertTrue(item["model"])


if __name__ == "__main__":
    unittest.main()
