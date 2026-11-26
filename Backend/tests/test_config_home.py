import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm import clients
from local_llm.config import Config

OMP_CONFIG = """\
# omp's own comments survive
setupVersion: 3
modelRoles:
  plan: kimi-code/k3.1-pro:max
  default: zai/glm-5.4:xhigh
browser:
  relay: true   # inline comment
"""
OMP_CONFIG_WITH_SMOL = OMP_CONFIG.replace(
    "  default: zai/glm-5.4:xhigh\n",
    "  default: zai/glm-5.4:xhigh\n  smol: ollmlx/small\n")


class ConfigHomeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlx-stack-config-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
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
        for harness in ("pi", "omp"):
            models = json.loads((self.stack / "configs" / f"{harness}-models.json")
                                .read_text())["providers"]["ollmlx"]["models"]
            self.assertEqual([m["id"] for m in models], ["org/model", "local"])
            self.assertEqual([m["contextWindow"] for m in models], [8192, 8192])

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
                         {"pi", "omp", "opencode", "claude", "dsh"})
        for item in settings:
            self.assertTrue(item["config_path"].startswith(str(self.stack)))
            self.assertIn("127.0.0.1", item["api_base"])
            self.assertTrue(item["command"])
            self.assertTrue(item["model"])

    def test_the_smol_role_is_one_line_that_comes_and_goes(self):
        external = self.home / ".omp" / "agent" / "config.yml"
        external.parent.mkdir(parents=True)
        external.write_text(OMP_CONFIG)
        cfg = Config.from_env()

        clients.write_client_configs(cfg, "org/model", small=True, max_input=8192)
        canonical = self.stack / "configs" / "omp-config.yml"
        self.assertTrue(external.is_symlink())
        self.assertEqual(external.read_text(), OMP_CONFIG_WITH_SMOL)

        clients.write_client_configs(cfg, "org/model", small=False, max_input=8192)
        self.assertEqual(external.read_text(), OMP_CONFIG)

    def test_a_smol_role_that_is_not_ours_and_an_unknown_layout_stay(self):
        external = self.home / ".omp" / "agent" / "config.yml"
        external.parent.mkdir(parents=True)
        cfg = Config.from_env()
        for text in ("modelRoles:\n  smol: other/model\n",
                     "modelRoles: {plan: other/model}\n",
                     "setupVersion: 3\n"):
            with self.subTest(config=text):
                external.write_text(text)
                clients.write_client_configs(cfg, "org/model", small=False,
                                             max_input=8192)
                self.assertEqual(external.read_text(), text)

    def test_a_harness_without_a_config_gets_the_provider(self):
        # opencode starts from an empty file, which is no config yet —
        # not "JSONC to leave alone"
        cfg = Config.from_env()
        clients.write_client_configs(cfg, "org/model", small=True, max_input=8192)
        opencode = json.loads((self.home / ".config" / "opencode" / "opencode.json")
                              .read_text())
        self.assertEqual(opencode["provider"]["ollmlx"]["models"]["local"]["name"],
                         "org/model")
        self.assertEqual(opencode["small_model"], "ollmlx/small")

    def test_real_jsonc_is_left_to_its_owner(self):
        external = self.home / ".config" / "opencode" / "opencode.json"
        external.parent.mkdir(parents=True)
        text = '{ // comment\n  "theme": "dark" }\n'
        external.write_text(text)
        cfg = Config.from_env()
        clients.write_client_configs(cfg, "org/model", small=False, max_input=8192)
        self.assertTrue(external.is_symlink())
        self.assertEqual(external.read_text(), text)

    def test_the_dsh_overlay_declares_the_proxy_as_a_custom_model_api(self):
        cfg = Config.from_env()
        with patch.object(clients.hf, "model_ctx", return_value=40960):
            clients.write_client_configs(cfg, "org/model", small=True, max_input=8192)
        overlay = (self.stack / "configs" / "dsh-ollmlx.yml").read_text()
        self.assertIn("id: llm-pi-ai", overlay)
        self.assertIn("api: openai-completions", overlay)
        self.assertIn("baseURL: http://127.0.0.1:4000/v1", overlay)
        self.assertIn("Authorization: Bearer sk-local-llm", overlay)
        for model in ("org/model", "local", "small"):
            self.assertIn(f"- id: {model}\n", overlay)
        self.assertIn("contextWindow: 8192", overlay)
        self.assertIn("contextWindow: 40960", overlay)   # the helper's own window
        # dsh's own configuration is left alone: only the overlay is written
        self.assertFalse((self.home / ".dsh").exists())
        command = next(item["command"] for item in clients.client_configs(cfg)
                       if item["name"] == "dsh")
        self.assertEqual(command, "dsh web --patch ~/.mlx-stack/configs/dsh-ollmlx.yml")

    def test_model_settings_override_the_stack_wide_ones_and_round_trip(self):
        cfg = Config.from_env()
        self.assertEqual(cfg.model_settings("org/model")["max_ctx"], cfg.max_ctx)
        values = {**cfg.model_settings("org/model"), "max_ctx": 4096, "top_k": 40}
        cfg.save_model_settings("org/model", values)
        saved = cfg.model_settings("org/model")
        self.assertEqual((saved["max_ctx"], saved["top_k"]), (4096, 40))
        self.assertEqual(cfg.model_settings("org/other")["max_ctx"], cfg.max_ctx)
        self.assertEqual(cfg.model_environment("org/model")["MLX_MAX_CTX"], "4096")
        self.assertEqual(cfg.model_environment("org/other")["MLX_MAX_CTX"],
                         str(cfg.max_ctx))


if __name__ == "__main__":
    unittest.main()
