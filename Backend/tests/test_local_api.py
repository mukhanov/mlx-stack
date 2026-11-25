"""The app's local API: bound to loopback, and closed to web pages."""
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.client import HTTPConnection

os.environ["MLX_STACK_HOME"] = tempfile.mkdtemp(
    prefix="mlx-stack-api-test-")   # before the backend resolves its paths

from local_llm import widget  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.net import HOST  # noqa: E402

REFUSED = "the local API only answers programs on this machine"
SNAPSHOT_KEYS = {
    "model", "small_model", "launch_error", "backend", "uptime_seconds",
    "max_input", "config_home", "available_models", "can_switch", "ports",
    "cpu", "memory", "swap", "processes", "process_summary", "top_eaters",
    "main", "small", "clients", "logs",
}
SETTINGS_KEYS = {
    "settings", "defaults", "kv_bytes_per_token", "effective_ctx", "model_ctx",
    "available_bytes", "reclaimable_bytes", "is_loaded", "budget_available_bytes",
    "weights_bytes", "cache_limit_bytes",
}


class LocalAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(shutil.rmtree, os.environ["MLX_STACK_HOME"], True)
        cls.cfg = Config.from_env()
        cls.cfg.widget_port = 0                 # any free port
        # ports nothing of this machine's own stack listens on
        cls.cfg.mlx_port, cls.cfg.mlx_small_port = 29797, 28081
        cls.cfg.proxy_port = 24000
        cls.controller = widget.StackController(cls.cfg)
        downloads = widget.DownloadManager(cls.cfg, cls.controller.refresh_models)
        cls.controller.downloads = downloads
        cls.server = widget._make_server(cls.cfg, cls.controller, downloads)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)

    def call(self, method, path, payload=None, headers=None):
        head = {"Content-Type": "application/json", **(headers or {})}
        body = json.dumps(payload).encode() if payload is not None else None
        conn = HTTPConnection(HOST, self.port, timeout=10)
        conn.request(method, path, body=body, headers=head)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        try:
            return resp.status, json.loads(data)
        except ValueError:
            return resp.status, data

    def test_the_api_binds_loopback_only(self):
        self.assertEqual(HOST, "127.0.0.1")
        self.assertEqual(self.server.server_address[0], HOST)

    def test_the_app_gets_a_complete_snapshot(self):
        status, snapshot = self.call("GET", "/snapshot")
        self.assertEqual(status, 200)
        self.assertTrue(SNAPSHOT_KEYS <= set(snapshot), SNAPSHOT_KEYS - set(snapshot))
        self.assertEqual(snapshot["model"], "no model loaded")
        self.assertEqual(snapshot["ports"]["main"]["up"], False)
        self.assertEqual(snapshot["config_home"], str(self.cfg.home))
        self.assertGreater(snapshot["memory"]["total"], 0)
        self.assertGreater(len(snapshot["cpu"]["cores"]), 0)
        self.assertEqual({c["name"] for c in snapshot["clients"]},
                         {"pi", "omp", "opencode", "claude"})

    def test_localhost_is_as_good_a_name_for_this_machine(self):
        status, _ = self.call("GET", "/snapshot",
                              headers={"Host": f"localhost:{self.port}"})
        self.assertEqual(status, 200)

    def test_a_page_reaching_the_port_by_dns_rebinding_is_refused(self):
        # the page's own site in Host: it could read the answers otherwise
        status, body = self.call("GET", "/snapshot",
                                 headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": REFUSED})
        status, body = self.call("POST", "/stop", {"model": "org/model"},
                                 headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)

    def test_a_cross_site_request_from_a_browser_is_refused(self):
        # browsers always name the page in Origin; URLSession never does
        page = {"Origin": "https://evil.example"}
        status, body = self.call("GET", "/snapshot", headers=page)
        self.assertEqual(status, 403)
        self.assertEqual(body, {"error": REFUSED})
        for path in ("/run", "/delete", "/download", "/stop"):
            with self.subTest(path=path):
                status, _ = self.call("POST", path, {"model": "org/model"},
                                      headers=page)
                self.assertEqual(status, 403)
        # and nothing was acted on
        self.assertIsNone(self.controller.proc)
        self.assertEqual(self.controller.downloads.all_progress(), {})

    def test_settings_are_per_model(self):
        status, body = self.call("GET", "/settings")
        self.assertEqual((status, body), (400, {"error": "expected a model name"}))
        status, body = self.call("GET", "/settings?model=org/model")
        self.assertEqual(status, 200)
        self.assertTrue(SETTINGS_KEYS <= set(body), SETTINGS_KEYS - set(body))
        self.assertEqual(body["is_loaded"], False)
        self.assertEqual(set(body["settings"]), set(body["defaults"]))
        self.assertEqual(body["reclaimable_bytes"], 0)

    def test_launch_settings_are_checked_before_they_are_saved(self):
        full = {"max_ctx": 110_000, "prompt_cache_bytes": 16 * 1024**3, "kv_bits": 0,
                "kv_group_size": 64, "load_timeout": 900, "client_timeout": 1800,
                "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                "max_tokens": 32768}
        model = {"model": "org/model", **full}
        cases = [
            ({"model": "org/model"}, "expected all model launch settings"),
            ({"temperature": 0.5}, "expected a model name"),
            ({**model, "temperature": 2}, "temperature must be between 0 and 1"),
            ({**model, "kv_bits": 7}, "KV bits must be 0, 4 or 8"),
            ({**model, "max_ctx": 1024}, "context must be between 4K and 262K tokens"),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload):
                status, body = self.call("POST", "/settings", payload)
                self.assertEqual((status, body), (400, {"error": message}))
        self.assertFalse((self.cfg.home / "model-settings.json").exists())

    def test_malformed_requests_are_rejected(self):
        conn = HTTPConnection(HOST, self.port, timeout=10)
        conn.request("POST", "/run", body=b"[1, 2, 3]",
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status, body = resp.status, json.loads(resp.read())
        conn.close()
        self.assertEqual((status, body), (400, {"error": "invalid JSON"}))
        self.assertEqual(self.call("GET", "/nowhere")[0], 404)
        self.assertEqual(self.call("POST", "/nowhere", {"model": "m"})[0], 404)


if __name__ == "__main__":
    unittest.main()
