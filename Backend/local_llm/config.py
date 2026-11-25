"""Paths, ports and settings. Everything is configured via env (see --help)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path


def stack_home() -> Path:
    """Canonical application home; OLLMLX_HOME remains a legacy override."""
    return Path(os.environ.get("MLX_STACK_HOME")
                or os.environ.get("OLLMLX_HOME")
                or "~/.mlx-stack").expanduser()


@dataclass
class Config:
    ollmlx_home: Path
    hf_hub: Path
    mlx_port: int
    proxy_port: int
    widget_port: int
    api_key: str
    # Second, tiny model for cheap fast calls (Claude Code's background
    # title generator and other haiku-slot calls, which use a short fixed
    # timeout and would otherwise queue behind the big model's generation).
    small_model: str
    mlx_small_port: int
    # KV-cache quantization: halves KV memory but, on mlx_lm 0.32, forces the
    # sequential serving path where the prompt cache cannot match long
    # conversations (QuantizedKVCache isn't trimmable) — every request then
    # re-prefills the whole 100k+ prompt, minutes per call. Enable only for
    # models whose weights (~100G of 128G RAM) plus fp16 KV would OOM Metal
    # (the generation thread dies and the server 404s until restarted).
    kv_bits: int
    kv_group_size: int
    # Prompt cache cap (bytes), 0 = no cap. Must exceed the KV of the
    # biggest session you keep alive, or every call re-prefills it from
    # scratch (~96KB/token on the 122B: a 123k-token context is ~12GB —
    # the old 8GB cap never held it, costing a 5-minute prefill per call).
    # 16GB = ~136k tokens of KV; with 65GB weights that's the budget for
    # one mega-session on a 128GB Mac (two concurrent ones risk Metal OOM,
    # which kills the generation thread until restart).
    prompt_cache_bytes: int
    load_timeout: int  # seconds to wait for weights to load
    # End-to-end request/stream budget for local inference. Long prefills can
    # exceed omp's 5-minute stream watchdog; keep the clients and the proxy
    # on the same ceiling.
    client_timeout: int
    # hard context ceiling (tokens) advertised to clients and enforced by
    # the server; the KV-budget clamp still applies if stricter. 0 = off.
    # 110k: the KV ceiling of the 122B-class on a 16GB cache is ~174k, but
    # sessions that actually live near it OOM the machine under pressure.
    max_ctx: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int = 32768
    # llama-server binary for GGUF models: explicit path, or empty = PATH,
    # or the managed download under ~/.mlx-stack/llama.cpp (llamacpp.py)
    llama_server: str = ""

    @property
    def claude_cfg(self) -> Path:
        return self.ollmlx_home / "configs" / "claude.json"

    # HF token for gated models and higher API limits: env takes precedence,
    # fallback is the private file ~/.mlx-stack/hf-token (set via `local-llm token`).
    @property
    def hf_token(self) -> str | None:
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if env and env.strip():
            return env.strip()
        try:
            t = (self.ollmlx_home / "hf-token").read_text().strip()
            return t or None
        except OSError:
            return None

    def model_settings(self, model):
        values = {name: getattr(self, name) for name in (
            "max_ctx", "prompt_cache_bytes", "kv_bits", "kv_group_size",
            "load_timeout", "client_timeout", "temperature", "top_p", "top_k",
            "min_p", "max_tokens")}
        try:
            blob = json.loads((self.ollmlx_home / "model-settings.json").read_text())
            values.update(blob.get(model, {}))
        except (OSError, ValueError):
            pass
        values["temperature"] = min(1.0, float(values["temperature"]))
        return values

    def save_model_settings(self, model, values):
        path = self.ollmlx_home / "model-settings.json"
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            blob = {}
        allowed = ("max_ctx", "prompt_cache_bytes", "kv_bits", "kv_group_size",
                   "load_timeout", "client_timeout", "temperature", "top_p",
                   "top_k", "min_p", "max_tokens")
        blob[model] = {name: values[name] for name in allowed}
        self.ollmlx_home.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, sort_keys=True))
        tmp.replace(path)

    def model_environment(self, model, values=None):
        values = values or self.model_settings(model)
        return {
            "MLX_MAX_CTX": str(values["max_ctx"]),
            "MLX_PROMPT_CACHE_BYTES": str(values["prompt_cache_bytes"]),
            "MLX_KV_BITS": str(values["kv_bits"]),
            "MLX_KV_GROUP_SIZE": str(values["kv_group_size"]),
            "LOAD_TIMEOUT": str(values["load_timeout"]),
            "CLIENT_TIMEOUT": str(values["client_timeout"]),
            "OLLMLX_TEMP": str(values["temperature"]),
            "OLLMLX_TOP_P": str(values["top_p"]),
            "OLLMLX_TOP_K": str(values["top_k"]),
            "OLLMLX_MIN_P": str(values["min_p"]),
            "OLLMLX_MAX_TOKENS": str(values["max_tokens"]),
        }

    def save_context(self):
        path = self.ollmlx_home / "context-settings.json"
        tmp = path.with_suffix(".tmp")
        self.ollmlx_home.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"max_ctx": self.max_ctx,
                                   "prompt_cache_bytes": self.prompt_cache_bytes}))
        tmp.replace(path)

    @classmethod
    def from_env(cls) -> "Config":
        home = stack_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "configs").mkdir(parents=True, exist_ok=True)
        try:
            overrides = json.loads((home / "context-settings.json").read_text())
        except (OSError, ValueError):
            overrides = {}
        max_ctx = int(os.environ.get("MLX_MAX_CTX", overrides.get("max_ctx", "110000")))
        prompt_cache_bytes = int(os.environ.get(
            "MLX_PROMPT_CACHE_BYTES", overrides.get("prompt_cache_bytes", str(16 * 1024**3))))
        return cls(
            ollmlx_home=home,
            hf_hub=Path(
                os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
            ).expanduser(),
            mlx_port=int(os.environ.get("MLX_PORT", "19797")),
            proxy_port=int(os.environ.get("PROXY_PORT", "4000")),
            widget_port=int(os.environ.get("WIDGET_PORT", "19798")),
            api_key="sk-local-llm",
            small_model=os.environ.get(
                "OLLMLX_SMALL_MODEL", "mlx-community/Qwen3-0.6B-4bit"),
            mlx_small_port=int(os.environ.get("MLX_SMALL_PORT", "8081")),
            kv_bits=int(os.environ.get("MLX_KV_BITS", "0")),
            kv_group_size=int(os.environ.get("MLX_KV_GROUP_SIZE", "64")),
            prompt_cache_bytes=prompt_cache_bytes,
            load_timeout=int(os.environ.get("LOAD_TIMEOUT", "900")),
            client_timeout=int(os.environ.get("CLIENT_TIMEOUT", "1800")),
            max_ctx=max_ctx,
            temperature=float(os.environ.get("OLLMLX_TEMP", "0.0")),
            top_p=float(os.environ.get("OLLMLX_TOP_P", "1.0")),
            top_k=int(os.environ.get("OLLMLX_TOP_K", "0")),
            min_p=float(os.environ.get("OLLMLX_MIN_P", "0.0")),
            max_tokens=int(os.environ.get("OLLMLX_MAX_TOKENS", "32768")),
            llama_server=os.environ.get("OLLMLX_LLAMA_SERVER", ""),
        )
