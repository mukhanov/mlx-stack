"""Paths, ports and settings. Everything is configured via env (see --help)."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Launch settings of a model and the environment variable that carries each
# one into `local-llm serve`: the app sets them for the model it loads, a
# terminal inherits them from the shell.
SETTING_ENV = {
    "max_ctx": "MLX_MAX_CTX",
    "prompt_cache_bytes": "MLX_PROMPT_CACHE_BYTES",
    "kv_bits": "MLX_KV_BITS",
    "kv_group_size": "MLX_KV_GROUP_SIZE",
    "load_timeout": "LOAD_TIMEOUT",
    "client_timeout": "CLIENT_TIMEOUT",
    "temperature": "OLLMLX_TEMP",
    "top_p": "OLLMLX_TOP_P",
    "top_k": "OLLMLX_TOP_K",
    "min_p": "OLLMLX_MIN_P",
    "max_tokens": "OLLMLX_MAX_TOKENS",
}
FLOAT_SETTINGS = ("temperature", "top_p", "min_p")

# What clients call the two model servers; the proxy maps these to the
# models that are actually loaded.
LOCAL_ALIAS = "local"         # what clients call the main model
SMALL_ALIAS = "small"         # what clients call the helper model

# setting -> accepted range, in the order the app's form is checked. The
# first element is the lowest value, or the few values that are allowed.
_SETTING_LIMITS = (
    ("temperature", 0, 1, "temperature must be between 0 and 1"),
    ("top_p", 0, 1, "top-p and min-p must be between 0 and 1"),
    ("min_p", 0, 1, "top-p and min-p must be between 0 and 1"),
    ("top_k", 0, 1024, "top-k must be between 0 and 1024"),
    ("max_tokens", 256, 262_144, "max tokens must be between 256 and 262144"),
    ("kv_bits", (0, 4, 8), None, "KV bits must be 0, 4 or 8"),
    ("kv_group_size", 16, 1024, "KV group size must be between 16 and 1024"),
    ("load_timeout", 1, None, "timeouts must be positive"),
    ("client_timeout", 1, None, "timeouts must be positive"),
    ("max_ctx", 4096, 262_144, "context must be between 4K and 262K tokens"),
    ("prompt_cache_bytes", 0, 512 * 1024**3,
     "prompt cache must be between 0 and 512 GB"),
)


def stack_home() -> Path:
    """Canonical application home; OLLMLX_HOME remains a legacy override."""
    return Path(os.environ.get("MLX_STACK_HOME")
                or os.environ.get("OLLMLX_HOME")
                or "~/.mlx-stack").expanduser()


def write_json(path: Path, data) -> None:
    """Replace the file atomically: nothing ever reads a half-written one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True))
    tmp.replace(path)


def launch_settings_from(payload: dict) -> dict:
    """Launch settings as the app's form sends them. ValueError names the
    first value that is missing or out of range."""
    try:
        values = {name: float(payload[name]) if name in FLOAT_SETTINGS
                  else int(payload[name]) for name in SETTING_ENV}
    except (KeyError, TypeError, ValueError):
        raise ValueError("expected all model launch settings") from None
    for name, low, high, message in _SETTING_LIMITS:
        value = values[name]
        allowed = value in low if isinstance(low, tuple) else (
            low <= value and (high is None or value <= high))
        if not allowed:
            raise ValueError(message)
    return values


@dataclass
class Config:
    home: Path
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
        return self.home / "configs" / "claude.json"

    # HF token for gated models and higher API limits: env takes precedence,
    # fallback is the private file ~/.mlx-stack/hf-token (set via `local-llm token`).
    @property
    def hf_token(self) -> str | None:
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if env and env.strip():
            return env.strip()
        try:
            token = (self.home / "hf-token").read_text().strip()
            return token or None
        except OSError:
            return None

    def model_settings(self, model: str) -> dict:
        """The model's launch settings: the stack-wide values, overridden by
        what is saved for it."""
        values = {name: getattr(self, name) for name in SETTING_ENV}
        try:
            saved = json.loads((self.home / "model-settings.json").read_text())
            values.update(saved.get(model) or {})
        except (OSError, ValueError, AttributeError):
            pass
        return values

    def save_model_settings(self, model: str, values: dict) -> None:
        path = self.home / "model-settings.json"
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        saved[model] = {name: values[name] for name in SETTING_ENV}
        write_json(path, saved)

    def model_environment(self, model: str, values: dict | None = None) -> dict[str, str]:
        """The model's launch settings as the environment of `local_llm serve`."""
        values = values if values is not None else self.model_settings(model)
        return {env: str(values[name]) for name, env in SETTING_ENV.items()}

    @classmethod
    def from_env(cls) -> "Config":
        home = stack_home()
        (home / "configs").mkdir(parents=True, exist_ok=True)
        try:
            # stack-wide context and prompt-cache limits saved by older versions
            saved = json.loads((home / "context-settings.json").read_text())
        except (OSError, ValueError):
            saved = {}
        return cls(
            home=home,
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
            prompt_cache_bytes=int(os.environ.get(
                "MLX_PROMPT_CACHE_BYTES", saved.get("prompt_cache_bytes", 16 * 1024**3))),
            load_timeout=int(os.environ.get("LOAD_TIMEOUT", "900")),
            client_timeout=int(os.environ.get("CLIENT_TIMEOUT", "1800")),
            max_ctx=int(os.environ.get("MLX_MAX_CTX", saved.get("max_ctx", 110_000))),
            temperature=float(os.environ.get("OLLMLX_TEMP", "0.0")),
            top_p=float(os.environ.get("OLLMLX_TOP_P", "1.0")),
            top_k=int(os.environ.get("OLLMLX_TOP_K", "0")),
            min_p=float(os.environ.get("OLLMLX_MIN_P", "0.0")),
            max_tokens=int(os.environ.get("OLLMLX_MAX_TOKENS", "32768")),
            llama_server=os.environ.get("OLLMLX_LLAMA_SERVER", ""),
        )
