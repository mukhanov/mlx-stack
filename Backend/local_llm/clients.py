"""Client configs: pi/omp (models.json, provider "ollmlx"), opencode
(provider merged into opencode.json) and claude (claude.json with env
pointing at the client proxy). Canonical copies live in
~/.mlx-stack/configs; the harness's own path is a symlink so the tool keeps
reading its expected file.
Clients are always launched the same way."""
import json
import os
from pathlib import Path

from . import hf, ui
from .models import GGUF_LORAS


def _config_target(cfg, external: Path, name: str) -> Path:
    target = cfg.ollmlx_home / "configs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if external.exists():
            import shutil
            shutil.copy2(external, target)
        else:
            target.touch()
    return target


def _ensure_link(external: Path, target: Path) -> None:
    external.parent.mkdir(parents=True, exist_ok=True)
    if external.is_symlink():
        if external.resolve() != target.resolve():
            external.unlink()
            external.symlink_to(target)
    elif external.exists():
        external.unlink()
        external.symlink_to(target)
    else:
        external.symlink_to(target)


def _write_omp_smol_role(small: bool, cfg) -> None:
    """Point omp's smol role at the tiny helper; pi has no such role hook."""
    external = Path("~/.omp/agent/config.yml").expanduser()
    target = _config_target(cfg, external, "omp-config.yml")
    want = "ollmlx/small" if small else None
    _ensure_link(external, target)
    try:
        import yaml
    except ImportError:
        ui.warn("pyyaml unavailable — skip ~/.omp/agent/config.yml (smol role)")
        return
    data = {}
    if target.exists():
        try:
            loaded = yaml.safe_load(target.read_text())
            data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            data = {}
    roles = data.get("modelRoles")
    roles = dict(roles) if isinstance(roles, dict) else {}
    if want:
        roles["smol"] = want
    elif roles.get("smol") == "ollmlx/small":
        roles.pop("smol")
    else:
        return
    if roles:
        data["modelRoles"] = roles
    else:
        data.pop("modelRoles", None)
    target.write_text(yaml.safe_dump(data, sort_keys=False))
    _ensure_link(external, target)
    ui.ok(f"{external} (smol role)")


def _entry(mid: str, name: str, ctx: int, reasoning: bool) -> dict:
    return {
        "id": mid,
        "name": name,
        "reasoning": reasoning,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": ctx,
        "maxTokens": min(ctx, 32768),
    }


def write_client_configs(cfg, model: str, small: bool = True,
                         max_input: int | None = None) -> None:
    # the advertised context is the KV-safe budget, not the config.json
    # maximum: past the prompt-cache ceiling the model dies on the next
    # cache extension (Metal OOM), so clients must compact earlier
    if max_input is None:
        max_input = hf.safe_context(cfg.hf_hub, model, cfg.prompt_cache_bytes,
                                    cfg.max_ctx, kv_bits=cfg.kv_bits)
    ctx = max_input
    reasoning = "Qwen3" in model and model not in GGUF_LORAS
    ui.info("Writing client configs")

    # pi/omp have no small/fast-model hook — the tiny helper is exposed as a
    # plain catalog entry (`--model ollmlx/small`, subagents, omp modes)
    models = [
        _entry(model, f"{model.split('/')[-1]} (local MLX)", ctx, reasoning),
        _entry("local", "Local MLX (whatever is running)", ctx, reasoning),
    ]
    if small:
        # no thinking mode: these calls want latency, not reasoning depth
        models.append(_entry("small", "Local MLX small — fast helper"
                             " (background calls, titles)",
                             hf.model_ctx(cfg.hf_hub, cfg.small_model), False))

    for external, filename in (
            (Path("~/.pi/agent/models.json").expanduser(), "pi-models.json"),
            (Path("~/.omp/agent/models.json").expanduser(), "omp-models.json")):
        target = _config_target(cfg, external, filename)
        data = {}
        if target.exists():
            try:
                data = json.loads(target.read_text())
            except Exception:
                data = {}
        providers = data.setdefault("providers", {})
        providers["ollmlx"] = {
            "baseUrl": f"http://127.0.0.1:{cfg.proxy_port}/v1",
            "api": "openai-completions",
            "apiKey": cfg.api_key,
            "models": models,
        }
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        _ensure_link(external, target)
        ui.ok(str(external))

    # omp caches the catalog in models.yml after first reading models.json — reset it
    omp_cache = Path("~/.omp/agent/models.yml").expanduser()
    if omp_cache.exists():
        omp_cache.unlink()
        ui.ok(f"{omp_cache} (cached)")
    _write_omp_smol_role(small, cfg)
    _write_opencode(cfg, model, small, ctx)

    cfg.claude_cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.claude_cfg.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{cfg.proxy_port}",
            "ANTHROPIC_AUTH_TOKEN": cfg.api_key,
            "ANTHROPIC_MODEL": "local",
            "ANTHROPIC_SMALL_FAST_MODEL": "small" if small else "local",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "local",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "local",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "small" if small else "local",
            "API_TIMEOUT_MS": str(cfg.client_timeout * 1000),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }, indent=2) + "\n")
    legacy = cfg.ollmlx_home / "claude-local.json"
    if not legacy.exists():
        legacy.symlink_to("configs/claude.json")


def _write_opencode(cfg, model: str, small: bool, ctx: int) -> None:
    """Register the ollmlx provider in opencode.json, preserving all other
    settings. JSONC is centralized unchanged rather than clobbered."""
    external = (Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
                / "opencode" / "opencode.json")
    target = _config_target(cfg, external, "opencode.json")
    data = {}
    if target.exists():
        try:
            data = json.loads(target.read_text())
        except Exception:
            _ensure_link(external, target)
            ui.warn(f"{external} is not plain JSON (JSONC?) — leaving it"
                    " untouched; add the ollmlx provider by hand")
            return
    models = {"local": {"name": model,
                        "limit": {"context": ctx,
                                  "output": min(ctx, 32768)}}}
    if small:
        models["small"] = {"name": cfg.small_model}
    data.setdefault("provider", {})["ollmlx"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "ollmlx (local MLX)",
        "options": {
            "baseURL": f"http://127.0.0.1:{cfg.proxy_port}/v1",
            "apiKey": cfg.api_key,
            "timeout": cfg.client_timeout * 1000,
            "chunkTimeout": cfg.client_timeout * 1000,
        },
        "models": models,
    }
    if small and data.get("small_model") in (None, "ollmlx/small"):
        data["small_model"] = "ollmlx/small"
    elif not small and data.get("small_model") == "ollmlx/small":
        data.pop("small_model")
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _ensure_link(external, target)
    ui.ok(str(external).replace(str(Path.home()), "~"))


def client_configs(cfg) -> list[dict]:
    """Connection parameters for the Tools screen and snapshot API."""
    home = str(Path.home())
    base = f"http://127.0.0.1:{cfg.proxy_port}/v1"
    return [
        {"name": "pi", "command": "pi --model ollmlx/local",
         "config_path": str(cfg.ollmlx_home / "configs" / "pi-models.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible model catalog"},
        {"name": "omp", "command": "omp --model ollmlx/local",
         "config_path": str(cfg.ollmlx_home / "configs" / "omp-models.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible model catalog"},
        {"name": "opencode", "command": "opencode --model ollmlx/local",
         "config_path": str(cfg.ollmlx_home / "configs" / "opencode.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible provider"},
        {"name": "claude", "command":
            f"claude --settings {str(cfg.claude_cfg).replace(home, '~')}",
         "config_path": str(cfg.claude_cfg),
         "api_base": f"http://127.0.0.1:{cfg.proxy_port}",
         "model": "local", "role": "Anthropic-compatible settings"},
    ]

