"""Client configs: pi/omp (models.json, provider "ollmlx"), opencode
(provider merged into opencode.json) and claude (claude.json with env
pointing at the client proxy). Canonical copies live in ~/.mlx-stack/configs;
the harness's own path is a symlink so the tool keeps reading its expected
file."""
import json
import os
import re
import shutil
from pathlib import Path

from . import hf, ui
from .config import LOCAL_ALIAS, SMALL_ALIAS
from .models import GGUF_LORAS
from .net import HOST

_ROLE_KEY = re.compile(r"^\s+([^\s#\"'][^:]*):\s*(.*?)\s*$")


def _config_target(cfg, external: Path, name: str) -> Path:
    """The canonical copy of a harness config. A file already at the
    harness's own path becomes its first content."""
    target = cfg.home / "configs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if external.exists():
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


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _set_smol_role(lines: list[str], role: str, enabled: bool) -> list[str] | None:
    """The config's lines with omp's modelRoles.smol pointing at `role`.
    With `enabled` off, that line is dropped when it held `role`, and an
    emptied modelRoles goes with it. None when modelRoles is not a plain
    block mapping — such a file is left to its owner."""
    header = next((i for i, line in enumerate(lines)
                   if line.startswith("modelRoles:")), None)
    if header is None:
        return ([*lines, "modelRoles:", f"  smol: {role}"] if enabled
                else list(lines))
    if not re.fullmatch(r"modelRoles:\s*(#.*)?", lines[header]):
        return None
    end = header + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
        end += 1
    body = lines[header + 1:end]
    at = next((i for i, line in enumerate(body)
               if (m := _ROLE_KEY.match(line)) and m[1] == "smol"), None)
    value = _ROLE_KEY.match(body[at])[2].strip("'\"") if at is not None else None
    if enabled:
        entry = f"  smol: {role}"
        if at is None:
            last = max((i for i, line in enumerate(body) if line.strip()),
                       default=-1)
            body.insert(last + 1, entry)
        elif value != role:
            body[at] = entry
    elif at is not None and value == role:
        body.pop(at)
    if any(line.strip() for line in body):
        return lines[:header + 1] + body + lines[end:]
    return lines[:header] + lines[end:]


def _write_omp_smol_role(cfg, small: bool) -> None:
    """Point omp's smol role at the helper model; pi has no such role hook."""
    external = Path("~/.omp/agent/config.yml").expanduser()
    target = _config_target(cfg, external, "omp-config.yml")
    role = f"ollmlx/{SMALL_ALIAS}"
    lines = target.read_text().splitlines()
    edited = _set_smol_role(lines, role, small)
    if edited is None:
        ui.warn(f"{external} does not keep modelRoles as a plain block —"
                f" add `smol: {role}` under it by hand")
    elif edited != lines:
        target.write_text("\n".join(edited) + "\n")
        ui.ok(f"{external} (smol role)")
    _ensure_link(external, target)


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


def write_client_configs(cfg, model: str, small: bool, max_input: int) -> None:
    # the advertised context is the KV-safe budget, not the config.json
    # maximum: past the prompt-cache ceiling the model dies on the next
    # cache extension (Metal OOM), so clients must compact earlier
    ctx = max_input
    reasoning = "Qwen3" in model and model not in GGUF_LORAS
    ui.info("Writing client configs")

    # pi/omp have no small/fast-model hook — the tiny helper is exposed as a
    # plain catalog entry (`--model ollmlx/small`, subagents, omp modes)
    models = [
        _entry(model, f"{model.split('/')[-1]} (local MLX)", ctx, reasoning),
        _entry(LOCAL_ALIAS, "Local MLX (whatever is running)", ctx, reasoning),
    ]
    if small:
        # no thinking mode: these calls want latency, not reasoning depth
        models.append(_entry(SMALL_ALIAS, "Local MLX small — fast helper"
                             " (background calls, titles)",
                             hf.model_ctx(cfg.hf_hub, cfg.small_model), False))

    for external, filename in (
            (Path("~/.pi/agent/models.json").expanduser(), "pi-models.json"),
            (Path("~/.omp/agent/models.json").expanduser(), "omp-models.json")):
        target = _config_target(cfg, external, filename)
        data = _read_json(target)
        data.setdefault("providers", {})["ollmlx"] = {
            "baseUrl": f"http://{HOST}:{cfg.proxy_port}/v1",
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
    _write_omp_smol_role(cfg, small)
    _write_opencode(cfg, model, small, ctx)

    cfg.claude_cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.claude_cfg.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": f"http://{HOST}:{cfg.proxy_port}",
            "ANTHROPIC_AUTH_TOKEN": cfg.api_key,
            "ANTHROPIC_MODEL": LOCAL_ALIAS,
            "ANTHROPIC_SMALL_FAST_MODEL": SMALL_ALIAS if small else LOCAL_ALIAS,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": LOCAL_ALIAS,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": LOCAL_ALIAS,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": SMALL_ALIAS if small else LOCAL_ALIAS,
            "API_TIMEOUT_MS": str(cfg.client_timeout * 1000),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }, indent=2) + "\n")


def _write_opencode(cfg, model: str, small: bool, ctx: int) -> None:
    """Register the ollmlx provider in opencode.json, preserving all other
    settings. JSONC is left unchanged rather than clobbered."""
    external = (Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
                / "opencode" / "opencode.json")
    target = _config_target(cfg, external, "opencode.json")
    text = target.read_text()   # _config_target made an empty one if there was none
    try:
        data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        _ensure_link(external, target)
        ui.warn(f"{external} is not plain JSON (JSONC?) — leaving it"
                " untouched; add the ollmlx provider by hand")
        return
    models = {LOCAL_ALIAS: {"name": model,
                            "limit": {"context": ctx,
                                      "output": min(ctx, 32768)}}}
    if small:
        models[SMALL_ALIAS] = {"name": cfg.small_model}
    data.setdefault("provider", {})["ollmlx"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "ollmlx (local MLX)",
        "options": {
            "baseURL": f"http://{HOST}:{cfg.proxy_port}/v1",
            "apiKey": cfg.api_key,
            "timeout": cfg.client_timeout * 1000,
            "chunkTimeout": cfg.client_timeout * 1000,
        },
        "models": models,
    }
    helper = f"ollmlx/{SMALL_ALIAS}"
    if small and data.get("small_model") in (None, helper):
        data["small_model"] = helper
    elif not small and data.get("small_model") == helper:
        data.pop("small_model")
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _ensure_link(external, target)
    ui.ok(str(external).replace(str(Path.home()), "~"))


def client_configs(cfg) -> list[dict]:
    """Connection parameters for the Tools screen and the snapshot."""
    home = str(Path.home())
    base = f"http://{HOST}:{cfg.proxy_port}/v1"
    return [
        {"name": "pi", "command": "pi --model ollmlx/local",
         "config_path": str(cfg.home / "configs" / "pi-models.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible model catalog"},
        {"name": "omp", "command": "omp --model ollmlx/local",
         "config_path": str(cfg.home / "configs" / "omp-models.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible model catalog"},
        {"name": "opencode", "command": "opencode --model ollmlx/local",
         "config_path": str(cfg.home / "configs" / "opencode.json"),
         "api_base": base, "model": "ollmlx/local",
         "role": "OpenAI-compatible provider"},
        {"name": "claude", "command":
            f"claude --settings {str(cfg.claude_cfg).replace(home, '~')}",
         "config_path": str(cfg.claude_cfg),
         "api_base": f"http://{HOST}:{cfg.proxy_port}",
         "model": LOCAL_ALIAS, "role": "Anthropic-compatible settings"},
    ]
