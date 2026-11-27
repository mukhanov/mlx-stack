"""Client configs: pi/omp (models.json, provider "ollmlx"), opencode
(provider merged into opencode.json), claude (claude.json with env
pointing at the client proxy) and dsh (an overlay declaring the proxy as a
custom model API). Canonical copies live in ~/.mlx-stack/configs;
the harness's own path is a symlink so the tool keeps reading its expected
file."""
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import hf, ui
from .config import LOCAL_ALIAS, SMALL_ALIAS
from .models import GGUF_LORAS
from .net import HOST

_ROLE_KEY = re.compile(r"^\s+([^\s#\"'][^:]*):\s*(.*?)\s*$")
CONFIG_CONSENT_ENV = "MLX_STACK_CONFIG_WRITE_CONSENT"


def _config_paths(cfg) -> list[Path]:
    """Every path the writer may create, replace or remove."""
    configs = cfg.home / "configs"
    opencode = (Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
                / "opencode" / "opencode.json")
    return [
        configs / "pi-models.json",
        configs / "omp-models.json",
        configs / "omp-config.yml",
        configs / "opencode.json",
        cfg.claude_cfg,
        configs / "dsh-ollmlx.yml",
        Path("~/.pi/agent/models.json").expanduser(),
        Path("~/.omp/agent/models.json").expanduser(),
        Path("~/.omp/agent/config.yml").expanduser(),
        Path("~/.omp/agent/models.yml").expanduser(),
        opencode,
    ]


def config_preview(cfg) -> dict:
    """Paths shown before the user consents; no config is read or changed."""
    return {
        "files": [str(path) for path in dict.fromkeys(_config_paths(cfg))],
        "backup_root": str(cfg.home / "config-backups"),
    }


def confirm_config_writes(cfg) -> bool:
    """CLI consent; the app sends CONFIG_CONSENT_ENV after its own preview."""
    if os.environ.get(CONFIG_CONSENT_ENV) == "1":
        return True
    preview = config_preview(cfg)
    ui.info("Agent configuration changes require approval:")
    for path in preview["files"]:
        print(f"  {path}")
    print(f"  backups: {preview['backup_root']}")
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("Back up and update these agent configuration files? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower().startswith("y")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if path.exists() and not path.is_symlink():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _atomic_copy(source: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _replace_symlink(external: Path, target: Path) -> None:
    external.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{external.name}.", dir=external.parent)
    os.close(fd)
    Path(tmp).unlink()
    try:
        Path(tmp).symlink_to(target)
        os.replace(tmp, external)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _load_manifest(cfg, backup: Path) -> dict:
    manifest = json.loads((backup / "manifest.json").read_text())
    allowed = {str(path) for path in _config_paths(cfg)}
    if (manifest.get("version") != 1
            or not isinstance(manifest.get("files"), list)
            or any(row.get("path") not in allowed for row in manifest["files"])):
        raise ValueError(f"invalid agent config backup: {backup}")
    return manifest


def _backup_configs(cfg, model: str) -> Path:
    root = cfg.home / "config-backups"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(tempfile.mkdtemp(prefix=stamp + "-", dir=root))
    rows = []
    try:
        for index, path in enumerate(dict.fromkeys(_config_paths(cfg))):
            row = {"path": str(path)}
            if path.is_symlink():
                row.update(kind="symlink", target=os.readlink(path))
            elif path.exists():
                if not path.is_file():
                    raise ValueError(f"not a config file or symlink: {path}")
                stored = backup / f"{index}.config"
                shutil.copy2(path, stored)
                row.update(kind="file", stored=stored.name)
            else:
                row["kind"] = "absent"
            rows.append(row)
        _atomic_write(backup / "manifest.json", json.dumps({
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "files": rows,
        }, indent=2) + "\n")
        return backup
    except BaseException:
        shutil.rmtree(backup, ignore_errors=True)
        raise


def _restore_backup(cfg, backup: Path) -> None:
    manifest = _load_manifest(cfg, backup)
    for row in manifest["files"]:
        path = Path(row["path"])
        kind = row["kind"]
        if kind == "file":
            _atomic_copy(backup / row["stored"], path)
        elif kind == "symlink":
            _replace_symlink(path, Path(row["target"]))
        elif kind == "absent":
            if path.is_dir() and not path.is_symlink():
                raise ValueError(f"refusing to remove a directory: {path}")
            path.unlink(missing_ok=True)
        else:
            raise ValueError(f"unknown config backup entry: {kind}")


def _backup_summary(manifest: dict, backup: Path) -> dict:
    return {"created_at": manifest["created_at"], "model": manifest["model"],
            "path": str(backup)}


def latest_config_backup(cfg) -> dict | None:
    root = cfg.home / "config-backups"
    try:
        candidates = sorted(root.iterdir(), reverse=True)
    except OSError:
        return None
    for backup in candidates:
        try:
            return _backup_summary(_load_manifest(cfg, backup), backup)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def rollback_latest_config_backup(cfg) -> dict:
    root = cfg.home / "config-backups"
    candidates = sorted(root.iterdir(), reverse=True) if root.exists() else []
    for backup in candidates:
        try:
            manifest = _load_manifest(cfg, backup)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        _restore_backup(cfg, backup)
        summary = _backup_summary(manifest, backup)
        shutil.rmtree(backup)
        return summary
    raise SystemExit("no agent configuration backups")


def _config_target(cfg, external: Path, name: str) -> Path:
    """The canonical copy of a harness config. A file already at the
    harness's own path becomes its first content."""
    target = cfg.home / "configs" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        if external.exists():
            _atomic_copy(external, target)
        else:
            _atomic_write(target, "")
    elif external.exists() and not external.is_symlink():
        if external.read_bytes() != target.read_bytes():
            raise ValueError(
                f"{external} and {target} differ; reconcile them or restore "
                "the symlink before loading a model")
    return target


def _ensure_link(external: Path, target: Path) -> None:
    if external.is_symlink() and external.resolve() == target.resolve():
        return
    _replace_symlink(external, target)


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
        _atomic_write(target, "\n".join(edited) + "\n")
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


def _write_client_configs(cfg, model: str, small: bool, max_input: int) -> None:
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
        _atomic_write(target, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        _ensure_link(external, target)
        ui.ok(str(external))

    # omp caches the catalog in models.yml after first reading models.json — reset it
    omp_cache = Path("~/.omp/agent/models.yml").expanduser()
    if omp_cache.exists():
        omp_cache.unlink()
        ui.ok(f"{omp_cache} (cached)")
    _write_omp_smol_role(cfg, small)
    _write_opencode(cfg, model, small, ctx)
    _write_dsh(cfg, model, small, ctx)

    cfg.claude_cfg.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(cfg.claude_cfg, json.dumps({
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
    _atomic_write(target, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _ensure_link(external, target)
    ui.ok(str(external).replace(str(Path.home()), "~"))


def _write_dsh(cfg, model: str, small: bool, ctx: int) -> None:
    """DeepSeek Harness: an overlay that adds the proxy as a custom model
    API provider (llm-pi-ai). Applied with --patch, so dsh's own
    cordis.patch.yml is never rewritten. The static local key travels as an
    Authorization header, so no environment variable is needed."""
    entries = [(model, f"{model.split('/')[-1]} (local MLX)", ctx),
               (LOCAL_ALIAS, "Local MLX (whatever is running)", ctx)]
    if small:
        entries.append((SMALL_ALIAS, "Local MLX small — fast helper"
                        " (background calls, titles)",
                        hf.model_ctx(cfg.hf_hub, cfg.small_model)))
    models = "\n".join(
        f"          - id: {mid}\n"
        f"            name: {name}\n"
        f"            contextWindow: {window}\n"
        f"            maxTokens: {min(window, 32768)}\n"
        f"            input: [text]"
        for mid, name, window in entries)
    target = cfg.home / "configs" / "dsh-ollmlx.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, f"""# MLX Stack as a custom model API in DeepSeek Harness.
# Apply with: dsh web --patch {str(target).replace(str(Path.home()), "~")}
- id: llm-pi-ai
  config:
    providers:
      ollmlx:
        api: openai-completions
        baseURL: http://{HOST}:{cfg.proxy_port}/v1
        headers:
          # the proxy's static local key; no environment variable is needed
          Authorization: Bearer {cfg.api_key}
        models:
{models}
""")
    ui.ok(str(target))


def write_client_configs(cfg, model: str, small: bool, max_input: int) -> Path:
    """Back up every touched path; restore all of them if any write fails."""
    backup = _backup_configs(cfg, model)
    try:
        _write_client_configs(cfg, model, small, max_input)
    except BaseException:
        _restore_backup(cfg, backup)
        raise
    ui.ok(f"agent config backup: {backup}")
    return backup


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
        {"name": "dsh", "command":
            f"dsh web --patch {str(cfg.home / 'configs' / 'dsh-ollmlx.yml').replace(home, '~')}",
         "config_path": str(cfg.home / "configs" / "dsh-ollmlx.yml"),
         "api_base": base, "model": LOCAL_ALIAS,
         "role": "Custom model API (llm-pi-ai overlay)"},
    ]
