"""Commands and orchestration: serve / stop / list / rm / token."""
import os
import signal
import sys
import threading
import time

import psutil

from . import applog, clients, hf, llmfit, models, servers, ui, widget
from .applog import log
from .config import Config, write_json
from .net import HOST

USAGE = """\
local-llm — the MLX Stack backend: a local model behind OpenAI and Anthropic
APIs on localhost.

Usage:
  local-llm serve <org/model> [quant] [--approve-configs]
                                        start the stack in the foreground
  local-llm <org/model> [quant]         the same, without "serve"
  local-llm list                        downloaded models, size and RAM estimate
  local-llm stop                        stop servers left over from an interrupted run
  local-llm rollback [--yes]            restore the latest agent config backup
  local-llm rm [org/repo…]              delete downloaded models (prompts when
                                        called without arguments)
  local-llm token [hf_… | --clear]      show, save or clear the Hugging Face token
  local-llm widget                      open MLX Stack.app

<org/model> is a Hugging Face repository. MLX weights run on mlx_lm.server,
GGUF weights on llama-server (downloaded on first use). [quant] names a GGUF
quantization — file name, quant name or a substring of it; the default is the
largest that fits free RAM.

serve fetches the model when it is missing, starts the model server (:19797),
the helper model for fast calls (:8081) and the client proxy (:4000), previews
and backs up pi, omp, opencode and Claude Code settings before updating them,
and runs until Ctrl-C or SIGTERM. Interactive starts ask before changing agent
configs; non-interactive starts require --approve-configs.

Environment: MLX_PORT, MLX_SMALL_PORT, PROXY_PORT, WIDGET_PORT,
MLX_STACK_HOME, HF_HUB_CACHE, HF_TOKEN; launch settings MLX_MAX_CTX,
MLX_PROMPT_CACHE_BYTES, MLX_KV_BITS, MLX_KV_GROUP_SIZE, LOAD_TIMEOUT,
CLIENT_TIMEOUT, OLLMLX_TEMP, OLLMLX_TOP_P, OLLMLX_TOP_K, OLLMLX_MIN_P,
OLLMLX_MAX_TOKENS; OLLMLX_SMALL_MODEL (helper model), OLLMLX_LLAMA_SERVER
(path to a llama-server binary; default: a managed download)."""


class LaunchProgress:
    """What the app shows while the stack comes up. The controller writes
    its stage to ~/.mlx-stack/launch.json, the local API hands it to the
    window, and the record goes away once the stack serves."""

    def __init__(self, cfg: Config, model: str):
        self.path = cfg.home / "launch.json"
        self.record = {"model": model, "pid": os.getpid(), "started": time.time()}

    def _write(self, **fields) -> None:
        try:
            write_json(self.path, {**self.record, **fields})
        except OSError:
            pass   # progress is for the window: it must not break a launch

    def stage(self, name: str) -> None:
        self._write(stage=name)

    def done(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def failed(self, reason: str, status: int = 1) -> None:
        self._write(stage="failed", status=status, error=reason)


def _failure(exc: BaseException) -> str:
    """One line for the user on why the stack did not come up."""
    reason = getattr(exc, "reason", None)
    if reason:
        return reason
    if isinstance(exc, SystemExit):
        code = exc.code
        return code.strip() if isinstance(code, str) else f"exit status {code}"
    return f"{type(exc).__name__}: {exc}"[:200]


def main(argv: list[str] | None = None) -> int:
    if argv is None:   # the console script calls main() with no args
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    applog.setup()   # first thing: a crash anywhere must already be logged
    cfg = Config.from_env()
    cmd, rest = argv[0], argv[1:]
    if cmd == "stop":
        return cmd_stop(cfg)
    if cmd == "token":
        return cmd_token(cfg, rest)
    if cmd in ("rm", "remove", "del"):
        return cmd_rm(cfg, rest)
    if cmd in ("list", "ls"):
        return cmd_list(cfg)
    if cmd == "widget":
        if rest:
            raise SystemExit("   usage: local-llm widget")
        return widget.run(cfg)
    if cmd == "rollback":
        return cmd_rollback(cfg, rest)
    if cmd == "serve":
        if "--approve-configs" in rest:
            os.environ[clients.CONFIG_CONSENT_ENV] = "1"
            rest = [arg for arg in rest if arg != "--approve-configs"]
        if not rest or "/" not in rest[0]:
            raise SystemExit(
                "   usage: local-llm serve <org/model> [quant] [--approve-configs]")
        return run_serve(cfg, rest[0], rest[1] if len(rest) > 1 else None)
    if "/" in cmd:
        # second word (when given) is the GGUF quant: file name, quant name
        # or any substring of it — the boot sorts MLX vs GGUF out itself
        if "--approve-configs" in rest:
            os.environ[clients.CONFIG_CONSENT_ENV] = "1"
            rest = [arg for arg in rest if arg != "--approve-configs"]
        return run_serve(cfg, cmd, rest[0] if rest else None)
    raise SystemExit(f"   unknown command '{cmd}' — see local-llm --help")


def _pick_gguf_group(model: str, groups, quant: str | None):
    """(label, [shard names]) of the quant to serve: the user's substring
    (Q4_K_M, q4, full file name), else the largest that fits free RAM with
    headroom — clamping to the smallest beats refusing to start."""
    if quant:
        q = quant.lower()
        for label, _total, names in groups:
            if q in label.lower():
                return label, names
        known = ", ".join(g[0] for g in groups[:8])
        raise SystemExit(f"   no GGUF file matching '{quant}' in {model};"
                         f" available: {known}")
    # 15% weights headroom + 2G floor for KV/runtime, like the crash guard
    budget = psutil.virtual_memory().available - 2 * 1024 ** 3
    for label, total, names in groups:          # largest first
        if total * 1.15 <= budget:
            return label, names
    ui.warn("no GGUF quant fits in free RAM with headroom —"
            " taking the smallest")
    return groups[-1][0], groups[-1][2]


def _gguf_prepare(cfg: Config, model: str, quant: str | None = None):
    """Local path of the GGUF file to serve, downloading just the chosen
    quant when needed. None — not a GGUF model (the MLX path continues).

    Local snapshot first (works offline, and the run before this one
    already chose a quant); the repo listing only when the user asks for
    a quant we don't have yet, or nothing local says GGUF."""
    local = hf.local_gguf_files(cfg.hf_hub, model)
    sizes = {name: size for name, size in local}
    groups = hf.gguf_groups(local)
    need_remote = (not groups
                   or not any(hf.gguf_set_complete(g[2]) for g in groups)
                   or (quant and not any(quant.lower() in g[0].lower()
                                         for g in groups)))
    if need_remote:
        remote = hf.repo_gguf_files(model, cfg.hf_token)
        if remote is None:
            if quant and not groups:
                raise SystemExit(
                    f"   HF API unreachable — can't list {model}'s GGUF"
                    " files; retry, or pass a quant that's already"
                    " downloaded")
        else:
            sizes.update({n: s for n, s in remote if n not in sizes})
            groups = hf.gguf_groups(list(sizes.items()))
    if groups and not any(hf.gguf_set_complete(g[2]) for g in groups):
        # shard2 without shard1 (fetch in flight or dead) must not reach
        # llama-server — it refuses anything but the first split; an empty
        # group list is an MLX model and falls through to the MLX path
        raise SystemExit(f"   {model}: no complete GGUF shard set locally"
                         " and the HF API is unreachable — retry when"
                         " online; a running fetch keeps its progress")
    if not groups:
        if quant:
            ui.warn(f"'{quant}' ignored — {model} has no GGUF files"
                    " (MLX model)")
        return None
    label, names = _pick_gguf_group(model, groups, quant)
    have = {name for name, _size in local}   # local lists base names
    if {n.rsplit("/", 1)[-1] for n in names} <= have:
        ui.ok(f"GGUF {label} already on disk")
    else:
        ui.info(f"Downloading GGUF {label}")
        hf.download(model, cfg.hf_token, cfg.hf_hub, allow_patterns=names)
    path = hf.gguf_local_path(cfg.hf_hub, model, label)
    if path is None:
        raise SystemExit(f"   {model}: {label} not in the snapshot after"
                         " download — see the log above")
    return path


def _boot(cfg: Config, model: str, quant: str | None,
          progress: LaunchProgress) -> bool:
    """Bring the stack up and return True once it is serving: pick and
    fetch the weights, start the proxy and the model servers, write the
    client configs, warm up, start the watchdog — telling `progress` each
    stage on the way. False — the user declined to start under memory
    pressure, so nothing is running. A startup failure raises; the caller
    stops whatever was started."""
    # a port collision is a config error — fail before touching anything
    # (the second server would die on bind and _wait_ready would kill the run)
    if (cfg.mlx_port == cfg.mlx_small_port
            or cfg.proxy_port in (cfg.mlx_port, cfg.mlx_small_port)):
        raise SystemExit(
            f"   port collision: mlx={cfg.mlx_port},"
            f" small={cfg.mlx_small_port}, proxy={cfg.proxy_port}"
            " — all three must differ (MLX_PORT / MLX_SMALL_PORT / PROXY_PORT)")
    if not clients.confirm_config_writes(cfg):
        if not sys.stdin.isatty():
            raise SystemExit(
                "agent configuration approval required; "
                "rerun with --approve-configs")
        ui.ok("cancelled — no agent configuration changed")
        return False
    progress.stage("prepare")
    # whatever an interrupted run left on these ports must not hold RAM
    servers.stop_stale(cfg)
    # GGUF repos run llama-server in the mlx slot; everything above the
    # port (proxy, clients, the desktop app) doesn't care which one it is.
    # A registered LoRA alias downloads its exact base and adapter instead
    # of treating the PEFT repo as standalone MLX weights.
    lora = models.GGUF_LORAS.get(model)
    weights_model = lora["base"] if lora else model
    gguf_path = _gguf_prepare(
        cfg, weights_model, quant or (lora["quant"] if lora else None))
    lora_path = (_gguf_prepare(cfg, lora["repo"], lora["file"])
                 if lora else None)
    backend = "llama" if gguf_path else "mlx"
    log.info("stack start: model=%s backend=%s mlx=:%d small=:%d proxy=:%d",
             model, backend, cfg.mlx_port, cfg.mlx_small_port,
             cfg.proxy_port)
    ui.info(f"Model: {model}"
            + (f" · {gguf_path.name} (GGUF)" if gguf_path else "")
            + (f" + {lora_path.name} (LoRA)" if lora_path else ""))
    if gguf_path:
        # the quant file and optional LoRA are the whole download;
        # _gguf_prepare fetched both
        weights = (gguf_path.stat().st_size
                   + (lora_path.stat().st_size if lora_path else 0))
    else:
        missing = hf.missing_files(cfg.hf_hub, model, cfg.hf_token)
        if missing is None:
            # HF API unreachable: can't verify completeness, trust the local
            # snapshot (all links resolve = best we can tell without network)
            if hf.is_cached(cfg.hf_hub, model):
                ui.ok("already downloaded (completeness not verified —"
                      " HF API unreachable)")
            else:
                ui.info("Downloading")
                hf.download(model, cfg.hf_token, cfg.hf_hub)
                ui.ok("download complete")
        elif missing:
            # partial model: some shards done, hub reuses finished blobs and
            # fetches only the rest
            ui.warn(f"incomplete download: {len(missing)} file(s) missing"
                    f" (first: {missing[0]}) — fetching the rest")
            hf.download(model, cfg.hf_token, cfg.hf_hub)
            missing = hf.missing_files(cfg.hf_hub, model, cfg.hf_token)
            if missing:
                raise SystemExit(f"   still incomplete after download:"
                                 f" {len(missing)} file(s) missing,"
                                 f" e.g. {missing[0]}")
            ui.ok("download complete")
        else:
            ui.ok("already downloaded")
        weights = hf.model_disk_bytes(cfg.hf_hub, model)

    if (not gguf_path
            and hf.packed_format(cfg.hf_hub, model) == "mlx-gsq-packed-v1"):
        servers.ensure_gsq_runtime()

    # tiny helper model for fast/background calls — if it can't be fetched
    # (HF unreachable, cache cold) we degrade instead of blocking the stack:
    # everything then runs on the big model like before
    small_ok = True
    if not hf.is_cached(cfg.hf_hub, cfg.small_model):
        ui.info(f"Downloading helper model {cfg.small_model} (~0.5GB)")
        try:
            hf.download(cfg.small_model, cfg.hf_token, cfg.hf_hub)
        except (Exception, SystemExit):  # noqa: BLE001 — degrade, not die
            pass
        small_ok = hf.is_cached(cfg.hf_hub, cfg.small_model)
        if not small_ok:
            ui.warn("helper model unavailable — continuing without it"
                    " (fast/background calls will use the main model)")

    # memory-pressure guard: weights + heavy swap is exactly how a session
    # dies mid-work (Metal OOM kills the generation thread -> 404s until
    # restart). Warn before loading, not after.
    avail = psutil.virtual_memory().available
    if weights and avail < weights + 4 * 1024**3:
        ui.warn(f"only {ui.human_bytes(avail)} RAM available, the model's"
                f" weights are {ui.human_bytes(weights)} — under this"
                " pressure a Metal OOM kills the server until restart")
        if sys.stdin.isatty():
            try:
                answer = input("?# Start anyway? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if not answer.startswith("y"):
                ui.ok("cancelled — free memory first (`local-llm stop` stops"
                      " leftovers)")
                return False
        else:
            ui.warn("non-interactive start — continuing")

    cache_limit = int(hf.memory_cache_limit(cfg.hf_hub, weights_model) * 0.95)
    if cfg.prompt_cache_bytes > cache_limit:
        ui.info(f"prompt cache capped at {ui.human_bytes(cache_limit)}"
                " to fit available RAM")
        cfg.prompt_cache_bytes = cache_limit
    max_input = hf.safe_context(cfg.hf_hub, weights_model,
                                cfg.prompt_cache_bytes, cfg.max_ctx,
                                kv_bits=cfg.kv_bits)
    if max_input == 0:
        raise SystemExit("not enough available RAM for a 4K context —"
                         " close other apps or choose a smaller model")
    if max_input < hf.model_ctx(cfg.hf_hub, weights_model):
        ui.info(f"model context capped at {max_input} tokens"
                " (prompt-cache budget) — oversized requests are refused,"
                " clients compact before it OOMs")
    state = {"big": None, "small": None, "backend": backend,
             "gguf_path": str(gguf_path) if gguf_path else None,
             "lora_path": str(lora_path) if lora_path else None}
    servers.start_proxy(cfg, model, small_ok)
    progress.stage("weights")
    state["big"] = (servers.start_llama(cfg, state["gguf_path"], model,
                                        max_input, state["lora_path"])
                    if gguf_path
                    else servers.start_mlx(cfg, model, max_input))
    if small_ok:
        progress.stage("helper")
        state["small"] = servers.start_mlx_small(cfg)
    clients.write_client_configs(cfg, model, small_ok, max_input)
    progress.stage("warmup")
    servers.warmup(cfg)
    log.info("stack ready: %s (max_input=%s, helper=%s)", model,
             max_input, cfg.small_model if small_ok else "off")
    servers.supervise(cfg, model, state, max_input=max_input)
    progress.done()
    print()
    ui.info(f"Ready. OpenAI and Anthropic APIs on"
            f" http://{HOST}:{cfg.proxy_port}")
    print(f"  pi:       pi --model ollmlx/local   ({model})")
    print(f"  omp:      omp --model ollmlx/local")
    print(f"  opencode: opencode --model ollmlx/local")
    print(f"  claude:   claude --settings {cfg.claude_cfg}")
    if small_ok:
        print(f"  fast/bg:  ollmlx/small  ({cfg.small_model} on"
              f" :{cfg.mlx_small_port} — titles, quick asks)")
    return True


def run_serve(cfg: Config, model: str, quant: str | None = None) -> int:
    """The stack in the foreground, for terminals and the app alike: same
    boot (quant pick, servers, proxy, client configs, watchdog), then blocks
    until SIGTERM or SIGINT and stops the children cleanly. Port collisions
    and other startup failures exit non-zero."""
    stop = threading.Event()

    def _signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    progress = LaunchProgress(cfg, model)
    try:
        started = _boot(cfg, model, quant, progress)
    except BaseException as exc:  # startup failed: nothing to supervise
        servers.stop_watchdog()
        servers.stop_children()
        status = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        progress.failed(_failure(exc), status)
        if isinstance(exc, SystemExit) and isinstance(exc.code, int):
            return exc.code
        raise
    if not started:
        progress.done()
        return 0
    log.info("serve: stack ready (%s) — waiting for SIGTERM/SIGINT", model)
    print(f"serve: ready — model {model}; OpenAI/Anthropic on"
          f" http://{HOST}:{cfg.proxy_port}; stop with Ctrl-C or SIGTERM",
          flush=True)
    try:
        stop.wait()
    finally:
        # also on SIGHUP: the app-log hook turns it into SystemExit
        log.info("serve: stopping the stack")
        print("serve: stopping", flush=True)
        servers.stop_watchdog()
        servers.stop_children()
    return 0


def cmd_rollback(cfg: Config, args: list[str]) -> int:
    if args not in ([], ["--yes"]):
        raise SystemExit("   usage: local-llm rollback [--yes]")
    latest = clients.latest_config_backup(cfg)
    if latest is None:
        raise SystemExit("no agent configuration backups")
    ui.info(f"Latest agent config backup: {latest['created_at']}"
            f" · {latest['model']}\n  {latest['path']}")
    if "--yes" not in args:
        if not sys.stdin.isatty():
            raise SystemExit("rollback needs confirmation: local-llm rollback --yes")
        try:
            answer = input("Restore these agent configuration files? [y/N] ")
        except EOFError:
            answer = ""
        if not answer.strip().lower().startswith("y"):
            ui.ok("cancelled — agent configs unchanged")
            return 0
    restored = clients.rollback_latest_config_backup(cfg)
    ui.ok(f"agent configs restored from {restored['path']}")
    return 0


def cmd_stop(cfg: Config) -> int:
    ui.info("Stopping leftover servers")
    ui.ok("servers stopped" if servers.stop_stale(cfg)
          else "no servers of the stack were running")
    return 0


def cmd_token(cfg: Config, args: list[str]) -> int:
    """Show/save/clear the HF token. Precedence: env HF_TOKEN > file."""
    token_file = cfg.home / "hf-token"
    if args and args[0] == "--clear":
        if token_file.exists():
            token_file.unlink()
            ui.ok("token deleted")
        else:
            ui.ok("there was no saved token")
        return 0
    if args:
        cfg.home.mkdir(parents=True, exist_ok=True)
        token_file.write_text(args[0].strip() + "\n")
        token_file.chmod(0o600)
        ui.ok(f"token saved to {token_file} (chmod 600)")
        return 0
    token = cfg.hf_token
    if not token:
        ui.warn("no token set: local-llm token <hf_...> or env HF_TOKEN")
        return 0
    from_env = bool(os.environ.get("HF_TOKEN") or
                     os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    src = "env HF_TOKEN" if from_env else str(token_file)
    masked = f"{token[:5]}…{token[-4:]}" if len(token) > 12 else "…"
    ui.info(f"HF token: {masked} (source: {src})")
    return 0


def cmd_list(cfg: Config) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    if not installed:
        ui.warn(f"No downloaded LLM models ({cfg.hf_hub})")
        return 0
    lf = llmfit.load(cfg.home) or {}
    sys_ram = ui.total_ram_gb()
    ui.info("💿 Installed models:")
    for rid, size, dling in installed:
        mark = "⬇ " if dling else "  "
        entry = lf.get(rid)
        ram = models.ram_estimate_gb(rid, entry, size=size)
        moe = " · MoE" if models.parse_model(rid)[4] else ""
        ctx = llmfit.ctx_str(entry) if entry else None
        lfsc = llmfit.score(entry, sys_ram) if entry else None
        extra = (f" · ctx {ctx}" if ctx else "")
        extra += f" · ⚡{lfsc['score']:.0f}" if lfsc else ""
        extra += " · 🔒 helper" if rid == cfg.small_model else ""
        warn = "  ⚠won't fit" if ram and ram > sys_ram else ""
        ram_txt = f"~{ram}GB RAM" if ram else "RAM ?"
        print(f"  {mark}{rid:<62} {size:>6}  {ram_txt}{moe}{extra}{warn}")
    return 0


def cmd_rm(cfg: Config, args: list[str]) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    if not installed:
        ui.warn(f"No downloaded LLM models ({cfg.hf_hub})")
        return 0
    ids = [t[0] for t in installed]

    if args:
        targets = list(args)
    else:
        ui.info("💿 Installed models:")
        for n, (rid, size, dling) in enumerate(installed, 1):
            print(f"  {n:2d}) {'⬇ ' if dling else '  '}{rid:<60} {size}")
        try:
            raw = input("?# Delete (numbers or org/repo, space-separated,"
                        " empty — cancel): ").strip()
        except EOFError:
            raw = ""
        if not raw:
            ui.ok("cancelled")
            return 0
        targets = []
        for t in raw.split():
            if t.isdigit():
                idx = int(t) - 1
                if 0 <= idx < len(ids):
                    targets.append(ids[idx])
                else:
                    ui.warn(f"no item #{t}")
            elif "/" in t:
                targets.append(t)
            else:
                ui.warn(f"can't parse: {t} (need a number or org/repo)")

    if not targets:
        ui.ok("nothing selected")
        return 0

    for d in targets:
        if "/" not in d:
            ui.warn(f"{d} — expected org/repo")
            continue
        if d == cfg.small_model:
            ui.warn(f"{d} — helper model for fast/background calls;"
                    " point OLLMLX_SMALL_MODEL elsewhere first")
            continue
        if not hf.model_dir(cfg.hf_hub, d).is_dir():
            ui.warn(f"{d} — not found in cache")
            continue
        size = next((s for i, s, _ in installed if i == d), "?")
        try:
            answer = input(f"?# Delete {d} ({size})? [y/N] ").strip()
        except EOFError:
            ui.ok("cancelled")
            return 0
        if answer.lower().startswith("y"):
            try:
                hf.delete_model(cfg.hf_hub, d)
                ui.ok(f"deleted: {d} (freed {size})")
            except OSError as exc:
                ui.warn(f"could not delete: {exc}")
        else:
            ui.ok(f"skipped: {d}")
    return 0


if __name__ == "__main__":  # python -m local_llm.cli
    sys.exit(main(sys.argv[1:]))
