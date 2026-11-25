"""Lifecycle of mlx_lm.server (:19797, OpenAI) and the client proxy (:4000,
OpenAI + Anthropic bridge): start, readiness wait, warmup, stop, plus a
watchdog that restarts mlx after a crash or the post-OOM zombie.

The servers run as foreground children of our process (no nohup/daemons):
stopping the stack (SIGTERM, SIGHUP or Ctrl-C) kills everything at once
(see cli.run_serve)."""
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import hf, proxy, ui
from .applog import log, logs_dir

_LOGS = logs_dir()   # persistent: /tmp is wiped on reboot, taking the
# evidence of a crash with it — ~/.mlx-stack/logs survives
MLX_LOG = str(_LOGS / "mlx-server.log")
MLX_SMALL_LOG = str(_LOGS / "mlx-small.log")
LLAMA_LOG = str(_LOGS / "llama-server.log")
PROXY_LOG = str(_LOGS / "proxy.log")
WATCHDOG_LOG = str(_LOGS / "mlx-watchdog.log")

# Popen handles of live children — killed in stop_children()
children: list[subprocess.Popen] = []


# our wrapper around mlx_lm.server (tok/s logging) or a raw server from an
# older local-llm — `stop` must kill either
_MLX_PATTERN = r"local_llm\.mlxwrap|mlx_lm\.server"


def _llama_pattern(port: int) -> str:
    """Match string for OUR llama-server child. The port is part of the
    pattern on purpose: `pkill -f llama-server` alone would take down an
    unrelated llama-server the user runs for something else."""
    return rf"llama-server.*--port {port}"


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def tail(path: str, n: int = 30):
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _pkill(pattern: str) -> bool:
    return subprocess.run(["pkill", "-f", pattern],
                          capture_output=True).returncode == 0


def _open_log(path: str):
    """Keep one generation of history: the previous run's log moves to
    path.1 before the fresh one truncates it — otherwise every restart
    destroys the evidence of whatever killed the run before it."""
    try:
        os.replace(path, path + ".1")
    except OSError:
        pass  # no previous log (or unwritable dir) — not fatal
    return open(path, "wb")


def stop_all(llama_port: int = 0) -> None:
    """For `local-llm stop`: kill processes leaked from a previous run."""
    ui.info("Stopping servers")
    ui.ok("mlx_lm.server stopped" if _pkill(_MLX_PATTERN)
          else "mlx_lm.server not running")
    if llama_port:
        ui.ok("llama-server stopped" if _pkill(_llama_pattern(llama_port))
              else "llama-server not running")


def stop_children() -> None:
    for p in children:
        if p.poll() is None:
            p.terminate()
    for p in children:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


# --- watchdog --------------------------------------------------------------------
#
# mlx dies in two ways: the process exits, or a Metal OOM kills the
# generation thread while the HTTP server lives on — the post-OOM zombie
# (seen 2026-10-05 23:58 with a 186k-token session): /v1/models keeps
# answering 200 and every completion 404s until restarted. The watchdog
# catches both and brings the model back without user intervention.

_watchdog_stop = threading.Event()


def _note(msg: str) -> None:
    """An event line in the watchdog's own log — the monitor tails it.
    NOT the mlx log: the server child holds that file at its own non-append
    offset and overwrites whatever anyone else appends after it (the
    13:19:35 restart note was lost exactly this way)."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(WATCHDOG_LOG, "a") as f:
            f.write(f"{stamp} - WATCHDOG - {msg}\n")
    except OSError:
        pass
    log.warning("watchdog: %s", msg)


def _probe_generation(port: int, model: str, timeout: float = 30.0) -> str:
    """'ok' | 'bad' | 'busy' — health is a real 1-token completion, not
    /v1/models (that one stays 200 in the zombie state). 'busy' = no
    answer in time: mid-prefill of a huge prompt, not proof of anything."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": model, "max_tokens": 1, "stream": False,
                         "messages": [{"role": "user", "content": "ping"}]}
                        ).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return "ok"
    except urllib.error.HTTPError:
        return "bad"     # 404/500 — the zombie signature
    except (OSError, urllib.error.URLError):
        return "busy"


def supervise(cfg, model: str, state: dict, interval: float = 45.0,
              loop_window: float = 600.0, loop_max: int = 3,
              backoff: float = 600.0,
              max_input: int | None = None) -> None:
    """Start the watchdog thread. `state` = {"big": Popen, "small": Popen|None,
    "backend": "mlx"|"llama", ...}; on a dead process or 2 consecutive bad
    probes it restarts the model server(s) in place — the client proxy and
    the client configs keep pointing at the same ports, so clients just see a
    pause.

    loop_max restarts within loop_window = the client session is bigger
    than this machine can serve (every restart the client resends its full
    context and the model OOMs again) — probing slows to `backoff` and a
    note in the watchdog log says to compact the session. Restarts never
    stop; when the window drains, normal cadence resumes."""
    _watchdog_stop.clear()
    restarts: list[float] = []

    def restart(reason: str) -> None:
        llama = state.get("backend") == "llama"
        if llama:
            # only our llama-server (the port keeps unrelated ones alive);
            # the MLX helper is a separate child and stays up
            _pkill(_llama_pattern(cfg.mlx_port))
        else:
            _pkill(_MLX_PATTERN)   # takes the helper down too; it comes back
        time.sleep(1)
        try:
            state["big"] = (start_llama(cfg, state["gguf_path"], model,
                                        max_input, state.get("lora_path"))
                            if llama
                            else start_mlx(cfg, model, max_input))
            if not llama and state["small"] is not None:
                state["small"] = start_mlx_small(cfg)
            _note(f"{'llama-server' if llama else 'mlx'} restarted"
                  f" ({reason}) — stack healthy again")
        except SystemExit as exc:
            _note(f"restart failed ({reason}): {exc}")

    def run():
        bad = 0
        iv = interval
        loop_mode = False
        while not _watchdog_stop.wait(iv):
            now = time.monotonic()
            while restarts and now - restarts[0] > loop_window:
                restarts.pop(0)

            big = state.get("big")
            dead = big is None or big.poll() is not None
            if not dead:
                if _probe_generation(cfg.mlx_port, model) != "bad":
                    bad = 0
                    continue
                bad += 1
                if bad < 2:
                    continue
                reason = ("generation thread dead — completions 404"
                          " (Metal OOM zombie)")
            else:
                reason = (f"process exited (code "
                          f"{big.returncode if big is not None else '?'})")

            restarts.append(now)
            if len(restarts) >= loop_max and not loop_mode:
                loop_mode = True
                iv = backoff
                _note(f"{len(restarts)} restarts in {loop_window / 60:.0f} min —"
                      " the client session is likely too big for this machine:"
                      " compact it (/compact) or start a fresh one. Backing off"
                      f" to a probe every {backoff / 60:.0f} min; restarts"
                      " continue")
            restart(reason)
            bad = 0
            if loop_mode and len(restarts) < loop_max:
                loop_mode = False
                iv = interval
                _note("no crashes in a while — back to normal probing")

    threading.Thread(target=run, daemon=True, name="mlx-watchdog").start()


def stop_watchdog() -> None:
    """Signal the watchdog to stand down before stop_children() — no
    restarts during shutdown."""
    _watchdog_stop.set()


_proxy_log = None
_proxy_log_lock = threading.Lock()


def start_proxy(cfg, model: str, small_ok: bool) -> None:
    """The client proxy as a thread of this process (local_llm.proxy):
    resolves the local/small aliases to the model servers, relays OpenAI
    requests and bridges Anthropic /v1/messages. Started before the model
    servers, so the port answers while they load and clients see a 503
    instead of a refused connection."""
    global _proxy_log
    ui.info(f"Starting client proxy on :{cfg.proxy_port}")
    _proxy_log = _open_log(PROXY_LOG)

    def access(line: str) -> None:
        with _proxy_log_lock:
            _proxy_log.write((line + "\n").encode())
            _proxy_log.flush()

    try:
        server = proxy.ProxyServer(
            cfg.proxy_port,
            proxy.build_routes(model, cfg.mlx_port, cfg.small_model,
                               cfg.mlx_small_port if small_ok else 0),
            cfg.api_key, cfg.client_timeout, access)
    except OSError as exc:
        raise SystemExit(
            f"   cannot bind the client proxy to 127.0.0.1:{cfg.proxy_port}"
            f" ({exc.strerror or exc}) — port busy (PROXY_PORT) or another"
            " stack is still running") from None
    threading.Thread(target=server.serve_forever, name="client-proxy",
                     daemon=True).start()
    log.info("client proxy: up (openai+anthropic on 127.0.0.1:%d, helper=%s)",
             cfg.proxy_port, small_ok)


def ensure_gsq_runtime(cfg) -> None:
    """Install the optional packed-GSQ loader only when such a model is run."""
    python = sys.executable
    probe = subprocess.run([python, "-c", "import mlx_gsq"],
                           capture_output=True)
    if probe.returncode == 0:
        return
    # Alpha runtime, so pin the reviewed revision instead of tracking main.
    # Base dependencies only: mlx and mlx-lm are already managed by local-llm.
    package = ("mlx-gsq-rco @ git+https://github.com/uqer1244/"
               "mlx-GSQ-RCO.git@3b0ef3129a6bd47cbfc44d463e3c32e7dda0f9d3")
    ui.info("Installing GSQ-RCO runtime (pinned revision)")
    subprocess.run(["uv", "pip", "install", "--python", python, package],
                   check=True)


def _wait_ready(url: str, proc, timeout: int, logpath: str,
                what: str, step: int = 3) -> int:
    waited = 0
    while not http_ok(url):
        if proc.poll() is not None:
            log.error("%s died (code %s) — tail of %s:", what,
                      proc.returncode, logpath)
            ui.warn(f"{what} died:")
            print("\n".join(tail(logpath, 30)))
            raise SystemExit(1)
        if waited >= timeout:
            log.error("%s not ready in %ds — tail of %s:", what, timeout,
                      logpath)
            ui.warn(f"{what} did not become ready in {timeout}s")
            print("\n".join(tail(logpath, 30)))
            raise SystemExit(1)
        print(f"\r   loading… {waited}s", end="", flush=True)
        time.sleep(step)
        waited += step
    print(f"\r   ready in {waited}s        ")
    log.info("%s ready in %ds (%s)", what, waited, url)
    return waited


def _server_flags(python: str) -> set[str]:
    """Flags the installed mlx_lm.server actually accepts. The qwen4_exp PR
    fork dropped --kv-bits/--kv-group-size (its linear-attention cache can't
    be KV-quantized); passing them makes the server die on argparse."""
    try:
        out = subprocess.run([python, "-m", "mlx_lm.server", "--help"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return set()  # can't ask -> pass nothing optional
    # the usage block is ANSI-colored (flag names wrapped in escapes)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return {w for line in plain.splitlines() for w in line.split()
            if w.startswith("--")}


def start_mlx(cfg, model: str, max_input: int | None = None) -> None:
    ui.info(f"Starting mlx_lm.server on :{cfg.mlx_port}")
    _pkill(_MLX_PATTERN)
    time.sleep(1)
    # All model processes inherit the standalone app's isolated runtime.
    python = sys.executable
    ctx = hf.model_ctx(cfg.hf_hub, model)
    supported = _server_flags(python)
    flags = []
    if cfg.kv_bits > 0:
        if {"--kv-bits", "--kv-group-size"} <= supported:
            flags += ["--kv-bits", str(cfg.kv_bits),
                      "--kv-group-size", str(cfg.kv_group_size)]
        else:
            ui.warn("KV-cache quantization not supported by this "
                    "mlx_lm.server — running without it")
    cache_bytes = min(cfg.prompt_cache_bytes,
                      int(hf.memory_cache_limit(cfg.hf_hub, model) * 0.95))
    if cache_bytes > 0 and "--prompt-cache-bytes" in supported:
        flags += ["--prompt-cache-bytes", str(cache_bytes)]
    for flag, env_name, default in (
            ("--temp", "OLLMLX_TEMP", "0.0"),
            ("--top-p", "OLLMLX_TOP_P", "1.0"),
            ("--top-k", "OLLMLX_TOP_K", "0"),
            ("--min-p", "OLLMLX_MIN_P", "0.0")):
        if flag in supported:
            flags += [flag, os.environ.get(env_name, default)]
    max_tokens = int(os.environ.get("OLLMLX_MAX_TOKENS", str(min(ctx, 32768))))
    if "--prefill-step-size" in supported:
        # qwen3.5/3.8 MoE: a 2048-token chunk fans out to thousands of
        # expert buffers and hits the Metal resource limit (499000) —
        # mlx-lm#1644's proven workaround. Costs ~2.5x prefill speed.
        flags += ["--prefill-step-size", "512"]
    logfile = _open_log(MLX_LOG)
    env = dict(os.environ)
    # By the time we start mlx, the cache is complete (cli verified it against
    # the repo file list). Force offline: otherwise a partial cache would make
    # mlx start its own background download — plain DNS (no DoH pin), xet, no
    # token — which stalls for hours while /v1/models already answers 200.
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    if max_input:
        # mlxwrap refuses longer prompts before the KV eval — the OOM that
        # kills the generation thread never happens
        env["OLLMLX_MAX_INPUT_TOKENS"] = str(max_input)
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
    cmd = [python, "-m", "local_llm.mlxwrap",   # mlx_lm.server + tok/s log
           "--model", model, "--host", "127.0.0.1",
           "--port", str(cfg.mlx_port),
           "--max-tokens", str(max_tokens), *flags]
    log.info("spawn: %s", shlex.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=logfile,
                                stderr=subprocess.STDOUT, env=env)
    finally:
        logfile.close()
    children.append(proc)
    _wait_ready(f"http://127.0.0.1:{cfg.mlx_port}/v1/models", proc,
                cfg.load_timeout, MLX_LOG, "mlx_lm.server")
    return proc


def _reasoning_template(cfg, gguf_path):
    """Accept OpenAI's 'high' as 'xhigh' only for templates that already
    understand both reasoning_effort and xhigh. Model names are irrelevant."""
    from . import llamacpp
    template = llamacpp.gguf_meta(gguf_path).get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        return None
    if "reasoning_effort" not in template or "xhigh" not in template:
        return None
    patched = ('{%- if reasoning_effort == "high" -%}'
               '{%- set reasoning_effort = "xhigh" -%}'
               '{%- endif -%}\n' + template)
    digest = hashlib.sha256(patched.encode()).hexdigest()[:20]
    path = cfg.ollmlx_home / "templates" / f"reasoning-{digest}.jinja"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != patched:
        path.write_text(patched)
    return path


def start_llama(cfg, gguf_path: str, alias: str,
                max_input: int | None = None,
                lora_path: str | None = None):
    """llama-server (llama.cpp) in the mlx slot: GGUF weights on
    cfg.mlx_port with the OpenAI API, so the proxy, the clients and the
    monitor keep working unchanged. Readiness = /v1/models 200, same as mlx."""
    from . import llamacpp
    server = llamacpp.ensure_llama_server(cfg.llama_server)
    ui.info(f"Starting llama-server on :{cfg.mlx_port}")
    _pkill(_llama_pattern(cfg.mlx_port))
    time.sleep(1)
    flags = ["-m", gguf_path, "--host", "127.0.0.1",
             "--port", str(cfg.mlx_port),
             # the alias is what /v1/models reports — keep it the repo id
             # so it matches the model catalog
             "-a", alias, "-ngl", "999",
             # -c caps the KV allocation AND is the refusal ceiling: prompts
             # past it get a clean 400 instead of an OOM later (the same
             # job OLLMLX_MAX_INPUT_TOKENS does for mlxwrap)
             "-c", str(max_input or 0),
             "--temp", os.environ.get("OLLMLX_TEMP", "0.0"),
             "--top-p", os.environ.get("OLLMLX_TOP_P", "1.0"),
             "--top-k", os.environ.get("OLLMLX_TOP_K", "0"),
             "--min-p", os.environ.get("OLLMLX_MIN_P", "0.0"),
             "--n-predict", os.environ.get("OLLMLX_MAX_TOKENS", "32768")]
    template = _reasoning_template(cfg, gguf_path)
    if template:
        flags += ["--chat-template-file", str(template)]
    if lora_path:
        # The pi adapter was trained and evaluated with thinking disabled.
        flags += ["--lora", lora_path,
                  "--chat-template-kwargs", '{"enable_thinking":false}']
    if cfg.kv_bits == 8:
        flags += ["-ctk", "q8_0", "-ctv", "q8_0"]
    elif cfg.kv_bits:
        ui.warn(f"MLX_KV_BITS={cfg.kv_bits}: llama-server quantizes KV to"
                " q8_0 only — running without KV quantization")
    logfile = _open_log(LLAMA_LOG)
    cmd = [server, *flags]
    log.info("spawn: %s", shlex.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=logfile,
                                stderr=subprocess.STDOUT)
    finally:
        logfile.close()
    children.append(proc)
    _wait_ready(f"http://127.0.0.1:{cfg.mlx_port}/v1/models", proc,
                cfg.load_timeout, LLAMA_LOG, "llama-server")
    return proc


def start_mlx_small(cfg) -> None:
    """The tiny helper model on its own port: cheap classification/title
    calls (Claude Code's background requests have a short fixed timeout and
    would queue for minutes behind the big model's generation) plus a quick
    `ollmlx/small` for any client. Same flags as the big one are pointless
    here: no KV quant (nothing to save on ~0.4G weights), no prompt-cache
    cap (its caches are a few MB)."""
    ui.info(f"Starting helper model {cfg.small_model} on :{cfg.mlx_small_port}")
    # only stale instances of *this* model — start_mlx already swept the broad
    # pattern, and the main server must survive this pkill
    _pkill(f"local_llm\\.mlxwrap.*{re.escape(cfg.small_model)}")
    time.sleep(1)
    python = sys.executable
    # thinking off by default: hybrid-Qwen3 would burn ~200 hidden tokens
    # pondering a one-word answer (0.9s); a request can still re-enable it
    # via chat_template_kwargs — per-request args override the CLI ones
    flags = (["--chat-template-args", '{"enable_thinking":false}']
             if "--chat-template-args" in _server_flags(python) else [])
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    logfile = _open_log(MLX_SMALL_LOG)
    cmd = [python, "-m", "local_llm.mlxwrap",
           "--model", cfg.small_model, "--host", "127.0.0.1",
           "--port", str(cfg.mlx_small_port),
           "--max-tokens", "4096", *flags]
    log.info("spawn: %s", shlex.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=logfile,
                                stderr=subprocess.STDOUT, env=env)
    finally:
        logfile.close()
    children.append(proc)
    _wait_ready(f"http://127.0.0.1:{cfg.mlx_small_port}/v1/models", proc,
                120, MLX_SMALL_LOG, "helper model")
    return proc


def warmup(cfg) -> None:
    """End-to-end warmup: the client proxy (via the `local` alias, like the

    Fatal on failure: /v1/models answers 200 before the weights are loaded,
    so this is the only real readiness gate — a stack that fails here would
    only serve errors (cli.run_serve still stops the servers)."""
    ui.info("Warmup request")
    req = urllib.request.Request(
        f"http://127.0.0.1:{cfg.proxy_port}/v1/chat/completions",
        data=json.dumps({"model": "local", "max_tokens": 8,
                         "messages": [{"role": "user", "content": "Say OK"}]}
                        ).encode(),
        headers={"Authorization": f"Bearer {cfg.api_key}",
                 "content-type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(300, cfg.load_timeout)) as r:
            r.read()
        ui.ok("e2e OK")
    except (OSError, urllib.error.URLError) as exc:
        ui.warn(f"warmup failed ({exc})")
        ui.warn("if it repeats: memory pressure can kill the generation "
                "thread (Metal OOM -> 404s until restart) — free RAM/swap, "
                "`local-llm stop`, retry; the previous run's log is "
                f"{MLX_LOG}.1 / {LLAMA_LOG}.1")
        for path in (MLX_LOG, LLAMA_LOG, PROXY_LOG):
            lines = tail(path, 15)
            if lines:
                print(f"--- last lines of {path} ---")
                print("\n".join(lines))
        raise SystemExit(1) from None
