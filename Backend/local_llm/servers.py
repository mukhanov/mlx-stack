"""Lifecycle of the model servers and the client proxy: start, readiness
wait, warmup, stop, and the watchdog that restarts a crashed model server.

The model servers (mlx_lm.server or llama-server on :19797, the helper on
:8081) are foreground children of the stack controller and the client proxy
(:4000) is a thread of it — no daemons: stopping the controller (SIGTERM,
SIGHUP or Ctrl-C) kills everything at once (see cli.run_serve)."""
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
from functools import cache

from . import hf, monitor, proxy, ui
from .applog import log, logs_dir
from .config import LOCAL_ALIAS
from .net import HOST

_LOGS = logs_dir()   # ~/.mlx-stack/logs survives a reboot; /tmp does not
MLX_LOG = str(_LOGS / "mlx-server.log")
MLX_SMALL_LOG = str(_LOGS / "mlx-small.log")
LLAMA_LOG = str(_LOGS / "llama-server.log")
PROXY_LOG = str(_LOGS / "proxy.log")
WATCHDOG_LOG = str(_LOGS / "mlx-watchdog.log")

# The watchdog probes the model server this often; LOOP_MAX restarts within
# LOOP_WINDOW mean the client session is too big for this machine.
PROBE_INTERVAL = 45.0
LOOP_WINDOW = 600.0
LOOP_MAX = 3
BACKOFF = 600.0

# Model servers this controller started, oldest first; stop_children() ends them.
children: list[subprocess.Popen] = []


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _open_log(path: str):
    """Keep one generation of history: the previous run's log moves to
    path.1 before the fresh one truncates it — otherwise every restart
    destroys the evidence of whatever killed the run before it."""
    try:
        os.replace(path, path + ".1")
    except OSError:
        pass  # no previous log (or unwritable dir) — not fatal
    return open(path, "wb")


# --- stopping ---------------------------------------------------------------------


def _pkill(pattern: str) -> bool:
    return subprocess.run(["pkill", "-f", pattern],
                          capture_output=True).returncode == 0


def stale_server_pattern(port: int) -> str:
    """pkill pattern of a server of this stack listening on `port`. The
    port is part of the pattern on purpose: a model server the user runs
    for something else, on another port, must survive."""
    return rf"(local_llm\.mlxwrap|mlx_lm\.server|llama-server).*--port {port}( |$)"


def stop_stale(cfg) -> bool:
    """Kill what a hard-killed controller left on our ports. True when
    there was anything."""
    found = False
    for port in (cfg.mlx_port, cfg.mlx_small_port):
        found = _pkill(stale_server_pattern(port)) or found
    if found:
        time.sleep(1)   # the port is released as the process goes down
    return found


def _stop(proc) -> None:
    """End one of our servers: SIGTERM, then SIGKILL if it ignores that."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def stop_children() -> None:
    for proc in children:
        _stop(proc)
    children.clear()


# --- watchdog ----------------------------------------------------------------------
#
# The model server dies in two ways: the process exits, or a Metal OOM kills
# the generation thread while the HTTP server lives on — the post-OOM zombie
# (seen 2026-10-05 23:58 with a 186k-token session): /v1/models keeps
# answering 200 and every completion 404s until restarted. The watchdog
# catches both and brings the model back without user intervention.

_watchdog_stop = threading.Event()


def _note(msg: str) -> None:
    """An event line in the watchdog's own log — the monitor tails it.
    NOT the model log: the server child holds that file at its own non-append
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
        f"http://{HOST}:{port}/v1/chat/completions",
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


def supervise(cfg, model: str, state: dict, max_input: int | None = None) -> None:
    """Start the watchdog thread. `state` = {"big": Popen, "small": Popen|None,
    "backend": "mlx"|"llama", ...}; on a dead process or 2 consecutive bad
    probes it restarts the model server(s) in place — the client proxy and
    the client configs keep pointing at the same ports, so clients just see a
    pause.

    LOOP_MAX restarts within LOOP_WINDOW = the client session is bigger than
    this machine can serve (every restart the client resends its full context
    and the model OOMs again) — probing slows to BACKOFF and a note in the
    watchdog log says to compact the session. Restarts never stop; when the
    window drains, the normal cadence resumes."""
    _watchdog_stop.clear()
    restarts: list[float] = []

    def restart(reason: str) -> None:
        llama = state["backend"] == "llama"
        _stop(state["big"])
        if not llama and state["small"] is not None:
            # both load onto the same Metal device: the helper goes down with
            # the big model and comes back with it
            _stop(state["small"])
        try:
            state["big"] = (start_llama(cfg, state["gguf_path"], model,
                                        max_input, state.get("lora_path"))
                            if llama
                            else start_mlx(cfg, model, max_input))
            if not llama and state["small"] is not None:
                state["small"] = start_mlx_small(cfg)
            _note(f"{'llama-server' if llama else 'mlx'} restarted"
                  f" ({reason}) — stack healthy again")
        except (SystemExit, OSError) as exc:
            _note(f"restart failed ({reason}): {exc}")

    def run():
        bad = 0
        interval = PROBE_INTERVAL
        loop_mode = False
        while not _watchdog_stop.wait(interval):
            now = time.monotonic()
            while restarts and now - restarts[0] > LOOP_WINDOW:
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
            if len(restarts) >= LOOP_MAX and not loop_mode:
                loop_mode = True
                interval = BACKOFF
                _note(f"{len(restarts)} restarts in {LOOP_WINDOW / 60:.0f} min —"
                      " the client session is likely too big for this machine:"
                      " compact it (/compact) or start a fresh one. Backing off"
                      f" to a probe every {BACKOFF / 60:.0f} min; restarts"
                      " continue")
            restart(reason)
            bad = 0
            if loop_mode and len(restarts) < LOOP_MAX:
                loop_mode = False
                interval = PROBE_INTERVAL
                _note("no crashes in a while — back to normal probing")

    threading.Thread(target=run, daemon=True, name="mlx-watchdog").start()


def stop_watchdog() -> None:
    """Signal the watchdog to stand down before stop_children() — no
    restarts during shutdown."""
    _watchdog_stop.set()


# --- starting ----------------------------------------------------------------------


def start_proxy(cfg, model: str, small_ok: bool) -> None:
    """The client proxy as a thread of this process (local_llm.proxy):
    resolves the local/small aliases to the model servers, relays OpenAI
    requests and bridges Anthropic /v1/messages. Started before the model
    servers, so the port answers while they load and clients see a 503
    instead of a refused connection."""
    ui.info(f"Starting client proxy on :{cfg.proxy_port}")
    access_log = _open_log(PROXY_LOG)
    lock = threading.Lock()

    def access(line: str) -> None:
        with lock:
            access_log.write((line + "\n").encode())
            access_log.flush()

    try:
        server = proxy.ProxyServer(
            cfg.proxy_port,
            proxy.build_routes(model, cfg.mlx_port, cfg.small_model,
                               cfg.mlx_small_port if small_ok else 0),
            cfg.api_key, cfg.client_timeout, access)
    except OSError as exc:
        raise SystemExit(
            f"   cannot bind the client proxy to {HOST}:{cfg.proxy_port}"
            f" ({exc.strerror or exc}) — port busy (PROXY_PORT) or another"
            " stack is still running") from None
    threading.Thread(target=server.serve_forever, name="client-proxy",
                     daemon=True).start()
    log.info("client proxy: up (openai+anthropic on %s:%d, helper=%s)",
             HOST, cfg.proxy_port, small_ok)


def ensure_gsq_runtime() -> None:
    """Install the optional packed-GSQ loader only when such a model is run."""
    probe = subprocess.run([sys.executable, "-c", "import mlx_gsq"],
                           capture_output=True)
    if probe.returncode == 0:
        return
    # Alpha runtime, so pin the reviewed revision instead of tracking main.
    # Base dependencies only: mlx and mlx-lm are already managed by local-llm.
    package = ("mlx-gsq-rco @ git+https://github.com/uqer1244/"
               "mlx-GSQ-RCO.git@3b0ef3129a6bd47cbfc44d463e3c32e7dda0f9d3")
    ui.info("Installing GSQ-RCO runtime (pinned revision)")
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, package],
                   check=True)


@cache
def _server_flags(python: str) -> frozenset[str]:
    """Flags the installed mlx_lm.server accepts. The qwen4_exp PR fork
    dropped --kv-bits/--kv-group-size (its linear-attention cache can't
    be KV-quantized); passing them makes the server die on argparse."""
    try:
        out = subprocess.run([python, "-m", "mlx_lm.server", "--help"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()  # can't ask -> pass nothing optional
    # the usage block is ANSI-colored (flag names wrapped in escapes)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return frozenset(w for line in plain.splitlines() for w in line.split()
                     if w.startswith("--"))


def _spawn(cmd: list[str], env, logpath: str) -> subprocess.Popen:
    logfile = _open_log(logpath)
    log.info("spawn: %s", shlex.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=logfile,
                                stderr=subprocess.STDOUT, env=env)
    finally:
        logfile.close()
    children[:] = [p for p in children if p.poll() is None]
    children.append(proc)
    return proc


def _start(cmd: list[str], env, logpath: str, url: str, timeout: int,
           what: str) -> subprocess.Popen:
    proc = _spawn(cmd, env, logpath)
    waited = 0
    while not http_ok(url):
        if proc.poll() is not None:
            problem = f"died (code {proc.returncode})"
        elif waited >= timeout:
            problem = f"did not become ready in {timeout}s"
        else:
            print(f"\r   loading… {waited}s", end="", flush=True)
            time.sleep(3)
            waited += 3
            continue
        log.error("%s %s — tail of %s:", what, problem, logpath)
        ui.warn(f"{what} {problem}:")
        print("\n".join(monitor.tail_lines(logpath, 30)))
        raise SystemExit(1)
    print(f"\r   ready in {waited}s        ")
    log.info("%s ready in %ds (%s)", what, waited, url)
    return proc


def _model_env(cfg) -> dict[str, str]:
    # By the time a model server starts, the cache is complete (cli verified
    # it against the repo file list). Force offline: otherwise a partial cache
    # would make mlx start its own background download — plain DNS (no DoH
    # pin), xet, no token — which stalls for hours while /v1/models already
    # answers 200.
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
    return env


def _mlx_url(port: int) -> str:
    return f"http://{HOST}:{port}/v1/models"


def start_mlx(cfg, model: str, max_input: int | None = None):
    ui.info(f"Starting mlx_lm.server on :{cfg.mlx_port}")
    supported = _server_flags(sys.executable)
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
    max_tokens = int(os.environ.get(
        "OLLMLX_MAX_TOKENS", str(min(hf.model_ctx(cfg.hf_hub, model), 32768))))
    if "--prefill-step-size" in supported:
        # qwen3.5/3.8 MoE: a 2048-token chunk fans out to thousands of
        # expert buffers and hits the Metal resource limit (499000) —
        # mlx-lm#1644's proven workaround. Costs ~2.5x prefill speed.
        flags += ["--prefill-step-size", "512"]
    env = _model_env(cfg)
    if max_input:
        # mlxwrap refuses longer prompts before the KV eval — the OOM that
        # kills the generation thread never happens
        env["OLLMLX_MAX_INPUT_TOKENS"] = str(max_input)
    # our wrapper around mlx_lm.server: same CLI, plus the tok/s log
    return _start([sys.executable, "-m", "local_llm.mlxwrap",
                   "--model", model, "--host", HOST,
                   "--port", str(cfg.mlx_port),
                   "--max-tokens", str(max_tokens), *flags],
                  env, MLX_LOG, _mlx_url(cfg.mlx_port), cfg.load_timeout,
                  "mlx_lm.server")


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
    path = cfg.home / "templates" / f"reasoning-{digest}.jinja"
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
    flags = ["-m", gguf_path, "--host", HOST,
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
    return _start([server, *flags], None, LLAMA_LOG, _mlx_url(cfg.mlx_port),
                  cfg.load_timeout, "llama-server")


def start_mlx_small(cfg):
    """The tiny helper model on its own port: cheap classification/title
    calls (Claude Code's background requests have a short fixed timeout and
    would queue for minutes behind the big model's generation) plus a quick
    `small` alias for any client. Same flags as the big one are pointless
    here: no KV quant (nothing to save on ~0.4G weights), no prompt-cache
    cap (its caches are a few MB)."""
    ui.info(f"Starting helper model {cfg.small_model} on :{cfg.mlx_small_port}")
    # thinking off by default: hybrid-Qwen3 would burn ~200 hidden tokens
    # pondering a one-word answer (0.9s); a request can still re-enable it
    # via chat_template_kwargs — per-request args override the CLI ones
    flags = (["--chat-template-args", '{"enable_thinking":false}']
             if "--chat-template-args" in _server_flags(sys.executable) else [])
    return _start([sys.executable, "-m", "local_llm.mlxwrap",
                   "--model", cfg.small_model, "--host", HOST,
                   "--port", str(cfg.mlx_small_port),
                   "--max-tokens", "4096", *flags],
                  _model_env(cfg), MLX_SMALL_LOG, _mlx_url(cfg.mlx_small_port),
                  120, "helper model")


def warmup(cfg) -> None:
    """End-to-end warmup: the client proxy (via the `local` alias, like
    every client) through to the model server.

    Fatal on failure: /v1/models answers 200 before the weights are loaded,
    so this is the only real readiness gate — a stack that fails here would
    only serve errors (cli.run_serve still stops the servers)."""
    ui.info("Warmup request")
    req = urllib.request.Request(
        f"http://{HOST}:{cfg.proxy_port}/v1/chat/completions",
        data=json.dumps({"model": LOCAL_ALIAS, "max_tokens": 8,
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
            lines = monitor.tail_lines(path, 15)
            if lines:
                print(f"--- last lines of {path} ---")
                print("\n".join(lines))
        raise SystemExit(1) from None
