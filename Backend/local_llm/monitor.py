"""Telemetry readers for the desktop monitor (the MLX Stack app): port
status, the stack's processes, log tails, and the perf state parsed from
the servers' logs.

The tok/s numbers come from TOKPS lines that our mlx_lm.server wrapper
(local_llm.mlxwrap) prints into ~/.mlx-stack/logs/mlx-server.log and
~/.mlx-stack/logs/mlx-small.log ~once a second; GGUF stacks (llama-server)
have no wrapper — their per-request perf lines are parsed instead."""
import os
import re
import socket

import psutil


def top_eaters(k: int = 3, exclude_pids=()) -> list:
    """Top-k third-party processes by RSS as (rss, pid, name) — the kill
    candidates under memory pressure. Our own model processes are
    excluded: they are listed separately with their own RSS."""
    me = os.getpid()
    best = []
    for p in _safe_process_iter():
        try:
            if p.pid in (0, me) or p.pid in exclude_pids:
                continue
            rss = p.memory_info().rss
            if rss < 300 * 1024 * 1024:   # sub-300MB is not a lever
                continue
            best.append((rss, p.pid, os.path.basename(p.name() or "?")))
        except (psutil.Error, OSError, SystemError):
            continue
    best.sort(reverse=True)
    return best[:k]


def port_ok(port: int) -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def _safe_process_iter():
    """psutil.process_iter() with no attrs, hardened: macOS denies
    sysctl(KERN_PROCARGS2) for other users' processes, and current
    psutil/macOS builds leak that as SystemError/PermissionError instead
    of the AccessDenied that as_dict() would skip — so attrs-based
    iteration dies mid-generator (and emptied the process list with it).
    Attrs must be fetched by the caller under try/except; instance
    caching still lives in process_iter, keeping cpu_percent() deltas."""
    it = psutil.process_iter()
    while True:
        try:
            p = next(it)
        except StopIteration:
            return
        except (psutil.Error, OSError, SystemError):
            continue   # one unreadable pid must not end the scan
        yield p


def watch_procs(small_port: int = 0, mlx_port: int = 0):
    """[(label, Process)] for the model server(s); process_iter
    caches instances, so cpu_percent() between frames yields meaningful
    deltas."""
    out = []
    for p in _safe_process_iter():
        try:
            cmd = " ".join(p.cmdline() or ())
        except (psutil.Error, OSError, SystemError):
            continue
        if "mlx_lm.server" in cmd or "local_llm.mlxwrap" in cmd:
            # the helper is told apart by its --port (the big model's id
            # never contains it)
            label = ("mlx-small" if small_port and f"--port {small_port}" in cmd
                     else "mlx_lm.server")
            out.append((label, p))
        elif ("llama-server" in cmd
              and (not mlx_port or f"--port {mlx_port}" in cmd)):
            # ours only: the port tells it apart from an unrelated
            # llama-server the user runs for something else
            out.append(("llama-server", p))
    return out


def tail_errors(path: str, k: int = 2):
    """Last k error lines of a log, skipping benign disconnect noise
    (a client dropping mid-response logs a scary-but-harmless
    BrokenPipeError traceback in the servers). Only the tail of the file
    is read — cheap on a live log, no multi-megabyte scans."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    tokens = ("ERROR", "CRITICAL", "Traceback", "Exception")
    benign = ("BrokenPipeError", "ConnectionResetError", "ConnectionAbortedError")
    hits = []
    # logging writes each record between dash separators — and the final
    # exception line of a traceback can land in the block after one, so
    # check the block and its successor before calling it a real error
    blocks = data.split("-" * 40)
    for i, block in enumerate(blocks):
        if not any(t in block for t in tokens):
            continue
        window = block + (blocks[i + 1] if i + 1 < len(blocks) else "")
        if any(b in l for l in window.splitlines() for b in benign):
            continue  # client hung up mid-response — the request was fine
        hits.extend(l for l in block.splitlines()
                    if any(t in l for t in tokens))
    return hits[-k:]


def tail_lines(path: str, k: int = 2, window: int = 262144):
    """Last k log lines (not only errors): generation progress, requests.
    Reads up to `window` bytes of tail — enough history for scrolling."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - window))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [l for l in data.splitlines() if l.strip()]
    return lines[-k:]


_TOKPS_LIVE = re.compile(r"^TOKPS live #(\d+) ([\d.]+)")
_TOKPS_PRE = re.compile(r"^TOKPS prefill #(\d+) (\d+)")
_TOKPS_DONE = re.compile(r"^TOKPS done (.+)")
_TOKPS_CONTEXT = re.compile(r"^TOKPS context (\d+)")
_TOKPS_ACTIVE = re.compile(r"^TOKPS active (\d+)")
_DONE_CONTEXT = re.compile(
    r"^(\d+) tok.*? prompt (\d+) tok, (\d+) cached"
    r"(?: \(([\d.]+)s to first\))?")
_PREFILL_CONTEXT = re.compile(r"Prompt processing progress: \d+/(\d+)")

# llama-server has no TOKPS lines; its per-request perf summaries stand in
# (format verified on build b11899). The prompt line is checked first —
# "prompt eval time" contains "eval time" as a substring.
_LLAMA_PRE = re.compile(r"prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens"
                        r".*?([^\s(,]+) tokens per second")
_LLAMA_EVAL = re.compile(r"(?<!prompt )eval time =\s*[\d.]+ ms /\s*(\d+)"
                         r" (?:runs|tokens).*?([^\s(,]+) tokens per second")


def _llama_perf(data: str):
    """Perf state plus the latest request's prompt+output context."""
    n_eval = n_pre = 0
    rate = pre_rate = 0.0
    summary = ""
    prompt = context = None
    for l in data.splitlines():
        if m := _LLAMA_PRE.search(l):
            n_pre += 1
            pre_rate = _safe_float(m[2])
            prompt = context = int(m[1])
        elif m := _LLAMA_EVAL.search(l):
            n_eval += 1
            rate = _safe_float(m[2])
            summary = f"{m[1]} tokens at {rate:.1f} tok/s"
            if prompt is not None:
                context = prompt + int(m[1])
    return n_eval, rate, n_pre, pre_rate, summary, context, None


def _safe_float(s: str) -> float:
    try:
        return float(s)          # "inf" (0-time prompt eval) -> 0, not a crash
    except ValueError:
        return 0.0


def tail_tokps(path: str):
    """Perf state plus the latest request's prompt+output context.

    The seq numbers advance ~1/s while the model generates, so an unchanged
    seq means "not generating right now". Prefill progress exposes the prompt
    size before completion; the done line adds generated tokens. GGUF runs
    fall back to llama-server perf lines (one tick per request)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 16384))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return -1, 0.0, -1, 0.0, "", None, None
    live, pre = (-1, 0.0), (-1, 0.0)
    summary = ""
    context = active = None
    for l in data.splitlines():
        if m := _TOKPS_LIVE.match(l):
            live = (int(m[1]), float(m[2]))
        elif m := _TOKPS_PRE.match(l):
            pre = (int(m[1]), int(m[2]))
        elif m := _TOKPS_DONE.match(l):
            summary = m[1]
            if done := _DONE_CONTEXT.match(summary):
                context = int(done[1]) + int(done[2])
        elif m := _TOKPS_CONTEXT.match(l):
            context = int(m[1])
        elif m := _TOKPS_ACTIVE.match(l):
            active = int(m[1])
        elif m := _PREFILL_CONTEXT.search(l):
            context = int(m[1])
    if (live[0] < 0 and pre[0] < 0 and not summary
            and context is None and active is None):
        return _llama_perf(data)
    return live[0], live[1], pre[0], pre[1], summary, context, active
