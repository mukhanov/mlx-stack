"""mlx_lm.server with live tokens-per-second logging for the monitor.

Launched instead of `python -m mlx_lm.server` (same CLI, same behavior —
see servers.start_mlx). One hook: ResponseGenerator.generate is wrapped so
that every generated token bumps an aggregate counter, and once a second a
"TOKPS ..." line goes to stdout — servers.py redirects it to
~/.mlx-stack/logs/mlx-server.log, where monitor.py picks it up for its tok/s
graph.

Line formats (kept short and stable — the monitor parses them):
  TOKPS live #12 48.2          decode, ~1/s while tokens flow (aggregate
                               across concurrent requests; #seq lets the
                               monitor tell "still generating" from "stale")
  TOKPS prefill #3 4127        prompt-processing rate, uncached tokens
  TOKPS context 8750           latest request's prompt+generated tokens, ~1/s
  TOKPS active 2               currently generating requests, on start/finish
  TOKPS done 512 tok · decode 48.9 tok/s (10.5s) · prompt 8231 tok, 5179 cached
                               per-request summary when the stream ends
"""
import importlib
import importlib.util
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

import mlx_lm.server as mlxs

# Model types that shipped ahead of mlx_lm releases (e.g. qwen4_exp from the
# mlx-lm PR #1788 port, plus our loader fixes for the oMLX oQ layout). The
# file name is the model_type; registration only kicks in when the installed
# mlx_lm doesn't have the module itself — a later release wins automatically.
_VENDOR_DIR = Path(__file__).parent / "vendor"


def _vendor_models() -> None:
    for path in sorted(_VENDOR_DIR.glob("*.py")):
        name = f"mlx_lm.models.{path.stem}"
        try:
            importlib.import_module(name)
            continue  # the installed mlx_lm has it
        except ImportError:
            pass
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # importlib checks sys.modules first
        spec.loader.exec_module(mod)


def _gsq_model_dir(model_path) -> Path | None:
    """Local directory only when it contains the custom packed-GSQ manifest."""
    path = Path(str(model_path)).expanduser()
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(model_path),
                                          local_files_only=True))
        except Exception:
            return None
    manifest = path / "model.safetensors.manifest.json"
    try:
        if json.loads(manifest.read_text()).get("format") == "mlx-gsq-packed-v1":
            return path
    except (OSError, ValueError):
        pass
    return None


def _load_gsq_model(path: Path):
    """Load through a file object: PackedModelStore.resolve() drops the
    .safetensors suffix from HF's blob symlinks, and mx.load then rejects the
    extensionless blob. No copy of the multi-GB checkpoint is made."""
    import mlx.core as mx
    from mlx_gsq import load_model

    model_file = path / "model.safetensors"
    resolved = str(model_file.resolve())
    orig_mx_load = mx.load

    def mx_load(file, *args, **kwargs):
        if str(file) == resolved:
            with model_file.open("rb") as f:
                return orig_mx_load(f, *args, **kwargs)
        return orig_mx_load(file, *args, **kwargs)

    # ModelProvider loads on its single generation thread, so this scoped
    # workaround cannot race another model load.
    mx.load = mx_load
    try:
        return load_model(path)
    finally:
        mx.load = orig_mx_load


def _patch_gsq_loader() -> None:
    """Route packed GSQ checkpoints away from mlx_lm's standard sanitizer.

    Seen 2026-11-18: packed U8 payloads look like 1-D conv weights to
    qwen3_5.sanitize(), which dies in moveaxis. The model author's runtime
    attaches those payloads to custom Metal kernels instead."""
    orig_load = mlxs.load

    def load(model_path, adapter_path=None, tokenizer_config=None,
             return_config=False, **kwargs):
        path = _gsq_model_dir(model_path)
        if path is None:
            return orig_load(model_path, adapter_path=adapter_path,
                             tokenizer_config=tokenizer_config,
                             return_config=return_config, **kwargs)
        if adapter_path:
            raise ValueError("LoRA adapters are not supported by GSQ-RCO")
        from mlx_lm.tokenizer_utils import load as load_tokenizer
        print(f"GSQ-RCO loader: {path}", flush=True)
        model = _load_gsq_model(path)
        tokenizer = load_tokenizer(
            path, tokenizer_config_extra=tokenizer_config or {})
        if return_config:
            config = json.loads((path / "config.json").read_text())
            # This runtime is text-only even when the source config has vision.
            config.pop("vision_config", None)
            config["language_model_only"] = True
            return model, tokenizer, config
        return model, tokenizer

    mlxs.load = load


EMIT_EVERY = 1.0   # seconds between live lines
IDLE_RESET = 5.0   # a longer gap since the last token starts a fresh window

_lock = threading.Lock()
# aggregate decode state: tokens since the last emitted line and when the
# window opened (shared by all in-flight requests — the generation thread
# interleaves them, so a per-request rate would under-report)
_live = {"tokens": 0, "t0": None, "seq": 0}
_active = 0


def _emit(text: str) -> None:
    print(f"TOKPS {text}", flush=True)   # stdout -> the mlx-server.log servers.py opened


def _active_bump(delta: int) -> None:
    global _active
    with _lock:
        _active += delta
        _emit(f"active {_active}")


def _tok_bump(now: float) -> None:
    """One more token: open the live window, or emit + reset when it's full."""
    with _lock:
        st = _live
        if st["t0"] is None:
            st["t0"] = now
        st["tokens"] += 1
        dt = now - st["t0"]
        if dt >= IDLE_RESET:
            # long pause between requests: reporting the accumulated 2 tokens
            # over 30 s would read as ~0 — start over instead
            st["tokens"], st["t0"] = 0, now
        elif dt >= EMIT_EVERY:
            st["seq"] += 1
            _emit(f"live #{st['seq']} {st['tokens'] / dt:.1f}")
            st["tokens"], st["t0"] = 0, now


def _prefill_cb(orig_cb, enabled=True):
    """Wrap the handler's progress callback: emit prefill rate lines ~1/s."""
    state = {"last_p": 0, "t": time.monotonic(), "seq": 0}

    def cb(processed, total):
        now = time.monotonic()
        dt = now - state["t"]
        if enabled and dt >= EMIT_EVERY and processed > state["last_p"]:
            state["seq"] += 1
            _emit(f"prefill #{state['seq']} {(processed - state['last_p']) / dt:.0f}")
            state["last_p"], state["t"] = processed, now
        if orig_cb is not None:
            orig_cb(processed, total)

    return cb


def _counting(ctx, inner, internal=False):
    """Yield what `inner` yields, counting real requests and timing phases."""
    n = 0
    t0 = time.monotonic()
    t_first = None
    prompt = getattr(ctx, "prompt", ()) or ()
    context_t = t0
    if not internal:
        _active_bump(1)
    try:
        for resp in inner:
            if t_first is None:
                t_first = time.monotonic()
            n += 1
            now = time.monotonic()
            if not internal:
                _tok_bump(now)
            if prompt and now - context_t >= EMIT_EVERY:
                if not internal:
                    _emit(f"context {len(prompt) + n}")
                context_t = now
            yield resp
    finally:
        t_end = time.monotonic()
        cached = max(0, getattr(ctx, "prompt_cache_count", 0) or 0)
        parts = [f"done {n} tok"]
        if n and t_first:
            decode_t = max(1e-9, t_end - t_first)
            parts.append(f"decode {n / decode_t:.1f} tok/s ({decode_t:.1f}s)")
        if prompt:
            ttft = (t_first or t_end) - t0
            parts.append(f"prompt {len(prompt)} tok, {cached} cached"
                         f" ({ttft:.1f}s to first)")
        if not internal:
            _emit(" · ".join(parts))
            _active_bump(-1)


def _msg_text(content) -> str:
    """Message content as text: a string, or OpenAI-style parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return "".join(out)
    return str(content)


def _leading_system(messages):
    sys = []
    for m in messages:
        if m.get("role") == "system":
            sys.append(m)
        else:
            break
    return sys


def _trim_to_fit(messages, target_chars):
    """The newest messages whose text fits target_chars, always keeping the
    leading system messages and cutting only on a user boundary (never
    splitting a tool_call/tool_result chain). None when nothing can go."""
    total = sum(len(_msg_text(m.get("content"))) + 8 for m in messages)
    acc = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        acc += len(_msg_text(messages[i].get("content"))) + 8
        if acc > target_chars:
            cut = i + 1
            break
    while cut < len(messages) and messages[cut].get("role") != "user":
        cut += 1
    if cut >= len(messages):
        return None
    trimmed = _leading_system(messages) + messages[cut:]
    return trimmed if len(trimmed) < len(messages) else None


def _restore_tool_args(messages):
    """mlx_lm.server's process_message_content runs in place and re-parses
    tool_call arguments on every pass (str -> dict). Our verification pass
    re-tokenizes the same conversation, so it would feed already-parsed
    dicts back into json.loads and die with 'the JSON object must be str,
    bytes or bytearray'. Serialize them back before re-submitting."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            func = tc.get("function") or {}
            args = func.get("arguments")
            if isinstance(args, (dict, list)):
                func["arguments"] = json.dumps(args, ensure_ascii=False)


def _patch_arrays_cache() -> None:
    """Backport of mlx-lm#1632 (the state property of ArraysCache): include
    the metadata arrays (lengths, left_padding) in the evaluated cache
    state. Without it, advance() accumulates a lazy subtraction graph on
    every decode step and long sessions die with [metal::malloc] Resource
    limit (499000) exceeded (mlx-lm#1644 — our qwen3.8/3.5 models use
    ArraysCache). 0.32.0–0.32.2 ship unpatched — drop when a release has it."""
    import mlx.core as mx
    from mlx_lm.models import cache as cache_mod

    def state(self):
        left_padding = mx.array([]) if self.left_padding is None else self.left_padding
        lengths = mx.array([]) if self.lengths is None else self.lengths
        return self.cache, left_padding, lengths

    def set_state(self, v):
        self.cache, left_padding, lengths = v
        self.left_padding = left_padding if left_padding.size > 0 else None
        self.lengths = lengths if lengths.size > 0 else None

    cache_mod.ArraysCache.state = property(state, set_state)


def _internal_request(request, generation_args) -> bool:
    """The stack's own readiness probes are not user traffic."""
    messages = getattr(request, "messages", None) or []
    if not messages or not isinstance(messages[-1], dict):
        return False
    text = _msg_text(messages[-1].get("content")).strip()
    limit = getattr(generation_args, "max_tokens", None)
    return ((limit == 1 and text == "ping")
            or (limit == 8 and text == "Say OK"))


def _install_hook() -> None:
    orig_generate = mlxs.ResponseGenerator.generate
    # KV budget for this machine (tokens); 0 = unlimited. Set by servers.
    # start_mlx from safe_context(). Past it the prompt eval OOMs the Metal
    # command buffer and kills the generation thread — the whole server
    # 404s until restarted. Refusing the request keeps the server alive.
    max_input = int(os.environ.get("OLLMLX_MAX_INPUT_TOKENS", "0") or 0)

    def generate(self, request, generation_args, progress_callback=None):
        internal = _internal_request(request, generation_args)
        progress = _prefill_cb(progress_callback, enabled=not internal)
        ctx, inner = orig_generate(self, request, generation_args, progress)
        n = len(getattr(ctx, "prompt", ()) or ())
        if max_input and n > max_input:
            # the session outgrew the machine's KV budget: shrink it here —
            # clients like GUI apps can't /compact, and a 404 leaves them
            # stuck. Drop the oldest messages, keep the system prompt and a
            # fresh tail; the tail starts on a user message so tool chains
            # never split.
            messages = getattr(request, "messages", None)
            orig_len = len(messages) if isinstance(messages, list) else 0
            if isinstance(messages, list) and orig_len > 2:
                total_chars = sum(len(_msg_text(m.get("content"))) + 8
                                  for m in messages)
                target_chars = int(total_chars * max_input * 0.8 / n)
                for _ in range(3):
                    trimmed = _trim_to_fit(messages, target_chars)
                    if trimmed is None:
                        break
                    # private copies: the worker's pass-1 processing mutates
                    # the original dicts in place (tool arguments str->dict),
                    # and shared dicts would re-crash on the second pass
                    trimmed = copy.deepcopy(trimmed)
                    dropped = len(messages) - len(trimmed)
                    # honest to the model: without this note it would
                    # confidently answer about context it no longer has
                    sys_n = len(_leading_system(trimmed))
                    first = trimmed[sys_n]
                    content = first.get("content")
                    if isinstance(content, str):
                        first["content"] = (
                            f"[{dropped} oldest messages of this conversation"
                            " were dropped to fit the context budget; earlier"
                            " details are unavailable.]\n" + content)
                    _restore_tool_args(trimmed)
                    request.messages = trimmed
                    _emit(f"trim attempt: sending {len(trimmed)} of "
                          f"{orig_len} messages (dropped {dropped})")
                    ctx, inner = orig_generate(
                        self, request, generation_args, progress)
                    n = len(getattr(ctx, "prompt", ()) or ())
                    if n <= max_input:
                        _emit(f"context trimmed: dropped {dropped} oldest "
                              f"messages, {n} tokens fit the budget — "
                              "compact the client session")
                        return ctx, _counting(ctx, inner, internal=internal)
                    target_chars = int(target_chars * 0.7)
                    _emit(f"context trim pass: still {n} tokens over "
                          f"{max_input} — shrinking further")
            raise ValueError(
                f"context too long: {n} tokens > {max_input} (the KV budget"
                " of this machine) — compact the session and retry")
        return ctx, _counting(ctx, inner, internal=internal)

    mlxs.ResponseGenerator.generate = generate


def main() -> None:
    try:
        _vendor_models()
    except Exception as exc:  # vendoring must not kill the server either
        print(f"model vendoring failed: {exc}", file=sys.stderr, flush=True)
    try:
        _patch_gsq_loader()
    except Exception as exc:
        print(f"GSQ loader patch failed: {exc}", file=sys.stderr, flush=True)
    try:
        _patch_arrays_cache()
    except Exception as exc:  # the cache fix must not kill the server either
        print(f"arrays cache patch failed: {exc}", file=sys.stderr, flush=True)
    try:
        _install_hook()
    except Exception as exc:  # never let stats kill the server
        print(f"TOKPS hook not installed: {exc}", file=sys.stderr, flush=True)
    mlxs.main()


if __name__ == "__main__":
    main()
