"""Native desktop monitor: local snapshot API and SwiftUI app launcher.

Reads the stack's logs and processes through the telemetry readers in
monitor.py. The HTTP server binds localhost and lives as long as the app."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from . import applog, clients, hf, monitor, servers
from .applog import logs_dir
from .config import Config
from .model_catalog import ModelCatalog


def _client_state():
    """Current model and its advertised cap from the generated OMP catalog."""
    f = Path("~/.omp/agent/models.json").expanduser()
    try:
        models = json.loads(f.read_text())["providers"]["ollmlx"]["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return "local", 0
    model = next((m["id"] for m in models
                  if m.get("id") not in ("local", "small")), "local")
    local = next((m for m in models if m.get("id") == "local"), {})
    return model, int(local.get("contextWindow", 0))


def _backend():
    """Backend of the latest stack start, or mlx for an old/empty app.log."""
    path = logs_dir() / "app.log"
    for line in reversed(monitor.tail_lines(path, 200)):
        if "stack start:" in line and " backend=llama" in line:
            return "llama"
        if "stack start:" in line:
            return "mlx"
    return "mlx"


def _loaded_model(procs, server_up, port, helper=False):
    """Actual process model for the main server or the helper, not a catalog guess."""
    for name, p in procs:
        if helper:
            if name != "mlx-small":
                continue
        elif name not in ("mlx_lm.server", "llama-server"):
            continue
        try:
            cmd = p.cmdline()
            flag = "-a" if name == "llama-server" else "--model"
            if flag in cmd:
                return cmd[cmd.index(flag) + 1]
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            pass
    if server_up:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=0.5) as r:
                data = json.load(r).get("data", [])
            if data and isinstance(data[0].get("id"), str):
                return data[0]["id"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


class RequestTracker:
    """Monotonic per-log request counters; reads only bytes appended since the
    previous snapshot. MLX starts increment active; llama.cpp ends with eval."""
    def __init__(self):
        self.samples = {}
        self.lock = threading.Lock()

    def count(self, path):
        path = str(path)
        with self.lock:
            try:
                st = Path(path).stat()
            except OSError:
                return self.samples.get(path, {}).get("count", 0)
            state = self.samples.get(path)
            if state is None:
                self.samples[path] = {"inode": st.st_ino, "offset": st.st_size,
                                      "count": 0, "active": 0}
                return 0
            if state["inode"] != st.st_ino or st.st_size < state["offset"]:
                state.update(inode=st.st_ino, offset=0, count=0, active=0)
            if st.st_size == state["offset"]:
                return state["count"]
            try:
                with open(path, "rb") as f:
                    f.seek(state["offset"])
                    data = f.read(st.st_size - state["offset"])
            except OSError:
                return state["count"]
            state["offset"] = st.st_size
            for line in data.decode("utf-8", "replace").splitlines():
                if line.startswith("TOKPS active "):
                    try:
                        active = int(line.rsplit(" ", 1)[1])
                    except ValueError:
                        continue
                    if active > state["active"]:
                        state["count"] += active - state["active"]
                    state["active"] = active
                elif ("eval time =" in line
                      and "prompt eval time" not in line):
                    state["count"] += 1
            return state["count"]


def _perf(path, request_tracker=None):
    keys = ("live_seq", "live_rate", "pre_seq", "pre_rate", "summary",
            "context", "active", "requests")
    requests = request_tracker.count(path) if request_tracker else 0
    return dict(zip(keys, (*monitor.tail_tokps(path), requests)))


class StackController:
    """Models launched by the tray remain its foreground children.

    A stack started elsewhere is never taken over: its watchdog would restart
    the old model if the tray merely killed the children.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.deleted_path = cfg.ollmlx_home / "deleted-models.json"
        self.deleted = self._load_deleted()
        self.models = [rid for rid, _size, downloading
                       in hf.scan_installed(cfg.hf_hub) if not downloading]
        self.proc = None
        self.current = None
        self.request_tracker = RequestTracker()
        self.downloads = None
        self.lock = threading.Lock()

    def _load_deleted(self):
        try:
            blob = json.loads(self.deleted_path.read_text())
            return {model for model in blob.get("models", [])
                    if isinstance(model, str)}
        except (OSError, ValueError):
            return set()

    def _save_deleted(self):
        try:
            self.deleted_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.deleted_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"models": sorted(self.deleted)}))
            tmp.replace(self.deleted_path)
        except OSError:
            pass

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def can_switch(self):
        return self.running or not monitor.port_ok(self.cfg.mlx_port)

    def start(self, model, settings=None):
        with self.lock:
            if self.downloads and self.downloads.active(model):
                return 409, {"error": "wait for the download to finish"}
            if model not in self.models:
                return 400, {"error": "model is not fully downloaded"}
            if self.running and self.current == model:
                return 202, {"model": model}
            if monitor.port_ok(self.cfg.mlx_port) and not self.running:
                return 409, {"error": "stop the terminal stack before switching"}
            if not self._stop():
                return 409, {"error": "previous model is still stopping"}
            env = os.environ.copy()
            launch_settings = _safe_model_settings(
                self.cfg, model, settings or self.cfg.model_settings(model))
            env.update(self.cfg.model_environment(model, launch_settings))
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "local_llm", "serve", model], env=env)
            self.current = model
            return 202, {"model": model}

    def refresh_models(self, downloaded=None):
        with self.lock:
            if downloaded:
                self.deleted.discard(downloaded)
                self._save_deleted()
            self.models = [rid for rid, _size, downloading
                           in hf.scan_installed(self.cfg.hf_hub) if not downloading]

    def delete(self, model):
        with self.lock:
            installed = {rid for rid, _size, _downloading
                         in hf.scan_installed(self.cfg.hf_hub)}
            if model not in installed:
                return 404, {"error": "model is not downloaded"}
            procs = monitor.watch_procs(self.cfg.mlx_small_port, self.cfg.mlx_port)
            loaded = _loaded_model(procs, monitor.port_ok(self.cfg.mlx_port),
                                   self.cfg.mlx_port)
            if loaded == model:
                return 409, {"error": "stop the loaded model before deleting it"}
            try:
                hf.delete_model(self.cfg.hf_hub, model)
            except OSError as exc:
                return 500, {"error": str(exc)}
            self.models = [rid for rid, _size, downloading
                           in hf.scan_installed(self.cfg.hf_hub) if not downloading]
            self.deleted.add(model)
            self._save_deleted()
            return 200, {"deleted": model}

    def _stop(self):
        if not self.running:
            self.proc = None
            self.current = None
            return True
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return False
        self.proc = None
        self.current = None
        return True

    def stop_model(self):
        with self.lock:
            if self._stop():
                return 200, {"stopped": True}
            return 409, {"error": "model is still stopping"}

    def stop(self):
        with self.lock:
            self._stop()


class DownloadManager:
    """Background HF downloads; never takes StackController's model lock."""
    def __init__(self, cfg, on_change):
        self.cfg = cfg
        self.on_change = on_change
        self.jobs = {}
        self.lock = threading.Lock()
        self.state_path = cfg.ollmlx_home / "downloads.json"
        self.stopping = False
        self._resume()

    def _complete(self, model):
        """Local links alone are insufficient: early shards may exist while the
        rest of the repo has not been created yet. Compare against HF's list."""
        if not hf.is_cached(self.cfg.hf_hub, model):
            return False
        missing = hf.missing_files(self.cfg.hf_hub, model, self.cfg.hf_token)
        return missing == []

    def _spawn(self, model):
        env = os.environ.copy()
        token = self.cfg.hf_token
        if token:
            env["HF_TOKEN"] = token
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "local_llm.download_worker", model,
                 str(self.cfg.hf_hub)], env=env)
        except OSError as exc:
            self.jobs[model].update(active=False, error=str(exc))
            self._save()
            return
        self.jobs[model]["proc"] = proc
        self.jobs[model].setdefault("paused", False)
        threading.Thread(target=self._run, args=(model,), daemon=True).start()

    def _save(self):
        """Atomically persist every unfinished model; callers hold the lock
        (except single-threaded startup). Failed jobs resume on next launch."""
        try:
            self.cfg.ollmlx_home.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"models": sorted(self.jobs)}))
            tmp.replace(self.state_path)
        except OSError:
            pass

    def _resume(self):
        try:
            blob = json.loads(self.state_path.read_text())
            pending = blob.get("models", [])
        except (OSError, ValueError):
            pending = []
        for model in pending:
            if not isinstance(model, str) or self._complete(model):
                continue
            self.jobs[model] = {"active": True, "error": None,
                                "bytes": 0, "total": 0, "rate": 0.0}
        self._save()
        for model in self.jobs:
            self._spawn(model)

    def start(self, model):
        with self.lock:
            job = self.jobs.get(model)
            if job and job["active"]:
                return 202, {"downloading": model}
            if self._complete(model):
                self.jobs.pop(model, None)
                self._save()
                return 200, {"downloaded": model}
            self.jobs[model] = {"active": True, "error": None,
                                "bytes": 0, "total": 0, "rate": 0.0}
            self._save()
            self._spawn(model)
            return 202, {"downloading": model}

    def _run(self, model):
        completed = False
        try:
            try:
                total, _names, _files = hf._repo_file_meta(model, self.cfg.hf_token)
            except Exception:
                total = 0
            with self.lock:
                self.jobs[model]["total"] = total
            threading.Thread(target=self._watch, args=(model,), daemon=True).start()
            proc = self.jobs[model]["proc"]
            status = proc.wait()
            complete = self._complete(model) if status == 0 else False
            with self.lock:
                if model not in self.jobs:
                    return
                if status == 0 and complete:
                    completed = True
                    self.jobs.pop(model, None)
                elif status == 0:
                    self.jobs[model]["active"] = False
                    self.jobs[model]["error"] = "download finished with missing files; retry"
                elif not self.stopping and not self.jobs[model].get("cancelled"):
                    self.jobs[model]["active"] = False
                    self.jobs[model]["error"] = f"download exited with status {status}"
                self._save()
        except BaseException as exc:
            with self.lock:
                if model in self.jobs and not self.stopping:
                    self.jobs[model]["active"] = False
                    self.jobs[model]["error"] = str(exc)
                    self._save()
        finally:
            self.on_change(model if completed else None)

    def _watch(self, model):
        directory = hf.model_dir(self.cfg.hf_hub, model)
        last_done, _blob, _files = hf._scan_download(directory)
        last_t = time.monotonic()
        rate = 0.0
        while self.active(model):
            time.sleep(1)
            done, _blob, _files = hf._scan_download(directory)
            now = time.monotonic()
            if self.paused(model):
                last_done, last_t = done, now
                continue
            instant = max(0, done - last_done) / max(1e-9, now - last_t)
            rate = instant if rate == 0 else 0.3 * instant + 0.7 * rate
            with self.lock:
                if model in self.jobs:
                    self.jobs[model].update(bytes=done, rate=rate)
            last_done, last_t = done, now

    @staticmethod
    def _public(job):
        return {"active": job.get("active", False),
                "paused": job.get("paused", False),
                "error": job.get("error"),
                "bytes": job.get("bytes", 0),
                "total": job.get("total", 0),
                "rate": job.get("rate", 0.0)}

    def progress(self, model):
        with self.lock:
            return self._public(self.jobs.get(model, {}))

    def all_progress(self):
        with self.lock:
            return {model: self._public(job) for model, job in self.jobs.items()}

    def pause(self, model):
        with self.lock:
            job = self.jobs.get(model)
            if not job or not job["active"]:
                return 404, {"error": "model is not downloading"}
            proc = job.get("proc")
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGSTOP)
            job["paused"] = True
            job["rate"] = 0.0
            self._save()
            return 200, {"paused": model}

    def resume(self, model):
        with self.lock:
            job = self.jobs.get(model)
            if not job or not job["active"]:
                return 404, {"error": "model is not downloading"}
            proc = job.get("proc")
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGCONT)
            job["paused"] = False
            self._save()
            return 202, {"downloading": model}

    def cancel(self, model):
        with self.lock:
            job = self.jobs.get(model)
            if not job:
                return 404, {"error": "model is not downloading"}
            job["cancelled"] = True
            proc = job.get("proc")
            if proc and proc.poll() is None:
                if job.get("paused"):
                    proc.send_signal(signal.SIGCONT)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self.jobs.pop(model, None)
            self._save()
        self.on_change()
        return 200, {"cancelled": model}

    def stop(self):
        with self.lock:
            self.stopping = True
            procs = [job["proc"] for job in self.jobs.values()
                     if job.get("proc") and job["proc"].poll() is None]
            for proc in procs:
                proc.send_signal(signal.SIGCONT)
                proc.terminate()

    def paused(self, model):
        with self.lock:
            return bool(self.jobs.get(model, {}).get("paused"))

    def active_ids(self):
        with self.lock:
            return {model for model, job in self.jobs.items() if job["active"]}

    def active(self, model):
        with self.lock:
            return bool(self.jobs.get(model, {}).get("active"))

    def error(self, model):
        with self.lock:
            return self.jobs.get(model, {}).get("error")


def _reclaimable_model_bytes(cfg):
    main_up = monitor.port_ok(cfg.mlx_port)
    if not main_up:
        return 0
    model = _loaded_model(monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port),
                          True, cfg.mlx_port)
    return hf.model_disk_bytes(cfg.hf_hub, model) if model else 0


def _safe_model_settings(cfg, model, values, reclaimable_bytes=0):
    values = dict(values)
    values["temperature"] = min(1.0, values["temperature"])
    cache_limit = int(hf.memory_cache_limit(cfg.hf_hub, model, reclaimable_bytes) * 0.95)
    ram_limit = hf.memory_context_limit(cfg.hf_hub, model, reclaimable_bytes,
                                        values["prompt_cache_bytes"], values["kv_bits"])
    values["prompt_cache_bytes"] = min(values["prompt_cache_bytes"], cache_limit)
    if ram_limit is not None:
        values["max_ctx"] = min(values["max_ctx"], max(4096, ram_limit))
    values["max_tokens"] = min(
        values["max_tokens"],
        hf.safe_context(cfg.hf_hub, model, values["prompt_cache_bytes"], values["max_ctx"],
                        reclaimable_bytes, values["kv_bits"]))
    return values


def _context_settings(cfg, model=None):
    defaults = {name: getattr(cfg, name) for name in (
        "max_ctx", "prompt_cache_bytes", "kv_bits", "kv_group_size",
        "load_timeout", "client_timeout", "temperature", "top_p", "top_k",
        "min_p", "max_tokens")}
    values = cfg.model_settings(model) if model else dict(defaults)
    main_up = monitor.port_ok(cfg.mlx_port)
    loaded = _loaded_model(monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port),
                           main_up, cfg.mlx_port) if main_up else None
    subject = model or loaded
    weights = hf.model_disk_bytes(cfg.hf_hub, subject) if subject else 0
    reclaimable = 0 if model and model == loaded else _reclaimable_model_bytes(cfg)
    calculation_reclaimable = reclaimable + (weights if subject and subject == loaded else 0)
    if model:
        values = _safe_model_settings(cfg, model, values, calculation_reclaimable)
    kv = hf.model_kv_per_token(cfg.hf_hub, subject) if subject else None
    return {
        "settings": values,
        "defaults": defaults,
        "kv_bytes_per_token": kv,
        "effective_ctx": (hf.safe_context(
            cfg.hf_hub, subject, values["prompt_cache_bytes"], values["max_ctx"],
            calculation_reclaimable, values["kv_bits"])
            if subject else None),
        "model_ctx": hf.model_ctx(cfg.hf_hub, subject) if subject else None,
        "available_bytes": psutil.virtual_memory().available,
        "reclaimable_bytes": reclaimable,
        "is_loaded": bool(model and model == loaded),
        "budget_available_bytes": psutil.virtual_memory().available + reclaimable,
        "weights_bytes": weights,
        "cache_limit_bytes": hf.memory_cache_limit(cfg.hf_hub, subject, calculation_reclaimable) if subject else 0,
    }


def _update_context_settings(cfg, payload):
    model = payload.get("model")
    if model:
        try:
            values = {name: int(payload[name]) for name in (
                "max_ctx", "prompt_cache_bytes", "kv_bits", "kv_group_size",
                "load_timeout", "client_timeout", "top_k", "max_tokens")}
            for name in ("temperature", "top_p", "min_p"):
                values[name] = float(payload[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError("expected all model launch settings") from None
        if not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be between 0 and 1")
        if not 0 <= values["top_p"] <= 1 or not 0 <= values["min_p"] <= 1:
            raise ValueError("top-p and min-p must be between 0 and 1")
        if not 0 <= values["top_k"] <= 1024:
            raise ValueError("top-k must be between 0 and 1024")
        if not 256 <= values["max_tokens"] <= 262_144:
            raise ValueError("max tokens must be between 256 and 262144")
        if values["kv_bits"] not in (0, 4, 8):
            raise ValueError("KV bits must be 0, 4 or 8")
        if not 16 <= values["kv_group_size"] <= 1024:
            raise ValueError("KV group size must be between 16 and 1024")
        if values["load_timeout"] <= 0 or values["client_timeout"] <= 0:
            raise ValueError("timeouts must be positive")
        max_ctx = values["max_ctx"]
        prompt_cache_bytes = values["prompt_cache_bytes"]
    else:
        try:
            max_ctx = int(payload["max_ctx"])
            prompt_cache_bytes = int(payload["prompt_cache_bytes"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("expected max_ctx and prompt_cache_bytes") from None
        values = None
    if not 4096 <= max_ctx <= 262_144:
        raise ValueError("context must be between 4K and 262K tokens")
    if not 0 <= prompt_cache_bytes <= 512 * 1024**3:
        raise ValueError("prompt cache must be between 0 and 512 GB")
    if values is not None:
        main_up = monitor.port_ok(cfg.mlx_port)
        loaded = _loaded_model(monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port),
                               main_up, cfg.mlx_port) if main_up else None
        reclaimable = 0 if model and model == loaded else _reclaimable_model_bytes(cfg)
        weights = hf.model_disk_bytes(cfg.hf_hub, model)
        calculation_reclaimable = reclaimable + (weights if model == loaded else 0)
        cache_limit = int(hf.memory_cache_limit(cfg.hf_hub, model, calculation_reclaimable) * 0.95)
        values["prompt_cache_bytes"] = min(values["prompt_cache_bytes"], cache_limit)
        ram_limit = hf.memory_context_limit(cfg.hf_hub, model, calculation_reclaimable,
                                            values["prompt_cache_bytes"], values["kv_bits"])
        if ram_limit is not None:
            if ram_limit < 4096:
                raise ValueError("not enough available RAM for this model")
            values["max_ctx"] = min(values["max_ctx"], ram_limit)
            values["max_tokens"] = min(
                values["max_tokens"],
                hf.safe_context(cfg.hf_hub, model, values["prompt_cache_bytes"], values["max_ctx"],
                                calculation_reclaimable, values["kv_bits"]))
    if values is None:
        cfg.max_ctx = max_ctx
        cfg.prompt_cache_bytes = prompt_cache_bytes
        cfg.save_context()
    else:
        cfg.save_model_settings(model, values)


def snapshot(cfg, controller=None) -> dict:
    """One UI-independent monitor sample; all values are JSON primitives."""
    _, max_input = _client_state()
    backend = _backend()
    main_log = servers.LLAMA_LOG if backend == "llama" else servers.MLX_LOG
    main_up = monitor.port_ok(cfg.mlx_port)
    small_up = bool(cfg.mlx_small_port and monitor.port_ok(cfg.mlx_small_port))
    percpu = psutil.cpu_percent(percpu=True)
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    procs = monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port)
    rows = []
    ours = 0
    uptime = 0
    for name, p in procs:
        try:
            rss = p.memory_info().rss
            ours += rss
            if name in ("mlx_lm.server", "llama") and main_up:
                uptime = max(uptime, int(time.time() - p.create_time()))
            rows.append({"name": name, "pid": p.pid,
                         "cpu": p.cpu_percent(), "rss": rss,
                         "rss_percent": rss / vm.total * 100,
                         "threads": p.num_threads()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    model = _loaded_model(procs, main_up, cfg.mlx_port) or "no model loaded"
    small_model = (_loaded_model(procs, small_up, cfg.mlx_small_port, helper=True)
                   if small_up else None)
    request_tracker = controller.request_tracker if controller else None
    launch_error = None
    if controller and controller.proc is not None and not main_up:
        status = controller.proc.poll()
        if status is not None and controller.current:
            launch_error = {"model": controller.current, "status": status}
    eaters = monitor.top_eaters(3, {p.pid for _, p in procs})
    logs = [(tag, line) for tag, path in (
                (backend, main_log), ("proxy", servers.PROXY_LOG),
                ("mlx-small", servers.MLX_SMALL_LOG),
                ("watchdog", servers.WATCHDOG_LOG))
            for line in monitor.tail_errors(path)][-3:]
    log_rows = [{"tag": tag, "text": text, "error": True}
                for tag, text in logs]
    log_rows += [{"tag": tag, "text": text, "error": False}
                 for tag, path in ((backend, main_log),
                                   ("proxy", servers.PROXY_LOG))
                 for text in monitor.tail_lines(path, 100)]
    client_settings = clients.client_configs(cfg)
    available_models = controller.models if controller else []
    if controller and controller.downloads:
        available_models = [model for model in available_models
                            if not controller.downloads.active(model)]
    return {
        "model": model,
        "small_model": small_model,
        "launch_error": launch_error,
        "backend": backend,
        "uptime_seconds": uptime,
        "max_input": max_input,
        "config_home": str(cfg.ollmlx_home),
        "available_models": controller.models if controller else [],
        "can_switch": controller.can_switch if controller else False,
        "ports": {
            "main": {"label": backend, "port": cfg.mlx_port, "up": main_up},
            "small": ({"label": "small", "port": cfg.mlx_small_port,
                       "up": small_up} if cfg.mlx_small_port else None),
            "proxy": {"label": "proxy", "port": cfg.proxy_port,
                      "up": monitor.port_ok(cfg.proxy_port)},
        },
        "cpu": {"total": sum(percpu) / len(percpu) if percpu else 0,
                "cores": percpu},
        "memory": {"used": vm.used, "total": vm.total,
                   "available": vm.available, "percent": vm.percent},
        "swap": {"used": sw.used, "total": sw.total, "percent": sw.percent},
        "processes": rows,
        "process_summary": {
            "ours": ours, "others": max(0, vm.used - ours),
            "ours_percent": ours / vm.total * 100,
        },
        "top_eaters": [{"name": name, "pid": pid, "rss": rss}
                       for rss, pid, name in eaters],
        "main": _perf(main_log, request_tracker),
        "small": _perf(servers.MLX_SMALL_LOG, request_tracker) if small_up else None,
        "clients": client_settings,
        "logs": log_rows,
    }


def _make_server(cfg, controller=None):
    """Bind the snapshot API before launching the native window."""
    controller = controller or StackController(cfg)
    downloads = DownloadManager(cfg, controller.refresh_models)
    controller.downloads = downloads
    catalog = ModelCatalog(cfg, downloads, controller.deleted)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            if parsed.path == "/snapshot":
                self.reply(200, snapshot(cfg, controller))
                return
            if parsed.path == "/downloads":
                self.reply(200, downloads.all_progress())
                return
            if parsed.path == "/settings":
                model = urllib.parse.parse_qs(parsed.query).get("model", [None])[0]
                self.reply(200, _context_settings(cfg, model))
                return
            if parsed.path != "/models":
                self.send_error(404)
                return
            params = urllib.parse.parse_qs(parsed.query)

            def arg(name, default):
                return params.get(name, [default])[0]

            try:
                result = catalog.list(
                    scope=arg("scope", "community"), fmt=arg("format", "mlx"),
                    query=arg("query", ""), sort=arg("sort", "fit"),
                    offset=int(arg("offset", "0")), limit=int(arg("limit", "50")),
                    refresh=arg("refresh", "0") == "1",
                    local_only=arg("local_only", "0") == "1")
            except ValueError as exc:
                self.reply(400, {"error": str(exc)})
                return
            except Exception as exc:
                applog.log.exception("model catalog failed")
                self.reply(500, {"error": str(exc)})
                return
            self.reply(200, result)

        def do_POST(self):
            if self.path not in ("/run", "/delete", "/download", "/pause", "/resume", "/cancel", "/settings", "/stop"):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 4096:
                    raise ValueError
                payload = json.loads(self.rfile.read(size))
            except (ValueError, json.JSONDecodeError):
                self.reply(400, {"error": "invalid JSON"})
                return
            if self.path == "/settings":
                try:
                    _update_context_settings(cfg, payload)
                except ValueError as exc:
                    self.reply(400, {"error": str(exc)})
                    return
                self.reply(200, _context_settings(cfg, payload.get("model")))
                return
            model = payload.get("model")
            if not isinstance(model, str):
                self.reply(400, {"error": "expected a model name"})
                return
            if self.path == "/run":
                self.reply(*controller.start(model))
            elif self.path == "/delete":
                self.reply(*controller.delete(model))
            elif self.path == "/stop":
                self.reply(*controller.stop_model())
            elif self.path == "/pause":
                self.reply(*downloads.pause(model))
            elif self.path == "/resume":
                self.reply(*downloads.resume(model))
            elif self.path == "/cancel":
                self.reply(*downloads.cancel(model))
            else:
                self.reply(*downloads.start(model))

        def log_message(self, _format, *args):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", cfg.widget_port), Handler)
        server.downloads = downloads
        return server
    except OSError:
        raise SystemExit(f"   widget port :{cfg.widget_port} is in use"
                         " — set WIDGET_PORT to a free port") from None


def run_api(cfg) -> int:
    """Foreground API child of MLX Stack.app; SIGTERM also stops its model."""
    applog.setup()
    controller = StackController(cfg)
    server = _make_server(cfg, controller)
    parent = os.getppid()

    def watch_parent():
        # A killed/crashed app cannot run applicationWillTerminate. Once
        # launchd adopts us, keeping the API alive would leave a headless
        # model and a mystery Python process with no tray icon.
        while os.getppid() == parent:
            time.sleep(1)
        server.shutdown()

    threading.Thread(target=watch_parent, name="app-parent-watchdog",
                     daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
        return 0
    finally:
        server.server_close()
        server.downloads.stop()
        controller.stop()


def run(cfg) -> int:
    """Open the installed app; its bundle owns the backend and API child."""
    try:
        subprocess.run(["open", "-a", "MLX Stack", "--args", "--port",
                        str(cfg.widget_port)], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("   failed to open MLX Stack.app") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(run_api(Config.from_env()))
