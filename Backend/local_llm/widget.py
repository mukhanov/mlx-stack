"""The app's local API: the monitor snapshot, the model catalog, downloads,
launch settings and load/stop control.

Reads the stack's logs and processes through monitor.py and loads models as
`local-llm serve` children. The HTTP server binds loopback and lives as long
as the app."""
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
from .config import LOCAL_ALIAS, SETTING_ENV, Config, launch_settings_from, write_json
from .model_catalog import ModelCatalog
from .net import HOST, from_web_page, loopback_host


def _advertised_context(cfg) -> int:
    """Context the stack advertises to its clients for the model it serves
    — the KV-safe budget, not the config maximum. 0 until a catalog exists."""
    try:
        models = json.loads((cfg.home / "configs" / "omp-models.json")
                            .read_text())["providers"]["ollmlx"]["models"]
        local = next(m for m in models if m.get("id") == LOCAL_ALIAS)
        return int(local.get("contextWindow", 0))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return 0


def _backend() -> str:
    """Backend of the latest stack start: llama for GGUF weights, else mlx."""
    for line in reversed(monitor.tail_lines(applog.logs_dir() / "app.log", 200)):
        if "stack start:" in line:
            return "llama" if " backend=llama" in line else "mlx"
    return "mlx"


def _loaded_model(procs, server_up: bool, port: int,
                  helper=False) -> str | None:
    """The model a server serves: read from its command line, else from
    what it reports on its port while that port is up."""
    wanted = ("mlx-small",) if helper else ("mlx_lm.server", "llama-server")
    for name, proc in procs:
        if name not in wanted:
            continue
        try:
            cmd = proc.cmdline()
            flag = "-a" if name == "llama-server" else "--model"
            if flag in cmd:
                return cmd[cmd.index(flag) + 1]
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            pass
    if not server_up:
        return None
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/v1/models",
                                    timeout=0.5) as r:
            data = json.load(r).get("data", [])
        if data and isinstance(data[0].get("id"), str):
            return data[0]["id"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _main_model(cfg) -> str | None:
    """The model the main server serves while its port is up."""
    if not monitor.port_ok(cfg.mlx_port):
        return None
    procs = monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port)
    return _loaded_model(procs, True, cfg.mlx_port)


# The stages a controller goes through while it brings a stack up
# (cli._boot); the window shows each one as the model loads.
LAUNCH_STAGES = ("prepare", "weights", "helper", "warmup")
_RSS_BASE = 512 * 1024**2   # a model server's footprint before any weights


def _launch_record(cfg) -> dict:
    """launch.json — what the controller that loads a model last wrote
    (cli.LaunchProgress); {} when there is none."""
    try:
        record = json.loads((cfg.home / "launch.json").read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _drop_launch_record(cfg) -> None:
    try:
        (cfg.home / "launch.json").unlink(missing_ok=True)
    except OSError:
        pass


def _launch_in_progress(cfg, main_rss: int) -> dict | None:
    """The load under way and how far it has got, or None. The weights
    count as loaded in step with the model server's resident memory: once
    they are all in, its RSS is about their size on disk."""
    record = _launch_record(cfg)
    stage, started = record.get("stage"), record.get("started")
    if stage not in LAUNCH_STAGES or not isinstance(started, (int, float)):
        return None
    age = time.time() - started
    if not 0 <= age < 3600:
        return None
    try:   # a hard-killed controller leaves its record behind
        cmd = " ".join(psutil.Process(record["pid"]).cmdline())
    except (psutil.Error, KeyError, TypeError):
        return None
    if "local_llm" not in cmd or " serve" not in cmd:
        return None
    launch = {"model": record.get("model"), "stage": stage, "started": started,
              "elapsed": int(age), "fraction": None}
    weights = hf.model_disk_bytes(cfg.hf_hub, launch["model"])
    if stage == "weights" and weights:
        launch["fraction"] = min(1.0, max(0.0, (main_rss - _RSS_BASE) / weights))
    return launch


def _launch_failure(cfg, main_up: bool) -> dict | None:
    """Why the last launch stopped, for the window. Nothing once a stack
    serves: that has put the failure to rest."""
    record = _launch_record(cfg)
    if record.get("stage") != "failed":
        return None
    if main_up:
        _drop_launch_record(cfg)
        return None
    return {"model": record.get("model"), "status": record.get("status", 1),
            "reason": record.get("error")}


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
    """Models launched by the app remain its foreground children.

    A stack started elsewhere is never taken over: its watchdog would restart
    the old model if the app merely killed the servers."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.deleted_path = cfg.home / "deleted-models.json"
        self.deleted = self._load_deleted()
        self.models = [rid for rid, _size, downloading
                       in hf.scan_installed(cfg.hf_hub) if not downloading]
        self.proc = None
        self.current = None
        self.request_tracker = RequestTracker()
        self.downloads = None
        self.lock = threading.Lock()
        # a failure from before this app session ran is history
        if _launch_record(cfg).get("stage") == "failed":
            _drop_launch_record(cfg)

    def _load_deleted(self):
        try:
            blob = json.loads(self.deleted_path.read_text())
            return {model for model in blob.get("models", [])
                    if isinstance(model, str)}
        except (OSError, ValueError, AttributeError):
            return set()

    def _save_deleted(self):
        try:
            write_json(self.deleted_path, {"models": sorted(self.deleted)})
        except OSError:
            pass

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def can_switch(self):
        return self.running or not monitor.port_ok(self.cfg.mlx_port)

    def start(self, model, config_write_consent=False):
        with self.lock:
            if self.downloads and self.downloads.active(model):
                return 409, {"error": "wait for the download to finish"}
            if model not in self.models:
                return 400, {"error": "model is not fully downloaded"}
            if self.running and self.current == model:
                return 202, {"model": model}
            if not config_write_consent:
                return 409, {"error": "review and approve agent configuration"
                                      " changes before loading a model"}
            if monitor.port_ok(self.cfg.mlx_port) and not self.running:
                return 409, {"error": "stop the terminal stack before switching"}
            if not self._stop():
                return 409, {"error": "previous model is still stopping"}
            settings = _safe_model_settings(self.cfg, model,
                                            self.cfg.model_settings(model))
            env = {**os.environ, **self.cfg.model_environment(model, settings)}
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "local_llm", "serve", model,
                 "--approve-configs"], env=env)
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
            if _main_model(self.cfg) == model:
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
        self.state_path = cfg.home / "downloads.json"
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
        env = dict(os.environ)
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
        """Persist every unfinished model; callers hold the lock (except
        single-threaded startup). Failed jobs resume on next launch."""
        try:
            write_json(self.state_path, {"models": sorted(self.jobs)})
        except OSError:
            pass

    def _resume(self):
        try:
            pending = json.loads(self.state_path.read_text()).get("models", [])
        except (OSError, ValueError, AttributeError):
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
            total = hf.repo_size(model, self.cfg.hf_token)
            with self.lock:
                if model in self.jobs:
                    self.jobs[model]["total"] = total
            threading.Thread(target=self._watch, args=(model,), daemon=True).start()
            status = self.jobs[model]["proc"].wait()
            complete = self._complete(model) if status == 0 else False
            with self.lock:
                if model not in self.jobs:
                    return
                if status == 0 and complete:
                    completed = True
                    self.jobs.pop(model, None)
                elif status == 0:
                    self.jobs[model].update(active=False, error="download finished"
                                            " with missing files; retry")
                elif not self.stopping and not self.jobs[model].get("cancelled"):
                    self.jobs[model].update(active=False, error=f"download exited"
                                            f" with status {status}")
                self._save()
        except BaseException as exc:
            with self.lock:
                if model in self.jobs and not self.stopping:
                    self.jobs[model].update(active=False, error=str(exc))
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
            job.update(paused=True, rate=0.0)
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


# --- launch settings ---------------------------------------------------------------


def _reclaimable_model_bytes(cfg) -> int:
    """Disk size of the main model's weights while its port is up — what
    unloading it frees."""
    model = _main_model(cfg)
    return hf.model_disk_bytes(cfg.hf_hub, model) if model else 0


def _unload_headroom(cfg, model, loaded) -> tuple[int, int]:
    """(reclaimable, freed): the RAM that loading `model` sets free. A
    different model evicts the loaded one, whose weights are up for grabs;
    the loaded model itself only lends its own weights to its reload."""
    if model == loaded:
        return 0, hf.model_disk_bytes(cfg.hf_hub, model)
    reclaimable = _reclaimable_model_bytes(cfg)
    return reclaimable, reclaimable


def _safe_model_settings(cfg, model, values, reclaimable_bytes=0,
                         strict=False) -> dict:
    """Launch settings clamped to what this machine can serve the model
    with: the prompt cache inside free RAM after the weights and a reserve,
    the context inside the KV budget. strict=True refuses a model that
    cannot get a 4K context instead of clamping."""
    values = dict(values)
    values["temperature"] = min(1.0, values["temperature"])
    cache_limit = int(hf.memory_cache_limit(cfg.hf_hub, model, reclaimable_bytes) * 0.95)
    values["prompt_cache_bytes"] = min(values["prompt_cache_bytes"], cache_limit)
    ram_limit = hf.memory_context_limit(cfg.hf_hub, model, reclaimable_bytes,
                                        values["prompt_cache_bytes"], values["kv_bits"])
    if ram_limit is not None:
        if strict and ram_limit < 4096:
            raise ValueError("not enough available RAM for this model")
        values["max_ctx"] = min(values["max_ctx"], max(4096, ram_limit))
    values["max_tokens"] = min(
        values["max_tokens"],
        hf.safe_context(cfg.hf_hub, model, values["prompt_cache_bytes"],
                        values["max_ctx"], reclaimable_bytes, values["kv_bits"]))
    return values


def _context_settings(cfg, model) -> dict:
    """Launch settings of `model` for the app's form, with the memory
    figures its forecast is drawn from."""
    loaded = _main_model(cfg)
    reclaimable, freed = _unload_headroom(cfg, model, loaded)
    values = _safe_model_settings(cfg, model, cfg.model_settings(model), freed)
    available = psutil.virtual_memory().available
    return {
        "settings": values,
        "defaults": {name: getattr(cfg, name) for name in SETTING_ENV},
        "kv_bytes_per_token": hf.model_kv_per_token(cfg.hf_hub, model),
        "effective_ctx": hf.safe_context(
            cfg.hf_hub, model, values["prompt_cache_bytes"], values["max_ctx"],
            freed, values["kv_bits"]),
        "model_ctx": hf.model_ctx(cfg.hf_hub, model),
        "available_bytes": available,
        "reclaimable_bytes": reclaimable,
        "is_loaded": model == loaded,
        "budget_available_bytes": available + reclaimable,
        "weights_bytes": hf.model_disk_bytes(cfg.hf_hub, model),
        "cache_limit_bytes": hf.memory_cache_limit(cfg.hf_hub, model, freed),
        "config_preview": clients.config_preview(cfg),
    }


def _update_context_settings(cfg, payload) -> None:
    """Save the launch settings the app's form sent for a model."""
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("expected a model name")
    _reclaimable, freed = _unload_headroom(cfg, model, _main_model(cfg))
    values = _safe_model_settings(cfg, model, launch_settings_from(payload),
                                  freed, strict=True)
    cfg.save_model_settings(model, values)


# --- monitor snapshot --------------------------------------------------------------


def snapshot(cfg, controller) -> dict:
    """One monitor sample; all values are JSON primitives."""
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
            if name != "mlx-small" and main_up:
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
    launch = _launch_in_progress(
        cfg, sum(r["rss"] for r in rows if r["name"] != "mlx-small"))
    launch_error = _launch_failure(cfg, main_up)
    if (launch_error is None and launch is None and not main_up
            and controller.proc is not None and controller.current):
        # our own controller ended without saying why (a kill, a crash
        # before it could record it)
        status = controller.proc.poll()
        if status is not None:
            launch_error = {"model": controller.current, "status": status,
                            "reason": "the model process ended without reporting"}
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
    return {
        "model": model,
        "small_model": small_model,
        "launch_error": launch_error,
        "launch": launch,
        "backend": backend,
        "uptime_seconds": uptime,
        "max_input": _advertised_context(cfg),
        "config_home": str(cfg.home),
        "available_models": controller.models,
        "can_switch": controller.can_switch,
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
                       for rss, pid, name in
                       monitor.top_eaters(3, {p.pid for _, p in procs})],
        "main": _perf(main_log, controller.request_tracker),
        "small": (_perf(servers.MLX_SMALL_LOG, controller.request_tracker)
                  if small_up else None),
        "clients": clients.client_configs(cfg),
        "config_backup": clients.latest_config_backup(cfg),
        "logs": log_rows,
    }


# --- the HTTP server ---------------------------------------------------------------


def _make_server(cfg, controller, downloads) -> ThreadingHTTPServer:
    """Bind the local API before the native window asks for its first snapshot."""
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

        def _trusted(self) -> bool:
            """Only a program on this machine gets an answer: the app's
            URLSession addresses 127.0.0.1 and sends no Origin, a web page
            does one or the other (see net.py)."""
            if (not loopback_host(self.headers, self.server.server_address[1])
                    or from_web_page(self.headers)):
                self.reply(403, {"error": "the local API only answers programs"
                                         " on this machine"})
                return False
            return True

        def do_GET(self):
            if not self._trusted():
                return
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/snapshot":
                self.reply(200, snapshot(cfg, controller))
            elif url.path == "/downloads":
                self.reply(200, downloads.all_progress())
            elif url.path == "/settings":
                model = urllib.parse.parse_qs(url.query).get("model", [None])[0]
                if model:
                    self.reply(200, _context_settings(cfg, model))
                else:
                    self.reply(400, {"error": "expected a model name"})
            elif url.path == "/models":
                self._models(url.query)
            else:
                self.send_error(404)

        def _models(self, query):
            params = urllib.parse.parse_qs(query)

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
            except Exception as exc:
                applog.log.exception("model catalog failed")
                self.reply(500, {"error": str(exc)})
            else:
                self.reply(200, result)

        def do_POST(self):
            if not self._trusted():
                return
            if self.path not in ("/run", "/delete", "/download", "/pause",
                                 "/resume", "/cancel", "/settings", "/stop",
                                 "/rollback"):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 4096:
                    raise ValueError
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError
            except ValueError:
                self.close_connection = True   # the body was not read to the end
                self.reply(400, {"error": "invalid JSON"})
                return
            model = payload.get("model")
            if self.path == "/rollback":
                try:
                    restored = clients.rollback_latest_config_backup(cfg)
                except (OSError, ValueError, SystemExit) as exc:
                    self.reply(409, {"error": str(getattr(exc, "code", exc))})
                else:
                    self.reply(200, {"restored": restored})
            elif self.path == "/settings":
                try:
                    _update_context_settings(cfg, payload)
                except ValueError as exc:
                    self.reply(400, {"error": str(exc)})
                    return
                self.reply(200, _context_settings(cfg, model))
            elif not isinstance(model, str):
                self.reply(400, {"error": "expected a model name"})
            elif self.path == "/run":
                self.reply(*controller.start(
                    model, payload.get("config_write_consent") is True))
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
        return ThreadingHTTPServer((HOST, cfg.widget_port), Handler)
    except OSError:
        raise SystemExit(f"   widget port :{cfg.widget_port} is in use"
                         " — set WIDGET_PORT to a free port") from None


def run_api(cfg) -> int:
    """Foreground API child of MLX Stack.app; SIGTERM also stops its model."""
    applog.setup()
    controller = StackController(cfg)
    downloads = DownloadManager(cfg, controller.refresh_models)
    controller.downloads = downloads
    server = _make_server(cfg, controller, downloads)
    parent = os.getppid()

    def watch_parent():
        # A killed/crashed app cannot run applicationWillTerminate. Once
        # launchd adopts us, keeping the API alive would leave a headless
        # model and a mystery Python process with no status-bar icon.
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
        downloads.stop()
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
