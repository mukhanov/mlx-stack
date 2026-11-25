"""Thread-safe catalog projection shared by the native model manager."""
from concurrent.futures import ThreadPoolExecutor
import threading

from . import hf, llmfit, models, ui


class ModelCatalog:
    def __init__(self, cfg, download_manager=None, deleted=None):
        self.cfg = cfg
        self.download_manager = download_manager
        self.deleted = deleted if deleted is not None else set()
        self.lock = threading.Lock()
        self.ram_gb = ui.total_ram_gb()
        self.recommended = models.recommend_for_ram(self.ram_gb)
        self.lf = None

    def _load_llmfit(self, refresh=False):
        if self.lf is None or refresh:
            loaded = llmfit.load(self.cfg.ollmlx_home, refresh=refresh)
            if loaded is not None:
                self.lf = loaded
        return self.lf

    def _row(self, rid, entry=None, downloads=0, likes=0, installed=None,
             gguf=False):
        size, size_bytes, downloading = "", 0, False
        if installed is not None:
            _rid, size, downloading = installed
            size_bytes = hf.model_disk_bytes(self.cfg.hf_hub, rid)
        if self.download_manager:
            downloading = downloading or self.download_manager.active(rid)
        downloads = downloads or (entry.get("hf_downloads") or 0 if entry else 0)
        likes = likes or (entry.get("hf_likes") or 0 if entry else 0)
        local_score, ram, total, active = models.score(self.ram_gb, rid, downloads)
        if installed is not None:
            ram = models.ram_estimate_gb(rid, entry, size=size)
        elif ram is None and entry:
            ram = entry.get("recommended_ram_gb")
        rating = llmfit.score(entry, self.ram_gb) if entry else None
        progress = (self.download_manager.progress(rid)
                    if self.download_manager else {})
        return {
            "id": rid,
            "installed": installed is not None,
            "size": size,
            "size_bytes": size_bytes,
            "downloading": downloading,
            "download_error": (self.download_manager.error(rid)
                               if self.download_manager else None),
            "download_bytes": progress.get("bytes", 0),
            "download_total": progress.get("total", 0),
            "download_rate": progress.get("rate", 0.0),
            "download_paused": progress.get("paused", False),
            "deleted_before": rid in self.deleted,
            "downloads": int(downloads or 0),
            "likes": int(likes or 0),
            "score": round(rating["score"] if rating else local_score, 1),
            "rating": rating,
            "ram_gb": ram,
            "parameters_b": rating["params"] if rating else total,
            "active_parameters_b": active,
            "context": llmfit.ctx_str(entry) if entry else None,
            "tools": bool(entry and llmfit.has_tools(entry)),
            "gguf": gguf or "gguf" in rid.lower(),
            "recommended": rid == self.recommended,
            "helper": rid == self.cfg.small_model,
            "fits": None if ram is None else ram <= self.ram_gb,
        }

    def _refresh_metrics(self, rows, live_ids):
        """Refresh visible stale metrics and remote sizes; HF-page stats are live."""
        pending = {row["id"]: row for row in rows
                   if row["id"] not in live_ids
                   or (not row["installed"] and not row["size_bytes"])}
        if not pending:
            return

        def fetch(rid):
            try:
                return rid, hf.fetch_stats(rid, self.cfg.hf_token)
            except Exception:
                return rid, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(fetch, pending)
            for rid, stats in results:
                if stats is None:
                    continue
                row = pending[rid]
                row["downloads"] = stats["downloads"]
                row["likes"] = stats["likes"]
                if not row["installed"]:
                    row["size_bytes"] = stats["size_bytes"]
                    row["size"] = (ui.human_bytes(stats["size_bytes"])
                                   if stats["size_bytes"] else "")
                if row["rating"] is None:
                    score = models.score(self.ram_gb, rid, stats["downloads"])[0]
                    row["score"] = round(score, 1)

    def _downloaded(self, lf, query):
        installed = hf.scan_installed(self.cfg.hf_hub)
        rows = {row[0]: self._row(row[0], (lf or {}).get(row[0]), installed=row)
                for row in installed
                if not query or query.lower() in row[0].lower()}
        installed_ids = {row[0] for row in installed}
        active_ids = (self.download_manager.active_ids()
                      if self.download_manager else set())
        for rid in active_ids - rows.keys():
            if query and query.lower() not in rid.lower():
                continue
            row = self._row(rid, (lf or {}).get(rid))
            progress = self.download_manager.progress(rid)
            row["size_bytes"] = progress["bytes"]
            row["size"] = ui.human_bytes(progress["bytes"]) if progress["bytes"] else ""
            rows[rid] = row
        return list(rows.values()), installed_ids | active_ids

    def list(self, scope="community", fmt="mlx", query="", sort="fit",
             offset=0, limit=50, refresh=False, local_only=False):
        if scope not in ("community", "all") or fmt not in ("mlx", "gguf"):
            raise ValueError("invalid catalog scope or format")
        if sort not in ("fit", "dl"):
            raise ValueError("invalid sort mode")
        offset = max(0, int(offset))
        limit = max(1, min(100, int(limit)))
        query = query.strip()

        if local_only:
            downloaded, _local_ids = self._downloaded(self.lf, query)
            downloaded.sort(key=lambda row: (-row["score"], row["id"]))
            return {
                "ram_gb": self.ram_gb,
                "gpu": ui.gpu_name(),
                "recommended": self.recommended,
                "small_model": self.cfg.small_model,
                "downloaded": downloaded,
                "models": [],
                "has_more": False,
                "total": 0,
                "error": None,
            }

        with self.lock:
            lf = self._load_llmfit(refresh)
            downloaded, installed_ids = self._downloaded(lf, query)
            fetch_limit = offset + limit + len(installed_ids) + 1
            candidates = {}
            live_ids = set()
            error = None

            def add(rid, entry=None, downloads=0, likes=0, live=False):
                if rid in installed_ids:
                    return
                if live:
                    live_ids.add(rid)
                old = candidates.get(rid)
                candidates[rid] = (rid, entry or (old[1] if old else None),
                                   downloads or (old[2] if old else 0),
                                   likes or (old[3] if old else 0))

            if query:
                q = query.lower()
                if lf and fmt == "mlx":
                    for rid, entry in lf.items():
                        if q in rid.lower() and (scope == "all"
                                                 or rid.startswith(llmfit.PREFIX)):
                            add(rid, entry)
                try:
                    found = hf.fetch_page(0, fetch_limit, self.cfg.hf_token,
                                          scope, query, fmt)
                    if fmt == "mlx":
                        found += hf.fetch_page(0, fetch_limit, self.cfg.hf_token,
                                               scope, query, "gguf")
                    for item in found:
                        rid = item["id"]
                        add(rid, (lf or {}).get(rid), item["downloads"],
                            item["likes"], live=True)
                except Exception as exc:
                    error = f"Hugging Face search failed: {exc}"
            elif fmt == "mlx" and lf:
                for rid, entry in lf.items():
                    if scope == "all" or rid.startswith(llmfit.PREFIX):
                        add(rid, entry)
            else:
                try:
                    page = hf.fetch_page(0, fetch_limit,
                                         self.cfg.hf_token, scope, fmt=fmt)
                    for item in page:
                        rid = item["id"]
                        add(rid, (lf or {}).get(rid) if fmt == "mlx" else None,
                            item["downloads"], item["likes"], live=True)
                except Exception as exc:
                    error = f"Hugging Face catalog failed: {exc}"
                    page = []

            remote = [self._row(rid, entry, downloads, likes, gguf=fmt == "gguf")
                      for rid, entry, downloads, likes in candidates.values()]
            order = ((lambda row: (-row["downloads"], -row["score"],
                                   row["id"])) if sort == "dl" else
                     (lambda row: (-row["score"], -row["downloads"],
                                   row["id"])))
            remote.sort(key=order)
            total = len(remote)
            has_more = offset + limit < total
            remote = remote[offset:offset + limit]
            self._refresh_metrics(downloaded + remote, live_ids)
            downloaded.sort(key=lambda row: (-row["score"], row["id"]))
            remote.sort(key=order)
            return {
                "ram_gb": self.ram_gb,
                "gpu": ui.gpu_name(),
                "recommended": self.recommended,
                "small_model": self.cfg.small_model,
                "downloaded": downloaded,
                "models": remote,
                "has_more": has_more,
                "total": total,
                "error": error,
            }
