"""llama.cpp backend: a managed llama-server binary (GGUF models) and a
minimal GGUF header parser.

Each MLX Stack release pins a reviewed macOS arm64 llama.cpp build and its
SHA-256. The archive is verified before extraction and the extracted directory
is kept by build id. Upgrades are explicit code changes, not silent upgrades.
The binary needs its dylibs next to it (rpath-relative), so the whole extracted
directory is retained."""
import hashlib
import shutil
import struct
import tarfile
import urllib.request
import time
from pathlib import Path

from . import ui
from .config import stack_home

_LLAMA_TAG = "b12776"
_LLAMA_SHA256 = "cb0c5c97a1a4ad50a0d64e228a67694de5c3a876d1ae0077b39e3c2381f3a039"
_ASSET = "llama-{tag}-bin-macos-arm64.tar.gz"
_DOWNLOAD = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"


def _fetch(url, timeout: int, tries: int = 4):
    """urlopen with flaky-network retries for the release CDN."""
    last: Exception | None = None
    for n in range(1, tries + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — re-raised on the last try
            last = exc
            if n < tries:
                time.sleep(5 * n)
    raise last


def _llama_root() -> Path:
    return stack_home() / "llama.cpp"


def _download_build(root: Path) -> str:
    tag = _LLAMA_TAG
    url = _DOWNLOAD.format(tag=tag, asset=_ASSET.format(tag=tag))
    ui.info(f"Downloading llama.cpp {tag} (~16MB) from GitHub releases")
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / f"llama-{tag}.tar.gz"
    digest = hashlib.sha256()
    with _fetch(url, timeout=300) as r, open(tar_path, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {done * 100 / total:5.1f}%"
                      f" ({done // 1024 // 1024}MB)", end="", flush=True)
    print()
    actual = digest.hexdigest()
    if actual != _LLAMA_SHA256:
        tar_path.unlink(missing_ok=True)
        raise SystemExit(f"   llama.cpp {tag}: sha256 mismatch after download "
                         f"(expected {_LLAMA_SHA256}, got {actual})")
    try:
        # filter="data" (py3.12+): refuses path traversal / absolute members
        with tarfile.open(tar_path) as tf:
            tf.extractall(root, filter="data")
    finally:
        tar_path.unlink(missing_ok=True)
    server = root / f"llama-{tag}" / "llama-server"
    if not server.is_file():
        raise SystemExit(f"   llama-server missing in the extracted {tag} tarball")
    server.chmod(0o755)
    (root / "build.txt").write_text(tag + "\n")
    ui.ok(f"llama-server {tag} installed at {server}")
    return str(server)


def ensure_llama_server(override: str) -> str:
    """Absolute path to a llama-server binary: the path from
    OLLMLX_LLAMA_SERVER when it names a file > one on PATH > the managed
    release-pinned download under ~/.mlx-stack/llama.cpp."""
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p)
        raise SystemExit(f"   OLLMLX_LLAMA_SERVER={override} — not a file")
    found = shutil.which("llama-server")
    if found:
        return found
    root = _llama_root()
    server = root / f"llama-{_LLAMA_TAG}" / "llama-server"
    if server.is_file():
        return str(server)
    return _download_build(root)


# --- GGUF header -----------------------------------------------------------------
#
# GGUF repos ship no config.json; the context window and KV geometry live
# in the header of the .gguf itself. We parse just the metadata key/values
# (tensor tables are skipped) — everything the context math needs.

_MAGIC = b"GGUF"
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           10: "<Q", 11: "<q"}   # u8..i64 by spec type id
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_str(f) -> str:
    n = _u64(f)
    return f.read(n).decode("utf-8", "replace")


def _skip_array(f, elem_type: int, count: int) -> None:
    if elem_type == 8:
        for _ in range(count):
            f.seek(_u64(f), 1)
    elif elem_type == 9:
        raise ValueError("nested arrays are not in the spec — bail out")
    else:
        f.seek(_SIZES[elem_type] * count, 1)


def gguf_meta(path) -> dict:
    """Scalar metadata from a GGUF v2/v3 header (arrays skipped);
    {} when the file is unreadable or truncated."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != _MAGIC:
                return {}
            version = _u32(f)
            if version < 2:
                return {}
            f.seek(8, 1)          # tensor count — never read here
            out = {}
            for _ in range(_u64(f)):
                key = _read_str(f)
                t = _u32(f)
                if t == 8:
                    out[key] = _read_str(f)
                elif t == 9:
                    _skip_array(f, _u32(f), _u64(f))
                elif fmt := _SCALAR.get(t):
                    out[key] = struct.unpack(fmt, f.read(_SIZES[t]))[0]
                elif t == 6:
                    out[key] = struct.unpack("<f", f.read(4))[0]
                elif t in (7, 12):
                    out[key] = struct.unpack("<?" if t == 7 else "<d",
                                             f.read(1 if t == 7 else 8))[0]
                else:
                    break         # unknown type — the rest can't be skipped
            return out
    except (OSError, struct.error, ValueError):
        return {}


def gguf_ctx_kv(path) -> tuple[int, float | None]:
    """(context_length, kv_bytes_per_token) from a GGUF header, the same
    fp16-KV formula as the MLX path (2 x kv_heads x head_dim x layers x 2).
    (32768, None) when the header doesn't say — llama-server fills in the
    real defaults."""
    m = gguf_meta(path)
    arch = m.get("general.architecture")
    if not isinstance(arch, str):
        return 32768, None
    ctx = m.get(f"{arch}.context_length") or 32768
    layers = m.get(f"{arch}.block_count")
    kvh = m.get(f"{arch}.attention.head_count_kv") or m.get(f"{arch}.attention.head_count")
    head_dim = m.get(f"{arch}.attention.key_length")
    if not all(isinstance(x, (int, float)) and x for x in (layers, kvh, head_dim)):
        return int(ctx), None
    return int(ctx), 2 * kvh * head_dim * layers * 2
