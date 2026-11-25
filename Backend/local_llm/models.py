"""Model knowledge: repo-name parsing, RAM estimates and the hardware-fit score."""
import math
import re

# GGUF LoRA adapters are not standalone models: the user-facing repo is
# resolved to its exact base weights plus the adapter file. Q8_0 is the
# adapter author's recommended default; lower quants drift from BF16 training.
GGUF_LORAS = {
    "harness-hub/Qwen3.8-27B-pi-adapted": {
        "base": "ggml-org/Qwen3.8-27B-GGUF",
        "repo": "harness-hub/Qwen3.8-27B-pi-adapted-GGUF",
        "file": "Qwen3.8-27B-pi-lora-F16.gguf",
        "quant": "Q8_0",
    },
}

# what to recommend on a machine with at least this much RAM (GB)
_FITS_RAM = (
    (80, "mlx-community/Qwen3-235B-A22B-3bit"),
    (24, "mlx-community/Qwen3-30B-A3B-4bit"),
    (16, "mlx-community/Qwen3-32B-4bit"),
    (0, "mlx-community/Qwen3-8B-4bit"),
)

# family -> bonus for quality / MLX support
FAMILIES = (
    ("qwen3", 8), ("qwen2.5", 4), ("qwen2", 2), ("llama-3.3", 6),
    ("llama-3.2", 5), ("llama-3.1", 5), ("llama-3", 3), ("gpt-oss", 7),
    ("gemma-3", 6), ("gemma-2", 4), ("gemma", 3), ("kimi", 5),
    ("mixtral", 4), ("mistral", 3), ("deepseek", 4), ("glm", 3), ("phi", 3),
)

# quantization -> bytes per parameter in the weights. GGUF names first and
# longest-first: "iq4_xs" must win over "q4_k", and "iq3"/"iq2" match their
# -xxs/-m variants closely enough
QUANTS = (
    ("iq4_xs", 0.51), ("q8_0", 1.06), ("q6_k", 0.83), ("q5_k", 0.70),
    ("q4_k", 0.57), ("q3_k", 0.45), ("iq3", 0.36), ("q2_k", 0.32),
    ("iq2", 0.24),
    ("4bit", 0.58), ("8bit", 1.10), ("6bit", 0.82), ("7bit", 0.95),
    ("5bit", 0.70), ("3bit", 0.45), ("mxfp4", 0.60),
    ("bf16", 2.05), ("fp16", 2.05), ("f16", 2.05),
)

# rough RAM table (GB) for names without a recognized size; matched on
# [-_.] word boundaries (a bare "4b" would otherwise match inside "4bit"),
# longest pattern first ("30B-A3B" must win over "30B")
_RAM_TABLE = (
    ("0.5B", 2), ("0.6B", 2), ("1.7B", 4), ("1B", 4), ("3B", 6), ("4B", 6),
    ("7B", 8), ("8B", 8), ("9B", 8), ("14B", 12), ("20B", 16), ("27B", 16),
    ("30B-A3B", 24), ("30B", 20), ("32B", 20), ("70B", 40), ("235B", 80),
    ("gpt-oss-20b", 16),
)
_RAM_RES = tuple(
    (re.compile(rf"(?:^|[-_.]){re.escape(pat.lower())}(?:[-_.]|$)"), gb)
    for pat, gb in sorted(_RAM_TABLE, key=lambda t: -len(t[0])))


def recommend_for_ram(ram: int) -> str:
    return next(model for gb, model in _FITS_RAM if ram >= gb)


def parse_model(mid: str):
    """(total_B, active_B, bpp, family_bonus, moe, instruct) from repo name."""
    s = mid.lower()
    total = active = None
    m = re.search(r"(\d+(?:\.\d+)?)b[-_ ]?a(\d+(?:\.\d+)?)b", s)   # 235B-A22B
    if m:
        total, active = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"[-_](\d+(?:\.\d+)?)b(?:[-_]|$)", s)        # -8B- / -0.6B
        if m:
            total = float(m.group(1))
    bpp = 0.58  # most mlx-community repos are 4bit
    for q, b in QUANTS:
        if q in s:
            bpp = b
            break
    if mid in GGUF_LORAS:
        bpp = 1.06  # registered default: Q8_0 base + ~0.3GB F16 LoRA
    fam = 0
    for name, bonus in FAMILIES:
        if name in s:
            fam = bonus
            break
    moe = active is not None or "moe" in s or "mixtral" in s
    instruct = any(t in s for t in ("instruct", "-it", "it-", "chat"))
    return total, active, bpp, fam, moe, instruct


def ram_need(total_b, bpp: float):
    """Weights + headroom: without it Metal easily OOMs on the KV cache."""
    return None if total_b is None else total_b * bpp + 1.5


def ram_need_gb(mid: str) -> int | None:
    """RAM estimate in GB from the repo name (params × bytes/param + 1.5G
    headroom); rough table for names without a parseable size, None when
    the name tells nothing — callers show '?' instead of a guess."""
    total, _active, bpp, *_ = parse_model(mid)
    rn = ram_need(total, bpp)
    if rn is not None:
        return max(1, round(rn))
    for r, gb in _RAM_RES:
        if r.search(mid.lower()):
            return gb
    return None


def ram_estimate_gb(mid: str, lf_entry: dict | None = None,
                    size: str = "") -> int | None:
    """Best-effort RAM estimate (GB) for listings, honest about ignorance:

    1. the weights' actual disk size + ~15% headroom — for an installed
       model this is ground truth; the catalog's recommended_ram_gb lags
       exotic re-quants badly (a 95.8G MoE was "27GB")
    2. repo-name parsing (params × bytes/param, table fallback)
    3. the llmfit catalog's recommended_ram_gb (knows some odd names)
    None: print '?' rather than invent a number."""
    if size:
        m = re.match(r"([\d.]+)\s*([KMG])", size)
        if m:
            gb = (float(m.group(1))
                  * {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0}[m.group(2)])
            return max(1, round(gb * 1.15))
    rn = ram_need_gb(mid)
    if rn is not None:
        return rn
    if lf_entry:
        n = lf_entry.get("recommended_ram_gb")
        if isinstance(n, (int, float)) and n > 0:
            return int(n)
    return None


def score(sys_ram: int, mid: str, downloads: int):
    """Hardware-fit score for this machine: higher is better.

    RAM fit (weights = params × bytes/param of the quant), quality ~ √params
    (while it fits), MoE bonus (few active params — fast on Apple Silicon),
    family, instruct variant; downloads are only a soft tiebreak."""
    total, active, bpp, fam, moe, instruct = parse_model(mid)
    rn = ram_need(total, bpp)
    sc = 0.0
    if rn is not None:
        if rn <= sys_ram * 0.85:
            sc += 40          # fits with headroom
        elif rn <= sys_ram:
            sc += 25          # fits tightly
        elif rn <= sys_ram * 1.2:
            sc += 5           # swap only — will be slow
        else:
            sc -= 35          # won't fit
    if total:
        sc += min(22.0, 6.0 * math.sqrt(total))
    if moe:
        sc += 10
    sc += fam
    if instruct:
        sc += 4
    sc += 2.0 * math.log10(downloads + 1)
    return sc, rn, total, active
