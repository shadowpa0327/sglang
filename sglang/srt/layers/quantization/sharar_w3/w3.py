# Vendored from abdelfattah-lab/sharar @ 0ecb67f (sharar/kernels/w3.py, Apache-2.0) for the H100 port:
# the kernel builds for the local GPU's arch (sm_90a on H100, where csrc's FP8 mma path is replaced, see w3_gemm.cu),
# `enabled()` follows this package's forward-mode switch instead of sharar's engine config, and sharar's model
# attachment (attach / attach_model / attach_draft / interleave_rows) is replaced by `attach.py`.
"""Three-bit decode weights: an eight-entry E4M3 codebook, an E4M3 scale per 32 weights, and tensor-core GEMMs for up to 128 rows.

Format (`W3Weight`): for W [N, K], groups of 32 along K. Each weight is a
3-bit code into `codebook` (eight E4M3 values, the default the Lloyd-Max
levels of a Gaussian scaled so the largest is 3.5); each group has an E4M3
scale, each row an FP32 multiplier that restores the scale's range. That is
3 bits of code plus 8 bits of scale per 32 weights: 3.25 bits per weight, 72%
of NVFP4's 4.5 and 41% of FP8's 8. A four-entry codebook gives the two-bit
variant (two planes, 2.25 bits per weight; the MTP draft head's `w2` copy),
served by the same kernel.

Codes are stored as bit planes in the order the CUDA kernel's mma fragments
consume them (`csrc/w3_gemm.cu`): a warp owns 32 output rows and streams
128-k chunks; lane (g, t) holds, per chunk and per 8-row tile, one 32-bit
word per plane whose bit (4j + b + rot(s)) mod 32 is bit b of the weight at
row 8*tile + g, k = 16*s + 4*t + j. The kernel turns the three words into
E4M3 B-fragment bytes with three rotates, three masked ORs and one PRMT
lookup, so expansion costs under two integer instructions per weight and
the kernel is bound by the weight stream at 16 to 32 activation rows. From
33 rows a second kernel (`w3_gemm_wide`) stages the activations in shared
memory and feeds eight row tiles per warp from one expansion; `WIDE` picks
it and its tiling per shape and row count (the 27B's projections and draft; Flash-Next's trunk stays on the narrow
kernel, at most NARROW_ROWS rows).

Activations are E4M3: the engine's static-scale projection inputs as they
are, BF16 MLP inputs quantized per row and 32-group by `quant_groups` (the
kernel applies the activation group scale together with the weight group
scale; a single per-row scale measurably lost task accuracy). Enabled for
decode-sized calls (at most `MAX_ROWS` rows) when `SHARAR_W3_DECODE=1` and a
linear carries a packed copy (`attach_model`); prefill keeps the NVFP4 and
FP8 weights and kernels. Quantization here is nearest-level from the served
weights; calibrated source weights use `pack_codes` to retain fitted scales
and load through `SHARAR_W3_CHECKPOINT` before graph capture.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import torch



def enabled() -> bool:
    """Whether the current forward serves its covered linears from the 3-bit copies (`attach.w3_active`)."""
    from sglang.srt.layers.quantization.sharar_w3 import attach
    return attach.w3_active()


MAX_ROWS = 128
NARROW_ROWS = 64  # the narrow kernel's limit (four row tiles per warp)
GROUP = 32
CHUNK = 128
FP8_MAX = 448.0
# Eight levels, largest 3.5. Uniform is the classic (q - 3.5) grid; Lloyd-Max is the MSE-optimal
# Gaussian quantizer's levels (0.245, 0.756, 1.344, 2.152 sigma) scaled to 3.5 and rounded to E4M3.
CODEBOOKS = {
    "uniform": (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5),
    "lloyd": (-3.5, -2.25, -1.25, -0.40625, 0.40625, 1.25, 2.25, 3.5),
    # two-bit: the Gaussian Lloyd-Max levels (0.453, 1.510 sigma) scaled to 1.5 and rounded to E4M3
    "lloyd2": (-1.5, -0.4375, 0.4375, 1.5),
    # eight levels with a zero, for sparse matrices (Nemotron's early down projections are up to 94% exact zeros in
    # the NVFP4 checkpoint, which Lloyd's levels cannot hold: relative error 0.40 against 0.124 with these)
    "zero8": (-3.5, -2.5, -1.5, -0.625, 0.0, 0.875, 2.0, 3.5),
}
# Per-group scale multipliers `pack(fit_scales=True)` tries (the least squared error wins): with four levels,
# scaling the largest to the group's maximum wastes them on one outlier.
FIT_SCALES = tuple(0.55 + 0.05 * i for i in range(10))
DEFAULT_CODEBOOK = "lloyd"  # every served copy; "uniform" remains for the packing tests and studies
_SRC = Path(__file__).resolve().parent / "w3_gemm.cu"


@dataclass
class W3Weight:
    planes: torch.Tensor  # [N/32, K/128, bits, 32, 4] uint32, fragment order (module docstring)
    scales: torch.Tensor  # [N/32, K/128, 4, 32] uint8 E4M3: for lane column pair t, byte (4*tile + group)*2 + i is row 8*tile + 2*t + i
    row_scale: torch.Tensor  # [N] float32
    out_features: int
    in_features: int
    codebook: tuple[float, ...]
    lut: int  # the codebook as eight E4M3 bytes, entry i in byte i (the kernel's two 32-bit PRMT sources)

    @property
    def bytes(self) -> int:
        return self.planes.numel() * 4 + self.scales.numel() + self.row_scale.numel() * 4

    @property
    def bits(self) -> int:
        return self.planes.shape[2]


def lut_of(codebook: tuple[float, ...]) -> int:
    raw = torch.tensor(codebook, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).tolist()
    return struct.unpack("<q", bytes(raw + [0] * (8 - len(raw))))[0]


def _codebook(codebook) -> tuple[float, ...]:
    codebook = DEFAULT_CODEBOOK if codebook is None else codebook
    levels = CODEBOOKS[codebook] if isinstance(codebook, str) else tuple(float(v) for v in codebook)
    if len(levels) not in (4, 8) or list(levels) != sorted(levels):
        raise ValueError("a codebook is eight (or, two-bit, four) ascending levels")
    exact = torch.tensor(levels).to(torch.float8_e4m3fn).float().tolist()
    if exact != list(levels):
        raise ValueError("codebook levels must be exactly representable in E4M3")
    return tuple(levels)


def _rotations(device) -> torch.Tensor:
    s = torch.arange(8, device=device)
    return 16 * (s // 4) + (s % 4)  # rot(s)


def pack(weight: torch.Tensor, codebook=None, rows_per_chunk: int = 4096, fit_scales: bool = False) -> W3Weight:
    """Nearest-level 3-bit (2-bit with a four-level codebook) quantization of a [N, K] float matrix
    (N % 128 == 0, K % 128 == 0) in fragment order. `fit_scales`: each group's scale is the one of
    `FIT_SCALES` x its maximum with the least squared error, instead of the maximum itself.

    Rows are processed `rows_per_chunk` at a time so the integer temporaries stay
    a few hundred MB even for the lm_head (the whole-matrix version needed
    several 10 GB tensors and tripped the memory guard)."""
    n, k = weight.shape
    if k % CHUNK or n % 128:
        raise ValueError("N and K must be multiples of 128")
    levels = _codebook(codebook)
    planes, scales, row_scales = [], [], []
    for r0 in range(0, n, rows_per_chunk):
        pl, sc, rs = _pack_rows(weight[r0:r0 + rows_per_chunk], levels, fit_scales)
        planes.append(pl); scales.append(sc); row_scales.append(rs)
    cat = (lambda xs: xs[0] if len(xs) == 1 else torch.cat(xs))
    return W3Weight(cat(planes), cat(scales), cat(row_scales), n, k, levels, lut_of(levels))


def _pack_rows(weight: torch.Tensor, levels: tuple[float, ...], fit_scales: bool = False):
    n, k = weight.shape
    dev = weight.device
    w = weight.float().reshape(n, k // GROUP, GROUP)
    amax = w.abs().amax(dim=-1)  # [N, G]
    lv = torch.tensor(levels, device=dev)
    mids = (lv[1:] + lv[:-1]) / 2
    scale = amax / max(abs(v) for v in levels)
    if fit_scales:
        best = None
        for m in FIT_SCALES:
            s = scale * m
            err = ((lv[torch.bucketize(w / s.clamp_min(1e-30)[:, :, None], mids)] * s[:, :, None] - w) ** 2).sum(-1)
            if best is None:
                best, fitted = err, s
            else:
                better = err < best
                best, fitted = torch.where(better, err, best), torch.where(better, s, fitted)
        scale = fitted
    row_scale = (scale.amax(dim=-1) / FP8_MAX).clamp_min(1e-12)  # [N]
    scale_fp8 = (scale / row_scale[:, None]).clamp(0, FP8_MAX).to(torch.float8_e4m3fn)
    scale_eff = scale_fp8.float() * row_scale[:, None]
    normalized = w / scale_eff.clamp_min(1e-30)[:, :, None]
    del w
    q = torch.bucketize(normalized.reshape(n, k), mids, out_int32=True)  # [N, K] codes 0..7 (0..3)
    del normalized
    return _pack_codes_rows(q, scale_fp8, row_scale, 2 if len(levels) == 4 else 3)


def _pack_codes_rows(q: torch.Tensor, scale_fp8: torch.Tensor, row_scale: torch.Tensor, bits: int = 3):
    n, k = q.shape
    dev = q.device
    # fragment order: n = 32 nb + 8 nt + g, k = 128 kc + 16 s + 4 t + j
    q = q.reshape(n // 32, 4, 8, k // CHUNK, 8, 4, 4)  # [nb, nt, g, kc, s, t, j]
    rot = _rotations(dev)  # [s]
    j = torch.arange(4, device=dev)
    words = []
    for b in range(bits):
        pos = ((4 * j[None, :] + b + rot[:, None]) % 32).to(torch.int64)  # [s, j]
        bit = ((q >> b) & 1).to(torch.int64)
        words.append((bit << pos[None, None, None, None, :, None, :]).sum(dim=(4, 6)))  # [nb, nt, g, kc, t]
        del bit
    words = torch.stack(words, dim=2)  # [nb, nt, b, g, kc, t]
    words = words.permute(0, 4, 3, 5, 1, 2).reshape(n // 32, k // CHUNK, 32, 4 * bits)  # [nb, kc, lane, bits nt + b]
    planes = words.reshape(n // 32, k // CHUNK, 32, bits, 4).permute(0, 1, 3, 2, 4)  # [nb, kc, i, lane, e]
    planes = planes.bitwise_and(0xFFFFFFFF).to(torch.uint32).contiguous()
    # a lane's accumulator columns are n = 8 nt + 2 t + i, so scales are laid out per t: [nb, kc, t, (nt, grp, i)]
    sc = scale_fp8.view(torch.uint8).reshape(n // 32, 4, 4, 2, k // CHUNK, 4)  # [nb, nt, t, i, kc, grp]
    scales = sc.permute(0, 4, 2, 1, 5, 3).reshape(n // 32, k // CHUNK, 4, 32).contiguous()
    return planes, scales, row_scale.contiguous()


def pack_codes(q: torch.Tensor, scale_fp8: torch.Tensor, row_scale: torch.Tensor,
               codebook=None, rows_per_chunk: int = 4096) -> W3Weight:
    """Pack calibrated codes and their exact scales without quantizing again."""
    n, k = q.shape
    if n % 128 or k % CHUNK or rows_per_chunk < 128 or rows_per_chunk % 128:
        raise ValueError("N, K and rows_per_chunk must be multiples of 128")
    levels = _codebook(codebook)
    if q.dtype not in (torch.uint8, torch.int32, torch.int64) or torch.any((q < 0) | (q >= len(levels))):
        raise ValueError(f"codes must be integers between zero and {len(levels) - 1}")
    if scale_fp8.shape != (n, k // GROUP) or scale_fp8.dtype != torch.float8_e4m3fn or row_scale.shape != (n,):
        raise ValueError("calibrated scales do not match the weight shape")
    if not torch.isfinite(scale_fp8.float()).all() or not torch.isfinite(row_scale).all() or (row_scale <= 0).any() or (scale_fp8.float() < 0).any():
        raise ValueError("calibrated scales must be finite and nonnegative, with positive row scales")
    bits = 2 if len(levels) == 4 else 3
    chunks = [_pack_codes_rows(q[r:r + rows_per_chunk].int(), scale_fp8[r:r + rows_per_chunk], row_scale[r:r + rows_per_chunk], bits)
              for r in range(0, n, rows_per_chunk)]
    return W3Weight(*(torch.cat([part[i] for part in chunks]) for i in range(3)), n, k, levels, lut_of(levels))


def codes(w3: W3Weight) -> torch.Tensor:
    """The codes of a packed matrix, [N, K] int64."""
    n, k, bits = w3.out_features, w3.in_features, w3.bits
    dev = w3.planes.device
    words = w3.planes.to(torch.int64).permute(0, 1, 3, 2, 4).reshape(n // 32, k // CHUNK, 32, 4, bits)  # [nb, kc, lane, nt, b]
    words = words.reshape(n // 32, k // CHUNK, 8, 4, 4, bits).permute(0, 4, 2, 1, 3, 5)  # [nb, kc, g, t, nt, b] -> [nb, nt, g, kc, t, b]
    rot = _rotations(dev)
    j = torch.arange(4, device=dev)
    q = torch.zeros(n // 32, 4, 8, k // CHUNK, 8, 4, 4, dtype=torch.int64, device=dev)  # [nb, nt, g, kc, s, t, j]
    for b in range(bits):
        pos = (4 * j[None, :] + b + rot[:, None]) % 32  # [s, j]
        q |= ((words[..., b][:, :, :, :, None, :, None] >> pos[None, None, None, None, :, None, :]) & 1) << b
    return q.reshape(n, k)


def group_scales(w3: W3Weight) -> torch.Tensor:
    """Effective per-group scales, [N, K/32] float32."""
    n, k = w3.out_features, w3.in_features
    sc = w3.scales.reshape(n // 32, k // CHUNK, 4, 4, 4, 2).permute(0, 3, 2, 5, 1, 4).reshape(n, k // GROUP)  # [nb, kc, t, nt, grp, i] -> [nb, nt, t, i, kc, grp]
    return sc.contiguous().view(torch.float8_e4m3fn).float() * w3.row_scale[:, None]


def dequantize(w3: W3Weight) -> torch.Tensor:
    """FP32 reference of the packed matrix, [N, K]."""
    lv = torch.tensor(w3.codebook, device=w3.planes.device)
    return lv[codes(w3)] * group_scales(w3).repeat_interleave(GROUP, dim=1)


# ------------------------------------------------------------------ the kernel

@lru_cache(maxsize=1)
def _module():
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}")
    # Images can restore older source timestamps into a shared build cache.
    # Separate source revisions so Ninja cannot reuse another revision's object.
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:16]
    return load(name=f"sharar_w3_gemm_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3", "--use_fast_math", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"], verbose=False)


_WORKSPACE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Split-K factor per (N, K, rows): profiling/w3_bench.py on GB10. More blocks help the weight stream
# at 16 rows; at 64 rows the partial-sum traffic and the tensor-core work per block argue for fewer splits.
SPLITS: dict[tuple[int, int, int], int] = {  # (N, K, rows) -> splits
    (5120, 17408, 16): 8, (5120, 17408, 32): 2, (5120, 17408, 64): 2,
    (10240, 5120, 16): 1, (10240, 5120, 32): 1, (10240, 5120, 64): 1,
    (6144, 5120, 16): 1, (6144, 5120, 32): 1, (6144, 5120, 64): 2,
    (5120, 6144, 16): 2, (5120, 6144, 32): 2, (5120, 6144, 64): 2,
    (14336, 5120, 16): 1, (14336, 5120, 32): 1, (14336, 5120, 64): 1,
    (4096, 5120, 16): 4, (4096, 5120, 32): 4, (4096, 5120, 64): 2,
    (5120, 4096, 16): 4, (5120, 4096, 32): 1, (5120, 4096, 64): 2,
}
SPLIT_TARGET = {16: 192, 32: 96, 64: 48}  # blocks (4 warps, 128 rows) a launch should reach before K stays whole


def splits_for(n: int, k: int, m: int = 16) -> int:
    rows = 16 if m <= 16 else 32 if m <= 32 else 64
    if (n, k, rows) in SPLITS:
        return SPLITS[(n, k, rows)]
    blocks, chunks, splits = n // 128, k // CHUNK, 1
    while blocks * splits < SPLIT_TARGET[rows] and chunks % (2 * splits) == 0 and splits < 8:
        splits *= 2
    return splits


# The wide kernel per (N, K) and row bucket (48: 33-48 rows, 64: 49-64, 128: more): (config, splits), config =
# MT * 1000 + WM * 100 + WN * 10 + STAGES (csrc/w3_gemm.cu `wide`), or None for the narrow kernel. Fastest of the
# compiled configurations and splits on GB10 with the weights cycled through more than L2 holds (the 128 bucket:
# least total time over 80, 96, 112 and 128 rows). At 33-64 rows it is 2-10% faster than the narrow kernel on the
# large matrices and 20-40% on the small ones; up to 32 rows the narrow kernel's weight stream wins.
WIDE: dict[tuple[int, int], dict[int, tuple[int, int] | None]] = {
    # the 27B: MLP gate_up, down; GDN in_qkv, in_z (and attention o), out; attention qkv; DFlash2 draft q, o
    (34816, 5120): {48: None, 64: (4143, 1), 128: (8183, 1)},
    (5120, 17408): {48: None, 64: (2244, 1), 128: (8183, 2)},
    (10240, 5120): {48: None, 64: (4143, 1), 128: (8183, 1)},
    (6144, 5120): {48: (2244, 1), 64: (4144, 1), 128: (4243, 1)},
    (5120, 6144): {48: (2244, 1), 64: (4124, 1), 128: (4243, 1)},
    (14336, 5120): {48: (2244, 1), 64: None, 128: (4142, 1)},
    (4096, 5120): {48: (2144, 4), 64: (2144, 4), 128: (8144, 1)},
    (5120, 4096): {48: (2244, 1), 64: (2244, 1), 128: (8144, 1)},
}


def wide_for(n: int, k: int, m: int) -> tuple[int, int] | None:
    """(config, splits) of the wide kernel for an [m, k] x [k, n] call, or None for the narrow kernel."""
    if m <= 32:
        return None
    if (n, k) in WIDE:
        return WIDE[(n, k)][48 if m <= 48 else 64 if m <= 64 else 128]
    if m <= NARROW_ROWS:
        return None
    # other shapes: eight row tiles per warp, K split until the launch fills the GPU about once
    blocks, chunks, splits = (n // 128) * -(-m // 128), k // CHUNK, 1
    while blocks * splits < 40 and chunks % (2 * splits) == 0 and splits < 4:
        splits *= 2
    return 8144, splits


# Per-arch choices per (N, K) and row bucket, measured by the port's sweep (sharar_port/scripts/w3_tune_table.py,
# weights cycled through more than L2): H100 (SM90) and B200/GB200 (SM100). ["narrow", splits] or
# ["wide", config, splits]; a bucket covers the rows above the previous bucket up to itself. Other archs (GB10's
# SM120/121) use SPLITS / WIDE.
_TUNING_FILES = {9: "h100_tuning.json", 10: "sm100_tuning.json"}
_TUNED: dict[int, dict[str, dict[int, list]]] = {
    major: {nk: {int(m): c for m, c in d.items()} for nk, d in __import__("json").loads(path.read_text()).items()}
    for major, name in _TUNING_FILES.items() if (path := Path(__file__).resolve().parent / name).exists()
}
_TUNED_SHAPES = list(_TUNED.get(9, {}))  # the shapes the port serves (27B and Flash-Next TP2), for re-sweeping


def _tuned_choice(n: int, k: int, m: int, device) -> list | None:
    d = _TUNED.get(torch.cuda.get_device_capability(device)[0], {}).get(f"{n}x{k}")
    if d is None:
        return None
    for bucket in sorted(d):
        if m <= bucket:
            return d[bucket]
    return None


def _workspace(device: torch.device, floats: int, blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    ws, counters = _WORKSPACE.get(device, (None, None))
    if ws is None or ws.numel() < floats or counters.numel() < blocks:
        ws = torch.empty(max(floats, 8 << 20), dtype=torch.float32, device=device)
        counters = torch.zeros(max(blocks, 4096), dtype=torch.int32, device=device)
        _WORKSPACE[device] = (ws, counters)
    return ws, counters


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3 quantization of a BF16 [M, K] matrix: (x_q, scale[M]). One scale per row loses the small
    channels next to an outlier (GSM8K -1.7 points, HumanEval -4 with the uniform codebook); serving uses `quant_groups`."""
    out, scale = _module().quant_rows(x.contiguous())
    return out, scale


def quant_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(row, 32-group) E4M3 quantization of a BF16 [M, K] matrix: (x_q, scale[M, K/32]), MX-style block scaling
    that isolates outlier channels to their own group."""
    out, scale = _module().quant_groups(x.contiguous())
    return out, scale


def w3_linear(x: torch.Tensor, w3: W3Weight, x_scale: torch.Tensor | None = None, splits: int | None = None) -> torch.Tensor:
    """y = x @ W^T for x [M, K] in BF16 (quantized per row and 32-group here) or E4M3 with its scale; M <= MAX_ROWS.
    Returns BF16 [M, N]."""
    m, k = x.shape
    n = w3.out_features
    if m > MAX_ROWS or k != w3.in_features:
        raise ValueError("w3_linear: unsupported call")
    if x.dtype == torch.float8_e4m3fn:
        if x_scale is None:
            raise ValueError("FP8 activations need their scale")
        x_q, scale = x, x_scale
    else:
        x_q, scale = quant_groups(x)
    tuned = _tuned_choice(n, k, m, x.device) if w3.bits == 3 and splits is None else None
    if tuned is not None:
        if tuned[0] == "narrow":
            splits = tuned[1]
            ws, counters = _workspace(x.device, splits * m * n if splits > 1 else 0, n // 128)
            return _module().w3_gemm(x_q, scale, w3.planes, w3.scales, w3.row_scale, w3.lut, splits, ws, counters)
        config, parts = tuned[1], tuned[2]
        mt, wm, wn = config // 1000, config // 100 % 10, config // 10 % 10
        ws, counters = _workspace(x.device, parts * m * n if parts > 1 else 0, (n // (wn * 32)) * -(-m // (wm * mt * 16)))
        return _module().w3_gemm_wide(x_q, scale, w3.planes, w3.scales, w3.row_scale, w3.lut, config, parts, ws, counters)
    wide = wide_for(n, k, m) if w3.bits == 3 and splits is None else None
    if wide is not None:
        config, parts = wide
        rows = 16 * (config // 1000) * (config // 100 % 10)
        ws, counters = _workspace(x.device, parts * m * n if parts > 1 else 0, (n // 64) * -(-m // rows))
        return _module().w3_gemm_wide(x_q, scale, w3.planes, w3.scales, w3.row_scale, w3.lut, config, parts, ws, counters)
    if m > NARROW_ROWS:  # two-bit copies: the narrow kernel in parts
        tail = None if scale.numel() == 1 else scale
        return torch.cat([w3_linear(x_q[i:i + NARROW_ROWS], w3, scale if tail is None else tail[i:i + NARROW_ROWS], splits)
                          for i in range(0, m, NARROW_ROWS)])
    splits = splits or splits_for(n, k, m)
    ws, counters = _workspace(x.device, splits * m * n if splits > 1 else 0, n // 128)
    return _module().w3_gemm(x_q, scale, w3.planes, w3.scales, w3.row_scale, w3.lut, splits, ws, counters)
