# Vendored from abdelfattah-lab/sharar @ 0ecb67f (sharar/kernels/w3_moe.py, Apache-2.0); the NVFP4 packing helpers
# (`unpack_fp4`, cheap fits) still import sharar and are not used by the port.
"""Three-bit routed experts for Flash-Next and Nemotron: the W3 format of `kernels/w3.py` per expert, and a grouped
kernel for any row count (`csrc/w3_gemm.cu`, namespace `moe`).

Nemotron's latent experts (`model/nemotron/moe.py`) take the same format and kernel without a gate: fc1 is up_proj
alone [2688, 1024] in the checkpoint's row order, fc2 [1024, 2688], h = relu(up)^2 (bf16, as the checkpoint's
ReLU^2), top 22 (2.25 MB an expert against NVFP4's 3.10). The kernel tells the two apart by the input width and
fc1's rows per expert (`gated`). Each of fc1 and fc2 carries its codebook (a sparse down projection may take one with a
zero level); Flash-Next's share one.

Format: each expert is two W3 matrices, fc1 [1280, 2560] and fc2 [2560, 640]: 3-bit codes into an eight-level
E4M3 codebook, one E4M3 scale per 32 weights, an FP32 multiplier per row (3.25 bits per weight plus 4 bytes a
row; 2.01 MB an expert against NVFP4's 2.76 MB). A layer's experts are stacked: `W3Experts`, whose tensors carry
the expert index in front of `W3Weight`'s. fc1's rows are reordered so every 32-row block holds sixteen up rows
and then the sixteen matching gate rows: a warp of the kernel then has an intermediate column's up and gate
values in the same lane and applies silu(gate) * up in its epilogue. With these the NVFP4 experts are never
loaded (`MoeLayer`); the checkpoint carries them as `layers.N.experts.w13` / `.w2` (weights/w3_checkpoint.py).

Activations are E4M3 per (row, 32-group) as in the trunk's W3 path (`w3.quant_groups`), the intermediate
likewise; CUTLASS's fused path quantizes both to NVFP4. `calibrate` fits the format from the checkpoint's NVFP4
experts: per-group scales on a clipping grid minimizing the weight error (`w3_calibration.fit_scales`), then the
nearest level (docs/w3-calibration.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.quantization.sharar_w3 import w3

HIDDEN, INTER, TOPK, EXPERTS = 2560, 640, 10, 512  # Flash-Next; Nemotron: 1024 (latent), 2688, 22, 512
ORDER_IN_KERNEL = 640  # up to this many (row, k) pairs the kernel sorts them itself
# From this many rows each pass of an expert (up to 112 pairs) runs in its own block, next to the expert's other passes,
# so its weights stream from DRAM about once instead of once per pass. The same bits either way.
WIDE_ROWS = 1024
RATIOS = tuple(round(0.5 + 0.025 * i, 3) for i in range(21))  # group scale = amax / max level * ratio, 0.5 .. 1.0


@dataclass
class W3Experts:
    planes13: torch.Tensor  # [E, 2I/32, H/128, 3, 32, 4] uint32, fc1 rows in fc1_order
    scales13: torch.Tensor  # [E, 2I/32, H/128, 4, 32] uint8 E4M3
    row13: torch.Tensor     # [E, 2I] float32
    planes2: torch.Tensor   # [E, H/32, I/128, 3, 32, 4]
    scales2: torch.Tensor
    row2: torch.Tensor      # [E, H]
    codebook: tuple[float, ...]
    lut: int
    fc2_codebook: tuple[float, ...]
    fc2_lut: int
    base: int = 0           # the first expert's id: experts base .. base + E - 1 (expert parallel: one rank's range)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.planes13, self.scales13, self.row13, self.planes2, self.scales2, self.row2))

    @property
    def experts(self) -> int:
        return self.planes13.shape[0]

    def matrix(self, e: int, fc1: bool) -> w3.W3Weight:
        """Expert e's fc1 (rows in fc1_order) or fc2 as a W3Weight (views, no copy)."""
        p, s, r = (self.planes13, self.scales13, self.row13) if fc1 else (self.planes2, self.scales2, self.row2)
        n, k = p.shape[1] * 32, p.shape[2] * w3.CHUNK
        return w3.W3Weight(p[e], s[e], r[e], n, k, *((self.codebook, self.lut) if fc1 else (self.fc2_codebook, self.fc2_lut)))

    def stacked(self, fc1: bool) -> w3.W3Weight:
        """All experts' fc1 or fc2 as one W3Weight of E * N rows (the checkpoint entry)."""
        p, s, r = (self.planes13, self.scales13, self.row13) if fc1 else (self.planes2, self.scales2, self.row2)
        e, n, k = p.shape[0], p.shape[1] * 32, p.shape[2] * w3.CHUNK
        return w3.W3Weight(p.reshape(e * n // 32, *p.shape[2:]), s.reshape(e * n // 32, *s.shape[2:]), r.reshape(-1),
                           e * n, k, *((self.codebook, self.lut) if fc1 else (self.fc2_codebook, self.fc2_lut)))

    @classmethod
    def from_stacked(cls, fc1: w3.W3Weight, fc2: w3.W3Weight, experts: int, base: int = 0) -> "W3Experts":
        def split(m):
            return (m.planes.reshape(experts, -1, *m.planes.shape[1:]), m.scales.reshape(experts, -1, *m.scales.shape[1:]),
                    m.row_scale.reshape(experts, -1))
        return cls(*split(fc1), *split(fc2), fc1.codebook, fc1.lut, fc2.codebook, fc2.lut, base)


def gated(experts: W3Experts) -> bool:
    """Flash-Next's gated experts (fc1 = [up; gate], 2 I rows) or Nemotron's ReLU^2 ones (fc1 = up, I rows)."""
    return experts.row13.shape[1] == 2 * experts.planes2.shape[2] * w3.CHUNK


def fc1_order(inter: int, device=None) -> torch.Tensor:
    """Row permutation of w13 = [up; gate] (2 * inter rows): block b of 32 rows is up 16b..16b+15, then gate 16b..16b+15."""
    b = torch.arange(inter // 16, device=device)[:, None]
    j = torch.arange(16, device=device)[None, :]
    return torch.cat([16 * b + j, inter + 16 * b + j], dim=1).reshape(-1)


# ------------------------------------------------------------------ calibration from the NVFP4 experts

def unswizzle_batched(swizzled: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """Inverse of moe.swizzle_block_scales_batched: [E, ...] CUTLASS-interleaved -> float8_e4m3fn [E, rows, groups]."""
    e = swizzled.shape[0]
    mp, kp = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    s = swizzled.reshape(e, -1).view(torch.float8_e4m3fn).reshape(e, mp // 128, kp // 4, 32, 4, 4)
    return s.permute(0, 1, 4, 3, 2, 5).reshape(e, mp, kp)[:, :rows, :groups]


def nvfp4_dequant(packed: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    """NVFP4 experts (uint8 [E, R, K/2] low nibble first, float8_e4m3fn [E, R, K/16], FP32 global scales [E]) -> float32 [E, R, K]."""
    from sharar.weights.modelopt import unpack_fp4
    e, r, _ = packed.shape
    w = unpack_fp4(packed.reshape(e * r, -1), scales.reshape(e * r, -1)).reshape(e, r, -1)
    return w * scale2.float().reshape(e, 1, 1)


@torch.no_grad()
def quantize(weight: torch.Tensor, codebook="lloyd", ratios=RATIOS):
    """float32 [N, K] -> (codes int32 [N, K], group scales float8_e4m3fn [N, K/32], row scales [N]): the group scale
    minimizing the weight error on the ratio grid (`w3_calibration.fit_scales`, unit importance), nearest level."""
    from sharar.weights.w3_calibration import fit_scales
    levels = w3._codebook(codebook)
    n, k = weight.shape
    scales, row_scale = fit_scales(weight, torch.ones(k, device=weight.device), levels, ratios)
    effective = (scales.float() * row_scale[:, None]).clamp_min(1e-30)
    lv = torch.tensor(levels, device=weight.device)
    codes = torch.bucketize(weight.reshape(n, k // w3.GROUP, w3.GROUP) / effective[:, :, None], (lv[1:] + lv[:-1]) / 2, out_int32=True)
    return codes.reshape(n, k), scales, row_scale


@torch.no_grad()
def calibrate(w13, s13, scale2_13, w2, s2, scale2_2, codebook="lloyd", ratios=RATIOS, chunk: int = 16):
    """A layer's NVFP4 experts (w13 [E, 2I, H/2] with rows [up; gate], unswizzled block scales, global scales [E];
    w2 likewise) -> (W3Experts, report with the relative weight error of fc1 and fc2 against the NVFP4 values)."""
    levels = w3._codebook(codebook)
    order = fc1_order(w13.shape[1] // 2, w13.device)
    parts, report = {}, {}
    for key, (packed, scales, scale2, perm) in {"w13": (w13, s13, scale2_13, order), "w2": (w2, s2, scale2_2, None)}.items():
        e, r = packed.shape[:2]
        out, num, den = [], 0.0, 0.0
        for lo in range(0, e, chunk):
            hi = min(e, lo + chunk)
            w = nvfp4_dequant(packed[lo:hi], scales[lo:hi], scale2[lo:hi])
            if perm is not None:
                w = w[:, perm]
            w = w.reshape((hi - lo) * r, -1)
            codes, sc, rs = quantize(w, levels, ratios)
            m = w3.pack_codes(codes, sc, rs, levels, rows_per_chunk=r)
            num += (w3.dequantize(m) - w).square().sum().item()
            den += w.square().sum().item()
            out.append(m)
            del w, codes, sc, rs
        k = packed.shape[2] * 2
        parts[key] = w3.W3Weight(*(torch.cat([getattr(m, a) for m in out]) for a in ("planes", "scales", "row_scale")),
                                 e * r, k, levels, w3.lut_of(levels))
        report[key] = (num / den) ** 0.5 if den else 0.0
    return W3Experts.from_stacked(parts["w13"], parts["w2"], w13.shape[0]), report


# ------------------------------------------------------------------ the kernel

def w3_moe(experts: W3Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, pairs: bool = False,
           pairs_bf16: bool = False) -> torch.Tensor:
    """Routed experts for x [M, 2560] bf16 with the router's ids [M, k] int32 and weights [M, k] fp32 (k <= 10) -> bf16
    [M, 2560], the weighted sum in selection order (no shared expert). Nemotron's experts (`gated` false): x is the
    latent [M, 1024], k <= 22, the result [M, 1024]. Graph-capturable and deterministic. When
    `experts` holds one rank's range (expert parallel), pairs routed to another rank's experts are skipped: the result
    is this rank's partial sum, which the decoder layer all-reduces. An id of EXPERTS (512) names no expert and adds
    nothing (an unfilled slot, `flashnext_fused.route_budget` without `repeat`). With `pairs`, the FP32 output of each
    (row, k) pair [M * k, 2560], unsummed (`flashnext_fused.moe_combine_pairs` sums them; rows of skipped pairs are
    garbage). `pairs_bf16`: those pairs in BF16, rounded once from the FP32 the kernel accumulates (prefill calls: the
    pairs are the call's widest tensor, written by fc2 and read back by the combine)."""
    ids = ids.to(torch.int32).contiguous()
    order = ids.new_empty(0)
    if ids.numel() > ORDER_IN_KERNEL:  # a stable sort keeps each expert's pairs in row order
        key = ids.reshape(-1)
        if experts.experts < EXPERTS:  # this rank's pairs first, by expert; the others after them
            local = key - experts.base
            key = torch.where((local >= 0) & (local < experts.experts), local, EXPERTS)
        order = torch.sort(key, stable=True).indices.to(torch.int32)
    return w3._module().w3_moe(x.contiguous(), ids, weights.float().contiguous(), order,
                               experts.planes13, experts.scales13, experts.row13, experts.planes2, experts.scales2, experts.row2,
                               experts.lut, experts.fc2_lut, experts.base, WIDE_ROWS, 2 if pairs and pairs_bf16 else int(pairs))


def quant_e4m3(t: torch.Tensor, stage: str) -> torch.Tensor:
    """The kernel's activation rounding, dequantized: E4M3 per (row, 32-group), float32 [M, K]. x goes through
    `w3.quant_groups` (the prep kernel runs the same code), the intermediate through its torch mirror."""
    if stage == "x":
        q, s = w3.quant_groups(t.contiguous())
        return (q.float().reshape(t.shape[0], -1, 32) * s[..., None]).reshape(t.shape)
    g = t.float().reshape(t.shape[0], -1, 32)
    s = (g.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return ((g / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).reshape(t.shape)


@torch.no_grad()
def reference(experts: W3Experts, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, quant=quant_e4m3) -> torch.Tensor:
    """The kernel's math in torch on the dequantized 3-bit weights (FP32 matmuls): x -> quant(x, "x"), fc1 -> up, gate
    in BF16, h = bf16(silu(gate) * up) (non-gated: bf16(relu(up)^2), FP32 square) -> quant(h, "h"), fc2 in FP32,
    weighted sum in selection order -> BF16."""
    import torch.nn.functional as F
    xd = quant(x, "x")
    gate = gated(experts)
    inter = experts.row13.shape[1] // 2 if gate else experts.row13.shape[1]
    inverse = torch.argsort(fc1_order(inter, x.device)) if gate else torch.arange(inter, device=x.device)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    out = torch.zeros((x.shape[0], experts.row2.shape[1]), dtype=torch.float32, device=x.device)
    for r in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[r, k])
            if e not in cache:
                cache[e] = (w3.dequantize(experts.matrix(e, True))[inverse], w3.dequantize(experts.matrix(e, False)))
            w13, w2 = cache[e]
            h1 = (xd[r:r + 1] @ w13.T).to(torch.bfloat16).float()
            h = (F.silu(h1[:, inter:]) * h1[:, :inter] if gate else torch.relu(h1).square()).to(torch.bfloat16)
            out[r] += float(weights[r, k]) * (quant(h, "h") @ w2.T)[0]
    return out.to(torch.bfloat16)
