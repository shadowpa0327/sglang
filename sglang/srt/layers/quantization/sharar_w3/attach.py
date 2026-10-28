"""Serve SGLang's decode and verify forwards from sharar's calibrated 3-bit (W3) copies (research port, lossy).

sharar (abdelfattah-lab/sharar, `docs/w3-calibration.md`) keeps the NVFP4/FP8 checkpoint for prefill and reads a
calibrated 3-bit copy of each trunk projection for decode-sized calls. This module loads that published checkpoint
(`SGLANG_SHARAR_W3=<dir>`: a sharar w3-checkpoint directory with `manifest.json`), shards it like the SGLang linear it
replaces, and wraps the linear's quant method: a forward in decode or target-verify mode (or any call of at most
`w3.MAX_ROWS` rows, sharar's own rule) runs `w3.w3_linear`, everything else the original weights.

Activations follow sharar: inputs of FP8 static-scale linears are quantized with the layer's input scale
(`clamp(x / s, +-448)` to E4M3, sharar `kernels/fp8_quant.py`) and the scale passed to the kernel; inputs of the
other linears (NVFP4 MLPs) are quantized per row and 32-group inside `w3_linear` (`quant_groups`).

Calls of more than `w3.MAX_ROWS` rows in decode / verify mode (sharar's `fast` profile never makes them) run the same
kernels on `MAX_ROWS`-row pieces.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

_ACTIVE = False
_COVERAGE = {"seen": set(), "total": 0}


def w3_active() -> bool:
    return _ACTIVE


_SLOTS = None  # the forward's request slots (req_pool_indices; 0 is the dummy slot graph padding uses)


def set_mode(forward_batch) -> None:
    """Called at the top of the target model's forward: decode and target-verify forwards read the 3-bit copies
    (and, with expert sharing, route under the budget)."""
    global _ACTIVE, _SLOTS
    mode = getattr(forward_batch, "forward_mode", None)
    _ACTIVE = mode is not None and (mode.is_decode() or mode.is_target_verify())
    _SLOTS = getattr(forward_batch, "req_pool_indices", None) if _ACTIVE else None


class Checkpoint:
    def __init__(self, directory: str | Path, revision: str | None = None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != 1:
            raise ValueError(f"{directory}: unsupported sharar W3 checkpoint format {self.manifest.get('format')}")
        if revision is not None and self.manifest.get("target_revision") != revision:
            raise ValueError(f"{directory}: calibrated for target revision {self.manifest.get('target_revision')}, "
                             f"serving {revision}")
        self.weights = self.manifest["weights"]

    def load(self, name: str):
        """[planes, scales, row_scale] of `name` on the CPU, checked against the manifest's shapes."""
        e = self.weights[name]
        t = load_file(str(self.directory / e["file"]))
        n, k = e["out_features"], e["in_features"]
        expect = {"planes": (n // 32, k // 128, 3, 32, 4), "scales": (n // 32, k // 128, 4, 32), "row_scale": (n,)}
        for key, shape in expect.items():
            if tuple(t[key].shape) != shape:
                raise ValueError(f"{name}.{key}: shape {tuple(t[key].shape)}, expected {shape}")
        return t, tuple(e["codebook"])


def _rows(t: dict, lo: int, hi: int) -> dict:
    """Output rows [lo, hi) (multiples of 32) of a packed matrix."""
    assert lo % 32 == 0 and hi % 32 == 0, (lo, hi)
    return {"planes": t["planes"][lo // 32:hi // 32], "scales": t["scales"][lo // 32:hi // 32],
            "row_scale": t["row_scale"][lo:hi]}


def _cols(t: dict, lo: int, hi: int) -> dict:
    """Input columns [lo, hi) (multiples of 128) of a packed matrix."""
    assert lo % 128 == 0 and hi % 128 == 0, (lo, hi)
    return {"planes": t["planes"][:, lo // 128:hi // 128], "scales": t["scales"][:, lo // 128:hi // 128],
            "row_scale": t["row_scale"]}


def _cat(parts: list[dict]) -> dict:
    return {key: torch.cat([p[key] for p in parts]) for key in ("planes", "scales", "row_scale")}


def build(ckpt: Checkpoint, names: list[str], linear, column: bool, device) -> "w3.W3Weight":
    """The W3 copy of `linear` (this rank's shard) from the checkpoint matrices `names`, stacked by rows."""
    from sglang.srt.layers.quantization.sharar_w3 import w3

    loaded = [ckpt.load(name) for name in names]
    codebooks = {cb for _, cb in loaded}
    if len(codebooks) != 1:
        raise ValueError(f"{names}: mixed codebooks {codebooks}")
    whole = _cat([t for t, _ in loaded])
    tp, rank = getattr(linear, "tp_size", 1), getattr(linear, "tp_rank", 0)
    if type(linear).__name__ == "ReplicatedLinear":
        tp, rank = 1, 0
    n_total = whole["row_scale"].shape[0]
    indices = getattr(linear, "shard_indices", None)
    if indices is not None:  # a vocab-parallel lm_head: this rank's vocabulary rows
        lo, hi = indices.org_vocab_start_index, indices.org_vocab_end_index
        if indices.num_elements_padded != hi - lo or getattr(indices, "added_vocab_end_index", 0) != getattr(indices, "added_vocab_start_index", 0):
            raise ValueError(f"{names}: padded or extended vocabulary shards are not supported")
        shard = _rows(whole, lo, hi)
    elif column:
        sizes = list(getattr(linear, "output_sizes", None) or [n_total])
        if sum(sizes) != n_total:
            raise ValueError(f"{names}: {n_total} rows, the linear's segments {sizes}")
        parts, off = [], 0
        for size in sizes:  # each merged segment is split across ranks on its own (MergedColumnParallelLinear)
            per = size // tp
            parts.append(_rows(whole, off + rank * per, off + (rank + 1) * per))
            off += size
        shard = _cat(parts)
    else:
        k = whole["planes"].shape[1] * 128
        per = k // tp
        if per % 128:  # e.g. Flash-Next's shared down (K 640) at TP2: the kernel's 128-k chunks cannot be split
            levels = codebooks.pop()
            full = w3.W3Weight(*(whole[key].contiguous().to(device) for key in ("planes", "scales", "row_scale")),
                               n_total, k, w3._codebook(levels), w3.lut_of(w3._codebook(levels)))
            return DenseW3(w3.dequantize(full)[:, rank * per:(rank + 1) * per].contiguous())
        shard = _cols(whole, rank * per, (rank + 1) * per)
    levels = codebooks.pop()
    planes, scales, row_scale = (shard[key].contiguous().to(device) for key in ("planes", "scales", "row_scale"))
    weight = w3.W3Weight(planes, scales, row_scale, row_scale.shape[0], planes.shape[1] * 128, w3._codebook(levels),
                         w3.lut_of(w3._codebook(levels)))
    return weight


_NOTED: set = set()


def _note(what: str) -> None:
    """Log each kind of call once (at graph capture for the graphed modes): evidence of which path a run served."""
    if what not in _NOTED:
        _NOTED.add(what)
        logger.info("sharar W3: %s", what)


@torch.no_grad()
def _check_linear(module) -> float:
    """Relative error of the 3-bit copy against the original weights on random activations (this rank's shard).
    Quantization noise alone gives about 0.2-0.3; a wrong name, row order or shard gives 1 or more."""
    method = module.quant_method
    k = method.w3.in_features
    x = torch.randn(16, k, device=method.w3.planes.device, dtype=torch.bfloat16)
    if method.static_scale is not None:  # inside the FP8 input range the layer was calibrated for
        x = x * float(method.static_scale) * 64
    ref = method.base.apply(module, x, None).float()
    got = method._w3(x).float()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _report(what: str, checks) -> None:
    errs = [e for _, e in checks]
    worst = sorted(checks, key=lambda c: -c[1])[:5]
    logger.info("sharar W3 check (%s, rank shard): %d linears, rel err mean %.3f max %.3f; worst %s", what, len(errs),
                sum(errs) / len(errs), max(errs), [(n, round(e, 3)) for n, e in worst])
    bad = [(n, e) for n, e in checks if not e < 0.6]
    if bad:
        raise ValueError(f"sharar W3 check failed for {len(bad)} {what} entries, e.g. {bad[:4]}")


class DenseW3:
    """A 3-bit copy kept dequantized (FP32), for shards the kernel cannot serve: sharar's arithmetic (E4M3 activations
    per 32-group, FP32 products) in a torch matmul. The weights are the same bits; only the FP32 summation order differs."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight  # [N, K] float32, row scale applied
        self.out_features, self.in_features = weight.shape
        self.planes, self.scales, self.row_scale = weight, weight.new_empty(0), weight.new_empty(0)  # for byte counts

    def __call__(self, x: torch.Tensor, x_scale: torch.Tensor | None = None) -> torch.Tensor:
        from sglang.srt.layers.quantization.sharar_w3 import w3
        if x.dtype == torch.float8_e4m3fn:
            xd = x.float() * x_scale.float().reshape(-1)[0]
        else:
            q, s = w3.quant_groups(x.contiguous())
            xd = (q.float().reshape(x.shape[0], -1, 32) * s.reshape(x.shape[0], -1, 1)).reshape(x.shape)
        return (xd @ self.weight.T).to(torch.bfloat16)


class W3Method:
    """Wraps a linear's quant method: decode / verify calls read the 3-bit copy."""

    def __init__(self, base, weight, static_scale: torch.Tensor | None, rows: int = 128):
        self.base = base
        self.w3 = weight
        self.rows = rows  # sharar's call-size rule: calls of at most this many rows read the copy in any mode
        self.static_scale = static_scale  # FP8 static-scale linears: the input scale (sharar `fp8_linear`)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _w3(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.layers.quantization.sharar_w3 import w3
        if isinstance(self.w3, DenseW3):
            if self.static_scale is not None:
                xq = (x.float() / self.static_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                return self.w3(xq, self.static_scale)
            return self.w3(x)
        if self.static_scale is not None:
            xq = (x.float() / self.static_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return w3.w3_linear(xq, self.w3, x_scale=self.static_scale)
        return w3.w3_linear(x.contiguous(), self.w3)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None, *args, **kwargs):
        from sglang.srt.layers.quantization.sharar_w3 import w3
        m = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1]
        if x.dtype != torch.bfloat16 and _ACTIVE:
            _note(f"input {x.dtype}: original weights")  # a prequantized input cannot take the 3-bit path
        if x.dtype == torch.bfloat16 and (_ACTIVE or m <= self.rows) and m > 0:
            if _ACTIVE:
                _note(f"3-bit path in decode/verify ({'fp8-static' if self.static_scale is not None else 'groups'})")
                if id(self) not in _COVERAGE["seen"]:
                    _COVERAGE["seen"].add(id(self))
                    n, total = len(_COVERAGE["seen"]), _COVERAGE["total"]
                    if n == total:
                        logger.info("sharar W3: decode coverage %d/%d (every 3-bit linear served a decode/verify call)", n, total)
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            if m <= w3.MAX_ROWS:
                y = self._w3(x2)
            else:
                y = torch.cat([self._w3(x2[i:i + w3.MAX_ROWS]) for i in range(0, m, w3.MAX_ROWS)])
            y = y.view(*lead, y.shape[-1])
            return y + bias if bias is not None else y
        return self.base.apply(layer, x, bias, *args, **kwargs)


# sharar's W3 call-size limit per target: the 27B's wide kernel serves up to 128 rows, Flash-Next's trunk the narrow
# kernel's 64 (model/flashnext/linear.py `_uses_w3`).
ROWS = {"qwen3_5": 128, "qwen4_exp": 64, "nemotron_h": 128}

# Per architecture: SGLang linear suffix -> (sharar checkpoint names for layer N, column-parallel?)
MAPS = {
    # Qwen3.8-27B (sharar `w3_checkpoint.model_linears`): MLP gate|up and down; GDN in_qkv + in_z (SGLang merges them
    # into in_proj_qkvz, segments q, k, v, z), out; attention q(gate)|k|v and o.
    "qwen3_5": {
        r"linear_attn\.in_proj_qkvz": (["layers.{i}.mixer.in_qkv", "layers.{i}.mixer.in_z"], True),
        r"linear_attn\.out_proj": (["layers.{i}.mixer.out"], False),
        r"(?:self_attn\.)?qkv_proj": (["layers.{i}.mixer.qkv"], True),
        r"(?:self_attn\.)?o_proj": (["layers.{i}.mixer.o"], False),
        r"mlp\.gate_up_proj": (["layers.{i}.mlp.gate_up"], True),
        r"mlp\.down_proj": (["layers.{i}.mlp.down"], False),
    },
    # Qwen3.8-Flash-Next (sharar `FlashNext.linears()`, the published checkpoint's trunk entries): GDN and attention as
    # the 27B, the shared expert, layer 1's PLE projections and the lm_head. The routed experts: `attach_moe`.
    "qwen4_exp": {
        r"linear_attn\.in_proj_qkvz": (["layers.{i}.mixer.in_qkv", "layers.{i}.mixer.in_z"], True),
        r"linear_attn\.out_proj": (["layers.{i}.mixer.out"], False),
        r"(?:self_attn\.)?qkv_proj": (["layers.{i}.mixer.qkv"], True),
        r"(?:self_attn\.)?o_proj": (["layers.{i}.mixer.o"], False),
        r"mlp\.shared_expert\.gate_up_proj": (["layers.{i}.mlp.shared_gate_up"], True),
        r"mlp\.shared_expert\.down_proj": (["layers.{i}.mlp.shared_down"], False),
        r"ple\.key_proj": (["layers.{i}.ple.key_proj"], True),
        r"ple\.value_proj": (["layers.{i}.ple.value_proj"], True),
    },
    # NVIDIA-Nemotron-3.5-Super (sharar `nemotron` trunk, 97 matrices): Mamba-2 in_proj (SGLang's merged z | x | B | C | dt,
    # split per segment) and out_proj, attention qkv and o, the lm_head. The latent projections, shared expert and dense
    # MLPs keep their served weights; the routed experts: `attach_moe`.
    "nemotron_h": {
        r"mixer\.in_proj": (["layers.{i}.mixer.in_proj"], True),
        r"mixer\.out_proj": (["layers.{i}.mixer.out_proj"], False),
        r"mixer\.qkv_proj": (["layers.{i}.mixer.qkv"], True),
        r"mixer\.o_proj": (["layers.{i}.mixer.o"], False),
    },
}
# Unlayered linears: SGLang module name -> checkpoint name.
GLOBAL = {"qwen4_exp": {"lm_head": "lm_head"}, "nemotron_h": {"language_model.lm_head": "lm_head"}}


def _static_scale(linear) -> torch.Tensor | None:
    """The input scale of an FP8 static-scale linear, None for anything else."""
    w = getattr(linear, "weight", None)
    s = getattr(linear, "input_scale", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fn:
        return None
    s = s.detach().float().reshape(-1)
    if s.numel() != 1:  # a merged linear's per-shard scales: SGLang requantizes to their max
        s = s.max().reshape(1)
    return s.contiguous()


def attach(model, arch: str, directory: str | None = None, revision: str | None = None) -> dict:
    """Attach the W3 copies to `model`'s covered linears; wrap its forward so the mode switch follows each batch."""
    from sglang.srt.layers.quantization.sharar_w3 import w3

    directory = directory or os.environ.get("SGLANG_SHARAR_W3")
    if not directory:
        return {}
    ckpt = Checkpoint(directory, revision)
    patterns = [(re.compile(r"(?:^|\.)layers\.(\d+)\." + suffix + r"$"), names, column)
                for suffix, (names, column) in MAPS[arch].items()]
    device = next(model.parameters()).device
    used, count, total, kinds = set(), 0, 0, {"fp8-static": 0, "groups": 0}
    checks = []
    for name, module in model.named_modules():
        if getattr(module, "_enable_silu_fp4_quant_fusion", False):
            module._enable_silu_fp4_quant_fusion = False  # down_proj must see BF16 activations (W3 quant_groups)
        if "visual" in name or "mtp" in name or not hasattr(module, "quant_method"):
            continue
        targets = [(None, [GLOBAL.get(arch, {})[name]], True)] if name in GLOBAL.get(arch, {}) else patterns
        for pattern, names, column in targets:
            hit = pattern.search(name) if pattern is not None else None
            if pattern is not None and hit is None:
                continue
            ck_names = [n.format(i=int(hit.group(1))) for n in names] if hit is not None else names
            weight = build(ckpt, ck_names, module, column, device)
            k_local = getattr(module, "input_size_per_partition", None)
            n_local = sum(getattr(module, "output_partition_sizes", None) or [getattr(module, "output_size_per_partition", weight.out_features)])
            if (k_local and k_local != weight.in_features) or n_local != weight.out_features:
                raise ValueError(f"{name}: W3 copy {weight.out_features}x{weight.in_features}, linear {n_local}x{k_local}")
            scale = _static_scale(module)
            kinds["fp8-static" if scale is not None else "groups"] += 1
            module.quant_method = W3Method(module.quant_method, weight, scale, ROWS[arch])
            if os.environ.get("SGLANG_SHARAR_W3_CHECK") == "1":
                checks.append((name, _check_linear(module)))
            used.update(ck_names)
            count += 1
            total += weight.planes.numel() * 4 + weight.scales.numel() + weight.row_scale.numel() * 4
            break
    if checks:
        _report("trunk", checks)
    unfused = _unfuse_gdn_in_proj(model)
    _COVERAGE["total"] = count
    missing = {n for n in ckpt.weights if ".experts." not in n} - used  # routed experts: `attach_moe`
    if missing:
        raise ValueError(f"{len(missing)} checkpoint matrices not attached, e.g. {sorted(missing)[:4]}")
    # the split-K workspace before graph capture: the largest partials buffer any call can ask for
    w3._workspace(device, 16 * w3.MAX_ROWS * 34816, 1 << 16)
    w3._module()
    torch.cuda.synchronize()
    info = {"matrices": count, "gib": total / 2**30, "kinds": kinds, "unfused_gdn_in_proj": unfused, "dir": str(directory),
            "target_revision": ckpt.manifest.get("target_revision")}
    logger.info("sharar W3: %s", info)
    return info


# ------------------------------------------------------------------ Flash-Next routed experts

SHARE_MIN = 8  # sharar model/nemotron/moe.py: under a binding cap each token keeps at least its top SHARE_MIN


def route_budget_sigmoid(logits: torch.Tensor, bias: torch.Tensor | None, k: int, keep: int, budget: int,
                         slots: torch.Tensor | None = None, steps: int = 1, per_request: int = 0, repeat: bool = True):
    """sharar's Nemotron budgeted routing (model/nemotron/moe.py `route_budget`) for sigmoid routers: logits [R, E] ->
    (int32 ids [R, k], fp32 weights [R, k], renormalized, without the routed scaling factor). A row ranks experts by
    sigmoid + bias (equal scores: the lower expert) and its weights are its plain sigmoid scores. The call's set is every
    real row's top `keep` (when that union exceeds the cap, each row's top `fallback` only: cap / rows within
    [SHARE_MIN, keep), from the call's shape), then the experts with the most routing weight over the real rows' top k
    (each row's top-k sigmoid scores normalized), up to the cap min(budget, per_request x real requests) (or `budget`
    without `per_request`). Each real row takes those of its top k in the set, in rank order; its other slots and a
    padded row's (slots[r // steps] == 0) name its top expert at weight 0 with `repeat`, else no expert (id E).
    Plain tensor ops at static shapes, no host sync: CUDA-graph capturable (speed is not the point; sharar's own is Triton)."""
    r, e = logits.shape
    dev = logits.device
    s = torch.sigmoid(logits.float())
    choice = s + bias.float() if bias is not None else s
    cand = torch.sort(-choice, dim=1, stable=True).indices[:, :k]  # rank order, ties to the lower expert
    p = s.gather(1, cand)
    if slots is None:
        real = torch.ones(r, dtype=torch.bool, device=dev)
    else:
        real = (slots != 0).repeat_interleave(steps)[:r]
    reqs = real.view(-1, steps)[:, 0].sum() if r % steps == 0 else real.sum()

    def union(top: int) -> torch.Tensor:  # the real rows' top `top`, as a mask over E (padded rows add nothing)
        ids = torch.where(real[:, None], cand[:, :top], e)
        return torch.zeros(e + 1, dtype=torch.bool, device=dev).scatter_(0, ids.reshape(-1), True)[:e]

    core = union(keep)
    cap = torch.full((), budget, dtype=torch.int64, device=dev)
    if per_request > 0:
        cap = torch.minimum(cap, per_request * reqs)
    cap_shape = min(budget, per_request * (r // steps)) if per_request else budget  # sharar's fallback rule: the call's shape
    g = max(min(SHARE_MIN, keep), min(keep, cap_shape // r)) if budget else keep
    if budget and g < keep:
        core = torch.where(core.sum() > cap, union(g), core)
    dense = torch.zeros(r, e, dtype=torch.float32, device=dev).scatter_(1, cand, p)
    dense = torch.where(real[:, None], dense / dense.sum(1, keepdim=True).clamp_min(1e-30), 0.0)
    score = dense.sum(0)
    key = torch.where(core | (score <= 0), -1.0, score)
    order = torch.sort(-key, stable=True).indices  # highest score first, ties to the lower expert
    rank = torch.empty_like(order).scatter_(0, order, torch.arange(e, device=dev))
    need = (cap - core.sum()).clamp_min(0) if budget else torch.zeros((), dtype=torch.int64, device=dev)
    chosen_set = core | ((key > 0) & (rank < need))
    pull = chosen_set[cand] & real[:, None]
    slot = torch.cumsum(pull.to(torch.int64), 1) - 1
    dest = torch.where(pull, slot, k)  # column k collects what is not kept
    w_kept = torch.where(pull, p, 0.0)
    w_kept = w_kept / (w_kept.sum(1, keepdim=True) + 1e-20)
    fill = cand[:, :1].expand(r, k) if repeat else torch.full((r, k), e, dtype=cand.dtype, device=dev)
    if repeat and slots is not None:  # a padded row: row 0's top expert, weight 0
        fill = torch.where(real[:, None], fill, cand[:1, :1].expand(r, k))
    ids = torch.cat([fill, fill[:, :1]], 1).scatter_(1, dest, cand)[:, :k]
    w = torch.zeros(r, k + 1, dtype=torch.float32, device=dev).scatter_(1, dest, w_kept)[:, :k]
    return ids.to(torch.int32), w


class BudgetTopK:
    """sharar's expert sharing (`SHARAR_MOE_KEEP` / `SHARAR_MOE_BUDGET`, model/flashnext/moe.py `route`): in decode and
    verify forwards a call reads at most a budget of distinct experts chosen for all its rows (`route.route_budget`);
    every row keeps its top `guarantee` experts. Other forwards use SGLang's TopK unchanged."""

    MIN_GUARANTEE = 2  # sharar moe.py
    MIN_CAP = 16

    def __init__(self, base, k: int, keep: int, per_request: int, budget: int, repeat: bool):
        self.base, self.k, self.keep, self.per_request, self.budget, self.repeat = base, k, keep, per_request, budget, repeat

    def __getattr__(self, name):
        return getattr(self.base, name)

    def guarantee(self, rows: int, steps: int) -> int:
        cap = min(self.budget, max(self.MIN_CAP, self.per_request * (rows // steps))) if self.per_request else self.budget
        return max(min(self.MIN_GUARANTEE, self.keep), min(self.keep, 2 * cap // rows))

    def __call__(self, hidden_states, router_logits, *args, **kwargs):
        if not _ACTIVE or _SLOTS is None or router_logits.shape[0] == 0:
            return self.base(hidden_states, router_logits, *args, **kwargs)
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.quantization.sharar_w3.route import route_budget
        rows, reqs = router_logits.shape[0], _SLOTS.numel()
        if rows % reqs:
            raise ValueError(f"expert sharing: {rows} rows over {reqs} requests")
        steps = rows // reqs
        _note(f"expert sharing in decode/verify (rows {rows}, steps {steps})")
        cfg = self.base.topk_config
        if cfg.scoring_func == "sigmoid":  # Nemotron 3.5: sigmoid + bias choice, plain-sigmoid weights (sharar's rule)
            if cfg.use_grouped_topk and (cfg.num_expert_group or 1) > 1:
                raise ValueError("expert sharing: grouped sigmoid routing with more than one group is not ported")
            ids, w = route_budget_sigmoid(router_logits, cfg.correction_bias, self.k, self.keep, self.budget,
                                          _SLOTS, steps, per_request=self.per_request, repeat=self.repeat)
            if cfg.apply_routed_scaling_factor_on_output and cfg.routed_scaling_factor:
                w = w * cfg.routed_scaling_factor  # the runner expects routing to have applied it
            return StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=router_logits)
        ids, w = route_budget(router_logits.to(torch.bfloat16).contiguous(), self.k, self.guarantee(rows, steps), self.budget,
                              _SLOTS.to(torch.int32), steps, repeat=self.repeat, per_request=self.per_request,
                              min_cap=self.MIN_CAP)
        return StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=router_logits)


class W3ExpertsForward(torch.nn.Module):
    """Replaces a FusedMoE's routed computation with sharar's 3-bit experts (`w3_moe`) at every call size, as sharar
    serves them (the NVFP4 experts are then released). This rank holds a contiguous range of whole experts (sharar's
    expert-parallel split) and returns their partial sum, which the block's all-reduce adds like FusedMoE's TP partials."""

    def __init__(self, base, experts, scale: float | None = None):
        super().__init__()
        self.base = base
        self.w3 = experts
        self.scale = scale  # the routed scaling factor when routing left it to the experts (Nemotron: 5.0), else None

    def forward(self, hidden_states, topk_output, *args, **kwargs):
        from sglang.srt.layers.moe.topk import TopKOutputFormat
        from sglang.srt.layers.quantization.sharar_w3.w3_moe import w3_moe
        if topk_output.format == TopKOutputFormat.BYPASSED:  # SM100's flashinfer_trtllm runner routes in its kernel
            topk_output = topk_output.to_standard()
        ids, w = topk_output.topk_ids, topk_output.topk_weights
        if hidden_states.shape[0] == 0:
            return hidden_states.clone()
        _note("3-bit routed experts")
        out = w3_moe(self.w3, hidden_states, ids, w)
        if self.scale is not None:
            out = out * self.scale
        if getattr(self.base, "reduce_results", False):
            from sglang.srt.distributed import tensor_model_parallel_all_reduce
            out = tensor_model_parallel_all_reduce(out)
        return out


def _swap_child(parent, name: str, value) -> None:
    """Replace submodule `name` of `parent` with `value` (a module or a plain proxy object)."""
    if isinstance(value, torch.nn.Module):
        setattr(parent, name, value)
    else:
        parent._modules.pop(name, None)
        object.__setattr__(parent, name, value)


@torch.no_grad()
def _check_experts(ckpt, layer, fused, tp, rank, device, top_k: int = 10, scale: float | None = None) -> float:
    """The 3-bit experts against the NVFP4 FusedMoE on random rows and routing (this rank's partial sum)."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.sharar_w3.w3_moe import w3_moe
    g = torch.Generator(device=device).manual_seed(1234 + layer)  # the same rows and routing on every rank
    hidden, n_exp = getattr(fused, "hidden_size", 2560), getattr(fused, "num_experts", 512)
    x = torch.randn(16, hidden, device=device, dtype=torch.bfloat16, generator=g)
    ids = torch.stack([torch.randperm(n_exp, device=device, generator=g)[:top_k] for _ in range(16)]).int()
    w = torch.softmax(torch.randn(16, top_k, device=device, generator=g), -1)
    ref = fused(x, StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=torch.zeros(16, n_exp, device=device)))
    ref = ref if getattr(fused, "reduce_results", False) else tensor_model_parallel_all_reduce(ref.clone())
    experts = load_experts(ckpt, layer, n_exp, tp, rank, device)
    got = w3_moe(experts, x, ids, w)
    got = tensor_model_parallel_all_reduce(got * scale if scale is not None else got)
    del experts
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def load_experts(ckpt: Checkpoint, layer: int, experts: int, tp: int, rank: int, device):
    """Layer `layer`'s 3-bit experts for this rank: whole experts lo .. hi - 1 (sharar `expert_range`)."""
    from sglang.srt.layers.quantization.sharar_w3 import w3
    from sglang.srt.layers.quantization.sharar_w3.w3_moe import W3Experts
    lo, hi = experts * rank // tp, experts * (rank + 1) // tp
    mats = []
    fc1 = f"layers.{layer}.experts.w13" if f"layers.{layer}.experts.w13" in ckpt.weights else f"layers.{layer}.experts.w1"
    for name in (fc1, f"layers.{layer}.experts.w2"):  # fc1: Flash-Next's gated [up; gate] or Nemotron's up
        t, cb = ckpt.load(name)
        per = t["row_scale"].shape[0] // experts
        part = _rows(t, lo * per, hi * per)
        levels = w3._codebook(cb)
        mats.append(w3.W3Weight(*(part[k].contiguous().to(device) for k in ("planes", "scales", "row_scale")),
                                (hi - lo) * per, t["planes"].shape[1] * 128, levels, w3.lut_of(levels)))
    return W3Experts.from_stacked(mats[0], mats[1], hi - lo, base=lo)


def attach_moe(model, directory: str | None, sharing: tuple[int, int, int] | None, tp: int, rank: int) -> dict:
    """Expert sharing (`sharing` = (keep, per_request, budget), sharar fast: (4, 12, 64)) and/or the checkpoint's 3-bit
    routed experts (`directory` with layers.N.experts.*) on every sparse MoE block of `model`."""
    ckpt = Checkpoint(directory) if directory else None
    device = next(model.parameters()).device
    blocks = 0
    echecks = []
    for name, block in model.named_modules():
        if "mtp" in name or not (hasattr(block, "topk") and hasattr(block, "experts")):
            continue
        hit = re.search(r"layers\.(\d+)\.", name + ".")
        if hit is None:
            continue
        layer = int(hit.group(1))
        if ckpt is not None:
            if getattr(block, "enable_shared_expert_fusion", False):
                raise ValueError("3-bit experts need the shared expert outside the routed experts "
                                 "(--disable-shared-experts-fusion)")
            fused = block.experts
            cfg = block.topk.topk_config
            factor = getattr(cfg, "routed_scaling_factor", None)
            scale = factor if factor not in (None, 1.0) and not getattr(cfg, "apply_routed_scaling_factor_on_output", False) else None
            if os.environ.get("SGLANG_SHARAR_W3_CHECK") == "1":
                echecks.append((name, _check_experts(ckpt, layer, fused, tp, rank, device, cfg.top_k, scale)))
            for p in fused.parameters(recurse=True):  # the NVFP4 experts are not read again: release them first
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            torch.cuda.empty_cache()
            experts = load_experts(ckpt, layer, getattr(fused, "num_experts", 512), tp, rank, device)
            _swap_child(block, "experts", W3ExpertsForward(fused, experts, scale))
        if sharing is not None:
            keep, per_request, budget = sharing
            k = block.topk.topk_config.top_k
            _swap_child(block, "topk", BudgetTopK(block.topk, k, keep, per_request, budget, repeat=ckpt is None))
        blocks += 1
    torch.cuda.empty_cache()
    if echecks:
        _report("routed experts", echecks)
    info = {"moe_blocks": blocks, "w3_experts": ckpt is not None, "sharing": sharing}
    logger.info("sharar MoE: %s", info)
    return info


# ------------------------------------------------------------------ the MTP draft's own head (SGLANG_SHARAR_DRAFT_HEAD)

DRAFT_RESCORE = {"w3": 8, "w2": 16}  # sharar proposers/draft_head.py RESCORE: the 2-bit top 8 miss the FP8 argmax 0.6% of steps, the top 16 0.2%


class _ShortlistMethod:
    """sharar's `ShortlistHead` (proposers/draft_head.py) as an lm_head quant method for the MTP draft steps: a 2-bit
    (`w2`: four Lloyd levels, fitted group scales) or 3-bit (`w3`) copy of the target's FP8 head picks `rescore`
    candidates, the head's own FP8 rows score them, and the logits are -inf off the shortlist, so the draft's argmax is
    sharar's pick. Draft probabilities are not used (the chain is verified with zero draft probs), and under exact
    acceptance a draft head changes how many drafts are accepted, never what is emitted; under relaxed acceptance it
    changes which plausible tokens are drafted. Every row of the head (TP1: the full vocabulary)."""

    def __init__(self, head, fmt: str, rescore: int):
        from sglang.srt.layers.quantization.sharar_w3 import w3
        weight = head.weight
        scale = getattr(head, "weight_scale", None)
        n = head.num_embeddings_per_partition  # the head's rows (vocabulary, padded)
        if weight.shape[0] != n:  # FP8 linear methods may keep the weight transposed, [K, N], after loading
            if weight.shape[1] != n:
                raise ValueError(f"draft head: weight {tuple(weight.shape)} does not hold {n} vocabulary rows")
            weight = weight.T  # a [N, K] view; rows are gathered from it, never copied whole
        k = weight.shape[1]
        codebook, fit = ("lloyd2", True) if fmt == "w2" else (None, False)
        chunk = 16384
        def read(r0, r1):
            rows = weight[r0:r1].float()
            return rows * scale.float().reshape(-1)[0] if scale is not None else rows
        parts = [w3.pack(read(r0, min(r0 + chunk, n)), codebook, fit_scales=fit) for r0 in range(0, n, chunk)]
        first = parts[0]
        self.copy = first if len(parts) == 1 else w3.W3Weight(
            torch.cat([q.planes for q in parts]), torch.cat([q.scales for q in parts]),
            torch.cat([q.row_scale for q in parts]), n, k, first.codebook, first.lut)
        self.weight, self.scale, self.rescore, self.fmt = weight, scale, rescore, fmt
        self.checked, self.agree = 0, 0
        self.check = os.environ.get("SGLANG_SHARAR_W3_CHECK") == "1"

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None, *args, **kwargs):
        from sglang.srt.layers.quantization.sharar_w3 import w3
        flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
        if flat.shape[0] == 0:
            return torch.empty((0, self.weight.shape[0]), dtype=torch.float32, device=x.device)
        _note(f"MTP draft head: {self.fmt} shortlist of {self.rescore}, rescored by the FP8 rows")
        low = torch.cat([w3.w3_linear(part.contiguous(), self.copy) for part in flat.split(w3.MAX_ROWS)])
        cand = low.topk(self.rescore, dim=-1).indices  # [M, r]
        rows = self.weight[cand].float()  # [M, r, K] of the target's FP8 head
        exact = torch.bmm(rows, flat.float().unsqueeze(-1)).squeeze(-1)
        if self.scale is not None:
            exact = exact * self.scale.float().reshape(-1)[0]
        out = torch.full((flat.shape[0], self.weight.shape[0]), float("-inf"), dtype=torch.float32, device=x.device)
        out.scatter_(1, cand, exact)
        # the FP8 head's own argmax on calls run outside a CUDA graph (most draft steps replay one), for the report:
        # every doubling of the checked rows from 256 up to 65536
        if self.check and self.checked < 65536 and not torch.cuda.is_current_stream_capturing():
            full = flat.float() @ self.weight.float().T
            self.agree += int((out.argmax(-1) == full.argmax(-1)).sum())
            before, self.checked = self.checked, self.checked + flat.shape[0]
            if self.checked >= 256 and (before < 256 or self.checked.bit_length() > before.bit_length()):
                logger.info("sharar draft head check: %s shortlist of %d picks the FP8 argmax in %.2f%% of %d draft rows",
                            self.fmt, self.rescore, 100 * self.agree / self.checked, self.checked)
        return out


class DraftShortlistHead(torch.nn.Module):
    """The MTP draft's lm_head: the target's head module (shared, unchanged: the target keeps its own copy and modes)
    with `_ShortlistMethod` as its quant method. Every other attribute is the target head's."""

    def __init__(self, base, fmt: str, rescore: int):
        super().__init__()
        object.__setattr__(self, "_base", base)  # not a registered child: no duplicate parameters
        self.quant_method = _ShortlistMethod(base, fmt, rescore)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_base"), name)


def install_draft_head(model) -> None:
    """SGLANG_SHARAR_DRAFT_HEAD=w2|w3 (sharar's SHARAR_DRAFT_HEAD; fp8 or unset: the target's head as it is): give the
    MTP draft model `model` its own shortlist head over the target head it was handed (`model.lm_head`).
    SGLANG_SHARAR_DRAFT_HEAD_RESCORE overrides the shortlist size (default 16 for w2, 8 for w3)."""
    fmt = os.environ.get("SGLANG_SHARAR_DRAFT_HEAD", "fp8")
    if fmt == "fp8" or isinstance(model.lm_head, DraftShortlistHead):
        return
    if fmt not in DRAFT_RESCORE:
        raise ValueError(f"SGLANG_SHARAR_DRAFT_HEAD must be fp8, w3 or w2, not {fmt!r}")
    head = model.lm_head
    from sglang.srt.runtime_context import get_parallel
    if get_parallel().tp_size > 1:
        raise ValueError("SGLANG_SHARAR_DRAFT_HEAD: a vocabulary-parallel head (TP > 1) is not ported")
    rescore = int(os.environ.get("SGLANG_SHARAR_DRAFT_HEAD_RESCORE", DRAFT_RESCORE[fmt]))
    model.lm_head = DraftShortlistHead(head, fmt, rescore)
    torch.cuda.empty_cache()
    m = model.lm_head.quant_method
    logger.info("sharar draft head: %s copy of the %s head [%d, %d] (stored %s), shortlist %d (%.3f GiB)", fmt,
                head.weight.dtype, m.weight.shape[0], m.weight.shape[1], tuple(head.weight.shape), rescore, m.copy.bytes / 2**30)


# ------------------------------------------------------------------ calibration capture (re-fitting the 3-bit copies)

class _Capture:
    """Activations for GPTQ, in sharar's formats: x^T x of every covered trunk linear's full input (row-parallel inputs
    all-gathered over TP; rank 0 accumulates in FP32) and every MoE block's input rows (rank 0, bf16 `layerNN.bin`, as
    sharar `flashnext_w3_experts capture` writes them). Saved when `<dir>/SAVE` appears."""

    def __init__(self, directory: str, rank: int):
        self.dir, self.rank = Path(directory), rank
        self.dir.mkdir(parents=True, exist_ok=True)
        self.H, self.rows, self.names = {}, {}, {}
        self.files, self.moe_rows = {}, {}
        self.saved = False

    def linear(self, key: str, x: torch.Tensor, row_parallel: bool) -> None:
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 0 or self.saved:
            return
        if row_parallel:
            flat = tensor_model_parallel_all_gather(flat.contiguous(), dim=-1)
        if self.rank != 0:
            return
        f = flat.float()
        if key not in self.H:
            self.H[key] = torch.zeros((f.shape[1], f.shape[1]), dtype=torch.float32, device=f.device)
            self.rows[key] = 0
        self.H[key].addmm_(f.T, f)
        self.rows[key] += f.shape[0]

    def moe(self, layer: int, x: torch.Tensor) -> None:
        if self.rank != 0 or x.shape[0] == 0 or self.saved:
            return
        if layer not in self.files:
            self.files[layer] = open(self.dir / f"layer{layer:02d}.bin", "wb")
            self.moe_rows[layer] = 0
        flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
        self.files[layer].write(flat.view(torch.int16).cpu().numpy().tobytes())
        self.moe_rows[layer] += flat.shape[0]

    def poll(self) -> None:
        if self.saved or not (self.dir / "SAVE").exists():
            return
        self.saved = True
        if self.rank != 0:
            return
        for f in self.files.values():
            f.close()
        for key, h in self.H.items():
            torch.save({"H": h.cpu(), "rows": self.rows[key], "names": self.names[key]}, self.dir / f"cov_{key}.pt")
        rows = set(self.moe_rows.values())
        meta = {"rows": rows.pop() if len(rows) == 1 else sorted(rows), "hidden": 2560, "dtype": "bfloat16",
                "layers": sorted(self.moe_rows), "trunk": "bf16 (SGLang exact)", "experts": "nvfp4 (Marlin W4A16)",
                "linears": {k: self.rows[k] for k in self.H}}
        (self.dir / "capture.json").write_text(json.dumps(meta, indent=1))
        (self.dir / "SAVED").write_text("ok\n")
        logger.info("sharar capture saved: %d covariances, MoE rows %s", len(self.H), meta["rows"])


_CAPTURE: "_Capture | None" = None


class CaptureMethod:
    """Records a linear's input for the capture, then runs the original method."""

    def __init__(self, base, key: str, row_parallel: bool):
        self.base, self.key, self.row_parallel = base, key, row_parallel

    def __getattr__(self, name):
        return getattr(self.base, name)

    def apply(self, layer, x, bias=None, *args, **kwargs):
        if _CAPTURE is not None:
            _CAPTURE.linear(self.key, x, self.row_parallel)
        return self.base.apply(layer, x, bias, *args, **kwargs)


def capture_poll() -> None:
    if _CAPTURE is not None:
        _CAPTURE.poll()


def attach_capture(model, arch: str, directory: str, rank: int) -> dict:
    """Wrap every linear the arch's W3 map covers and every sparse MoE block for the capture."""
    global _CAPTURE
    _CAPTURE = cap = _Capture(directory, rank)
    patterns = [(re.compile(r"(?:^|\.)layers\.(\d+)\." + suffix + r"$"), names, column)
                for suffix, (names, column) in MAPS[arch].items()]
    n_lin = n_moe = 0
    for name, module in list(model.named_modules()):
        if "visual" in name or "mtp" in name:
            continue
        if hasattr(module, "topk") and hasattr(module, "experts"):
            hit = re.search(r"layers\.(\d+)\.", name + ".")
            if hit is not None:
                layer = int(hit.group(1))
                module.register_forward_pre_hook(lambda m, args, kwargs, layer=layer: cap.moe(layer, args[0] if args else kwargs["hidden_states"]),
                                                 with_kwargs=True)
                n_moe += 1
        if not hasattr(module, "quant_method"):
            continue
        targets = [(None, [GLOBAL.get(arch, {})[name]], True)] if name in GLOBAL.get(arch, {}) else patterns
        for pattern, names, column in targets:
            hit = pattern.search(name) if pattern is not None else None
            if pattern is not None and hit is None:
                continue
            ck_names = [n.format(i=int(hit.group(1))) for n in names] if hit is not None else names
            key = ck_names[0].replace(".", "_")
            cap.names[key] = ck_names  # every checkpoint matrix that takes this input
            module.quant_method = CaptureMethod(module.quant_method, key, row_parallel=not column)
            n_lin += 1
            break
    info = {"capture_linears": n_lin, "capture_moe_blocks": n_moe, "unfused_gdn_in_proj": _unfuse_gdn_in_proj(model), "dir": directory}
    logger.info("sharar capture: %s", info)
    return info


def _unfuse_gdn_in_proj(model) -> int:
    """SGLang fuses a BF16 GDN layer's in_proj_qkvz and in_proj_ba into one GEMM for calls of up to 1,024 rows
    (qwen3_5.py `_fused_in_proj_weight`), which never calls in_proj_qkvz's quant method: drop the fused weight where the
    port wraps it, so decode, verify and capture go through the wrapper (the two weights stay as they are)."""
    n = 0
    for m in model.modules():
        qkvz = getattr(m, "in_proj_qkvz", None)
        if getattr(m, "_fused_in_proj_weight", None) is not None and isinstance(getattr(qkvz, "quant_method", None), (W3Method, CaptureMethod)):
            m._fused_in_proj_weight = None
            m._fused_in_proj_qkvz_width = 0
            n += 1
    return n
