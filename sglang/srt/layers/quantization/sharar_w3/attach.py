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
ROWS = {"qwen3_5": 128, "qwen4_exp": 64}

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
}
# Unlayered linears: SGLang module name -> checkpoint name.
GLOBAL = {"qwen4_exp": {"lm_head": "lm_head"}}


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
    missing = {n for n in ckpt.weights if ".experts." not in n} - used  # routed experts: `attach_moe`
    if missing:
        raise ValueError(f"{len(missing)} checkpoint matrices not attached, e.g. {sorted(missing)[:4]}")
    # the split-K workspace before graph capture: the largest partials buffer any call can ask for
    w3._workspace(device, 16 * w3.MAX_ROWS * 34816, 1 << 16)
    w3._module()
    torch.cuda.synchronize()
    info = {"matrices": count, "gib": total / 2**30, "kinds": kinds, "dir": str(directory),
            "target_revision": ckpt.manifest.get("target_revision")}
    logger.info("sharar W3: %s", info)
    return info


# ------------------------------------------------------------------ Flash-Next routed experts

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
        ids, w = route_budget(router_logits.to(torch.bfloat16).contiguous(), self.k, self.guarantee(rows, steps), self.budget,
                              _SLOTS.to(torch.int32), steps, repeat=self.repeat, per_request=self.per_request,
                              min_cap=self.MIN_CAP)
        return StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=router_logits)


class W3ExpertsForward(torch.nn.Module):
    """Replaces a FusedMoE's routed computation with sharar's 3-bit experts (`w3_moe`) at every call size, as sharar
    serves them (the NVFP4 experts are then released). This rank holds a contiguous range of whole experts (sharar's
    expert-parallel split) and returns their partial sum, which the block's all-reduce adds like FusedMoE's TP partials."""

    def __init__(self, base, experts):
        super().__init__()
        self.base = base
        self.w3 = experts

    def forward(self, hidden_states, topk_output, *args, **kwargs):
        from sglang.srt.layers.quantization.sharar_w3.w3_moe import w3_moe
        ids, w = topk_output.topk_ids, topk_output.topk_weights
        if hidden_states.shape[0] == 0:
            return hidden_states.clone()
        _note("3-bit routed experts")
        out = w3_moe(self.w3, hidden_states, ids, w)
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
def _check_experts(ckpt, layer, fused, tp, rank, device) -> float:
    """The 3-bit experts against the NVFP4 FusedMoE on random rows and routing (this rank's partial sum)."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.sharar_w3.w3_moe import w3_moe
    g = torch.Generator(device=device).manual_seed(1234 + layer)  # the same rows and routing on every rank
    x = torch.randn(16, 2560, device=device, dtype=torch.bfloat16, generator=g)
    ids = torch.stack([torch.randperm(512, device=device, generator=g)[:10] for _ in range(16)]).int()
    w = torch.softmax(torch.randn(16, 10, device=device, generator=g), -1)
    ref = fused(x, StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=torch.zeros(16, 512, device=device)))
    ref = ref if getattr(fused, "reduce_results", False) else tensor_model_parallel_all_reduce(ref.clone())
    experts = load_experts(ckpt, layer, getattr(fused, "num_experts", 512), tp, rank, device)
    got = tensor_model_parallel_all_reduce(w3_moe(experts, x, ids, w))
    del experts
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def load_experts(ckpt: Checkpoint, layer: int, experts: int, tp: int, rank: int, device):
    """Layer `layer`'s 3-bit experts for this rank: whole experts lo .. hi - 1 (sharar `expert_range`)."""
    from sglang.srt.layers.quantization.sharar_w3 import w3
    from sglang.srt.layers.quantization.sharar_w3.w3_moe import W3Experts
    lo, hi = experts * rank // tp, experts * (rank + 1) // tp
    mats = []
    for name in (f"layers.{layer}.experts.w13", f"layers.{layer}.experts.w2"):
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
            if os.environ.get("SGLANG_SHARAR_W3_CHECK") == "1":
                echecks.append((name, _check_experts(ckpt, layer, fused, tp, rank, device)))
            for p in fused.parameters(recurse=True):  # the NVFP4 experts are not read again: release them first
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            torch.cuda.empty_cache()
            experts = load_experts(ckpt, layer, getattr(fused, "num_experts", 512), tp, rank, device)
            _swap_child(block, "experts", W3ExpertsForward(fused, experts))
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
