"""Pre-RoPE key space by inverting the model's own rotary table.

RoPE at absolute position p is an orthogonal rotation R(p) of each channel
pair. Inactive keys are handed to the codec as R(-p) K, computed in FP32 from
the FP32 cos/sin table of the model's RotaryEmbedding, and rotated back with
R(+p) during reconstruction. The table already encodes neox versus interleaved
pairing, partial rotary dimensions and every scaling variant (Llama3, YaRN,
dynamic NTK), so the inverse inherits all of them. There is no model forward
hook, no capture buffer and no CUDA graph restriction.

The stored key is not the projection output bit for bit: it carries the cache
dtype rounding of the native rotation (about 2^-9 relative for BF16), rotated
into the pre-RoPE frame. Under audit the round trip R(p) R(-p) K is checked
against the native key with a bound of two ulps of the largest channel.
"""

import torch

from kvcompress.api import CompressedPayload, KVCompressionPlugin

KEY_SPACES = ("pre_rope", "post_rope")


def _fused_rope_kernel():
    """The kernel RotaryEmbedding.forward_cuda uses; None without CUDA support."""
    try:
        from sglang.kernels.ops.attention.rope import (
            apply_rope_with_cos_sin_cache_inplace,
        )
    except ImportError:
        return None
    return apply_rope_with_cos_sin_cache_inplace


def resolve_key_space(configured, declared):
    """Pick the key space from config (auto/None, pre_rope, post_rope) and the plugin.

    Returns (key_space, source). An explicit config value wins over the plugin's
    declaration and is reported as "config_override" so the log makes it visible.
    """
    if declared is not None and declared not in KEY_SPACES:
        raise ValueError(
            f"Plugin key_space must be one of {KEY_SPACES}, got {declared!r}"
        )
    if configured in (None, "auto"):
        if declared is None:
            return "post_rope", "default"
        return declared, "plugin"
    if configured not in KEY_SPACES:
        raise ValueError("key_space must be auto, pre_rope or post_rope")
    if declared is not None and declared != configured:
        return configured, "config_override"
    return configured, "config"


class RotaryTable:
    """FP32 cos/sin rows plus the pairing layout of one RotaryEmbedding."""

    def __init__(self, rope, device):
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        # MRotaryEmbedding is accepted: text-only requests carry 1-D positions,
        # for which the model applies the base cos/sin table unchanged.
        if not isinstance(rope, RotaryEmbedding):
            raise ValueError(
                f"Pre-RoPE inversion needs a RotaryEmbedding table, got {type(rope).__name__}"
            )
        cache = rope.cos_sin_cache
        if cache.dtype != torch.float32:
            # Recompute with the subclass formula instead of up-casting a rounded table.
            cache = rope._compute_cos_sin_cache()
        self.head_size = int(rope.head_size)
        self.rotary_dim = int(rope.rotary_dim)
        self.is_neox_style = bool(rope.is_neox_style)
        if (
            self.rotary_dim % 2
            or self.rotary_dim > self.head_size
            or cache.ndim != 2
            or cache.shape[-1] != self.rotary_dim
        ):
            raise ValueError("Unexpected rotary table layout")
        # [max_pos, rotary_dim] = [cos | sin]; the conjugate table encodes R(-p).
        self.forward_table = (
            cache.detach().to(device=device, dtype=torch.float32).contiguous()
        )
        self.cos, self.sin = self.forward_table.chunk(2, dim=-1)
        self.inverse_table = torch.cat([self.cos, -self.sin], dim=-1).contiguous()
        self.fused = _fused_rope_kernel() if device.type == "cuda" else None
        self.description = {
            "class": type(rope).__name__,
            "head_size": self.head_size,
            "rotary_dim": self.rotary_dim,
            "is_neox_style": self.is_neox_style,
            "positions": int(self.cos.shape[0]),
            "table_dtype": "float32",
            "kernel": "fused-cuda" if self.fused is not None else "torch-fp32",
        }

    def check_positions(self, positions):
        if positions.numel():
            low, high = int(positions.min()), int(positions.max())
            if low < 0 or high >= self.cos.shape[0]:
                raise ValueError(
                    f"RoPE table with {self.cos.shape[0]} positions does not cover "
                    f"[{low}, {high}]"
                )

    def rows(self, positions):
        """[tokens, 1, rotary_dim / 2] cos and sin rows for absolute positions."""
        self.check_positions(positions)
        positions = positions.to(self.cos.device)
        return self.cos[positions].unsqueeze(1), self.sin[positions].unsqueeze(1)

    def rotate_fused_(self, key, positions, *, inverse, query):
        """In-place R(+-p) on cache-dtype key [tokens, heads, head_size].

        Uses the model kernel; query is a same-dtype [tokens, 1, head_size] zero
        buffer the kernel rotates alongside.
        """
        self.fused(
            positions=positions,
            q=query[..., : self.rotary_dim],
            k=key[..., : self.rotary_dim],
            cos_sin_cache=self.inverse_table if inverse else self.forward_table,
            is_neox=self.is_neox_style,
        )

    def rotate(self, x, cos, sin, *, inverse, out, scratch):
        """out = R(+p) x, or R(-p) x when inverse.

        FP32 [tokens, heads, head_size]; writes only out and scratch.
        """
        half = self.rotary_dim // 2
        xr, orr = x[..., : self.rotary_dim], out[..., : self.rotary_dim]
        if self.is_neox_style:
            x1, x2 = xr[..., :half], xr[..., half:]
            o1, o2 = orr[..., :half], orr[..., half:]
        else:
            x1, x2 = xr[..., ::2], xr[..., 1::2]
            o1, o2 = orr[..., ::2], orr[..., 1::2]
        # Forward: o1 = x1 cos - x2 sin, o2 = x2 cos + x1 sin. Inverse flips sin.
        torch.mul(x1, cos, out=o1)
        torch.mul(x2, sin, out=scratch)
        (o1.add_ if inverse else o1.sub_)(scratch)
        torch.mul(x2, cos, out=o2)
        torch.mul(x1, sin, out=scratch)
        (o2.sub_ if inverse else o2.add_)(scratch)
        if self.rotary_dim < self.head_size:
            out[..., self.rotary_dim :].copy_(x[..., self.rotary_dim :])


class PreRoPETransform:
    """Adapter-owned key-space transform between native post-RoPE and pre-RoPE keys.

    Tensors have the store layout [pages, layers, page_tokens, heads, head_size].
    Positions are absolute: token (g, t) sits at start_position + g * page + t.
    """

    def __init__(self, table: RotaryTable, *, layers: int, page_size: int, record=None):
        self.table = table
        self.layers = layers
        self.page_size = page_size
        self.record = record

    @classmethod
    def from_model_ropes(cls, adapter, record=None):
        """Use the process-wide RoPE cache populated while the model was built."""
        from sglang.srt.layers.rotary_embedding import factory
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        head_dim = int(adapter.kv.head_dim)
        candidates = []
        for key in list(factory._ROPE_DICT):
            rope = factory._get_live_rope_cache_entry(key)
            if isinstance(rope, RotaryEmbedding) and rope.head_size == head_dim:
                candidates.append(rope)
        if len(candidates) != 1:
            raise ValueError(
                "Pre-RoPE inversion needs exactly one live RotaryEmbedding with "
                f"head_size={head_dim}; found {len(candidates)}"
            )
        return cls(
            RotaryTable(candidates[0], device=adapter.device),
            layers=len(adapter.layer_ids),
            page_size=adapter.page_size,
            record=record,
        )

    def _check(self, key):
        pages, layers, page, heads, dim = key.shape
        if (
            layers != self.layers
            or dim != self.table.head_size
            or page != self.page_size
        ):
            raise ValueError(
                f"Key tensor {tuple(key.shape)} does not match layers={self.layers}, "
                f"page={self.page_size}, head_size={self.table.head_size}"
            )
        return pages * page, heads

    def _workspace(self, tokens, heads, device, dtype, start_position):
        """Positions and buffers for one object: fused CUDA kernel or torch FP32."""
        if start_position < 0:
            raise ValueError("start_position must be nonnegative")
        positions = torch.arange(
            start_position, start_position + tokens, device=device, dtype=torch.int64
        )
        head = self.table.head_size
        if self.table.fused is not None and device.type == "cuda":
            self.table.check_positions(positions)
            return {
                "positions": positions,
                "k": torch.empty((tokens, heads, head), dtype=dtype, device=device),
                "q": torch.zeros((tokens, 1, head), dtype=dtype, device=device),
            }
        cos, sin = self.table.rows(positions)
        shape = (tokens, heads, head)
        return {
            "cos": cos,
            "sin": sin,
            "x": torch.empty(shape, dtype=torch.float32, device=device),
            "out": torch.empty(shape, dtype=torch.float32, device=device),
            "scratch": torch.empty(
                (tokens, heads, self.table.rotary_dim // 2),
                dtype=torch.float32,
                device=device,
            ),
        }

    def _apply(self, src, dst, *, inverse, work):
        """dst = R(+-p) src for one layer; both are [pages, page, heads, dim] views."""
        if "k" in work:
            k = work["k"]
            k.view_as(src).copy_(src)
            self.table.rotate_fused_(
                k, work["positions"], inverse=inverse, query=work["q"]
            )
            dst.copy_(k.view_as(src))
            return
        x, out = work["x"], work["out"]
        x.view_as(src).copy_(src)
        self.table.rotate(
            x,
            work["cos"],
            work["sin"],
            inverse=inverse,
            out=out,
            scratch=work["scratch"],
        )
        dst.copy_(out.view_as(src))

    @torch.no_grad()
    def derotate(self, key, *, start_position, audit=False):
        """Native post-RoPE keys -> pre-RoPE keys in the same dtype and layout."""
        tokens, heads = self._check(key)
        layers = key.shape[1]
        work = self._workspace(tokens, heads, key.device, key.dtype, start_position)
        result = torch.empty_like(key)
        restored = torch.empty_like(key[:, 0]) if audit else None
        worst, mismatched = 0.0, 0
        for layer in range(layers):
            self._apply(key[:, layer], result[:, layer], inverse=True, work=work)
            if audit:
                self._apply(result[:, layer], restored, inverse=False, work=work)
                native = key[:, layer]
                mismatched += int((restored != native).sum())
                error = (restored.float() - native.float()).abs().max()
                worst = max(worst, float(error))
        if audit:
            # Reductions only; key.float() would materialize a second full copy.
            scale = float(torch.maximum(key.max(), -key.min()))
            bound = 2 * torch.finfo(key.dtype).eps * scale
            event = {
                "event": "audit_pre_rope_roundtrip",
                "tokens": tokens,
                "layers": layers,
                "start_position": start_position,
                "mismatched_elements": mismatched,
                "max_abs_error": worst,
                "max_abs_native": scale,
                "bound": bound,
                "rope": self.table.description,
            }
            if worst > bound:
                raise RuntimeError(
                    f"Pre-RoPE inversion round trip exceeds two ulps: {event}"
                )
            if self.record is not None:
                self.record(event)
        return result

    def prepare_rerotation(self, key_out, *, start_position):
        """Allocate positions and workspace outside the reconstruction timer."""
        tokens, heads = self._check(key_out)
        return self._workspace(
            tokens, heads, key_out.device, key_out.dtype, start_position
        )

    @torch.no_grad()
    def rerotate_(self, key_out, scratch):
        """Pre-RoPE keys -> native post-RoPE keys, in place, without allocation."""
        for layer in range(key_out.shape[1]):
            view = key_out[:, layer]
            self._apply(view, view, inverse=False, work=scratch)


class NativeRoPEDecodePlugin(KVCompressionPlugin):
    """Wrap any numerical codec: pre-RoPE input -> ordinary post-RoPE output.

    RoPE is model-owned infrastructure, not an algorithm's responsibility. The
    forward rotation runs inside the reconstruction timer; its positions and
    FP32 workspace are allocated in prepare_decompression.
    """

    def __init__(self, inner, transform):
        self.inner, self.transform = inner, transform

    @property
    def configuration(self):
        return {
            "scheme": "pre-rope-inverse-wrapper",
            "version": 2,
            "input_key_space": "pre_rope",
            "output_key_space": "post_rope",
            "rope": "fp32-model-table-inverse",
            "inner": self.inner.configuration,
        }

    def compress(self, tensors, *, context):
        if context.get("key_space") != "pre_rope":
            raise ValueError("Pre-RoPE codec must receive de-rotated keys")
        payload = self.inner.compress(tensors, context=context)
        return CompressedPayload(
            payload.tensors,
            {
                "inner": payload.metadata,
                "start_position": context["start_position"],
                "layer_ids": context["layer_ids"],
            },
        )

    def prepare_decompression(self, payload, *, out):
        inner = CompressedPayload(payload.tensors, payload.metadata["inner"])
        if len(payload.metadata["layer_ids"]) != out["key"].shape[1]:
            raise ValueError("Layer metadata does not match the reconstruction")
        return {
            "inner_payload": inner,
            "inner": self.inner.prepare_decompression(inner, out=out),
            "rope": self.transform.prepare_rerotation(
                out["key"], start_position=payload.metadata["start_position"]
            ),
        }

    def decompress(self, payload, *, out, scratch):
        self.inner.decompress(
            scratch["inner_payload"], out=out, scratch=scratch["inner"]
        )
        self.transform.rerotate_(out["key"], scratch["rope"])
