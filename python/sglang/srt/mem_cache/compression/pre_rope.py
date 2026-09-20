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

from kvcompress.api import (
    CompressedPayload,
    KVCompressionPlugin,
    ensure_workspace_buffer,
)


def _fused_rope_kernel():
    """The kernel RotaryEmbedding.forward_cuda uses; None without CUDA support."""
    try:
        from sglang.kernels.ops.attention.rope import (
            apply_rope_with_cos_sin_cache_inplace,
        )
    except ImportError:
        return None
    return apply_rope_with_cos_sin_cache_inplace


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
        # [max_pos, rotary_dim] = [cos | sin].  Keep the model's forward table
        # only; R(-p) is S R(p) S, where S negates the second member of every
        # rotary pair.  A separate conjugate table would retain hundreds of MiB
        # for long-context models such as Gemma4.
        self.forward_table = (
            cache.detach().to(device=device, dtype=torch.float32).contiguous()
        )
        self.cos, self.sin = self.forward_table.chunk(2, dim=-1)
        self.fused = _fused_rope_kernel() if device.type == "cuda" else None
        self.description = {
            "class": type(rope).__name__,
            "head_size": self.head_size,
            "rotary_dim": self.rotary_dim,
            "base": float(rope.base),
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
        key_rotary = key[..., : self.rotary_dim]
        paired_second = (
            key_rotary[..., self.rotary_dim // 2 :]
            if self.is_neox_style
            else key_rotary[..., 1::2]
        )
        if inverse:
            paired_second.neg_()
        self.fused(
            positions=positions,
            q=query[..., : self.rotary_dim],
            k=key_rotary,
            cos_sin_cache=self.forward_table,
            is_neox=self.is_neox_style,
        )
        if inverse:
            paired_second.neg_()

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
    def from_model_ropes(
        cls,
        adapter,
        record=None,
        *,
        allow_no_rope=False,
        component="full",
        selector=None,
    ):
        """Use the process-wide RoPE cache populated while the model was built.

        Models such as Nemotron-H have no rotary table. ``allow_no_rope``
        treats that case as an identity transform; multiple matching tables
        remain ambiguous and fail closed.
        """
        from sglang.srt.layers.rotary_embedding import factory
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        if component == "full":
            pool, layer_ids = adapter.kv, adapter.layer_ids
        elif component == "swa":
            pool, layer_ids = adapter.swa_kv, adapter.swa_layer_ids
        else:
            raise ValueError(f"Unknown RoPE component: {component}")
        head_dim = int(pool.head_dim)
        candidates = []
        for key in list(factory._ROPE_DICT):
            rope = factory._get_live_rope_cache_entry(key)
            if isinstance(rope, RotaryEmbedding) and rope.head_size == head_dim:
                candidates.append(rope)
        if selector is not None:
            if "base" in selector:
                candidates = [
                    rope
                    for rope in candidates
                    if float(rope.base) == float(selector["base"])
                ]
            if "rotary_dim" in selector:
                candidates = [
                    rope
                    for rope in candidates
                    if int(rope.rotary_dim) == int(selector["rotary_dim"])
                ]
            if "class" in selector:
                candidates = [
                    rope
                    for rope in candidates
                    if type(rope).__name__ == selector["class"]
                ]
        if not candidates and allow_no_rope:
            return None
        if len(candidates) != 1:
            raise ValueError(
                "Pre-RoPE inversion needs exactly one live RotaryEmbedding with "
                f"head_size={head_dim} for {component}; found {len(candidates)}"
            )
        return cls(
            RotaryTable(candidates[0], device=adapter.device),
            layers=len(layer_ids),
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

    def ensure_rerotation_workspace(self, key_out, *, start_position, workspace):
        """Reuse one grow-only rerotation workspace and refresh its positions."""
        tokens, heads = self._check(key_out)
        device, dtype = key_out.device, key_out.dtype
        workspace = {} if workspace is None else workspace
        positions = ensure_workspace_buffer(
            workspace, "positions", (tokens,), dtype=torch.int64, device=device
        )
        torch.arange(start_position, start_position + tokens, out=positions)
        self.table.check_positions(positions)
        workspace["positions"] = positions
        head = self.table.head_size
        shape = (tokens, heads, head)
        if self.table.fused is not None and device.type == "cuda":
            workspace["k"] = ensure_workspace_buffer(
                workspace, "k", shape, dtype=dtype, device=device
            )
            workspace["q"] = ensure_workspace_buffer(
                workspace, "q", (tokens, 1, head), dtype=dtype, device=device
            )
            workspace["q"].zero_()
            for name in ("cos", "sin", "x", "out", "scratch"):
                workspace.pop(name, None)
            return workspace

        half = self.table.rotary_dim // 2
        cos = ensure_workspace_buffer(
            workspace, "cos", (tokens, half), dtype=torch.float32, device=device
        )
        sin = ensure_workspace_buffer(
            workspace, "sin", (tokens, half), dtype=torch.float32, device=device
        )
        torch.index_select(self.table.cos, 0, positions, out=cos)
        torch.index_select(self.table.sin, 0, positions, out=sin)
        workspace["cos"], workspace["sin"] = cos.unsqueeze(1), sin.unsqueeze(1)
        workspace["x"] = ensure_workspace_buffer(
            workspace, "x", shape, dtype=torch.float32, device=device
        )
        workspace["out"] = ensure_workspace_buffer(
            workspace, "out", shape, dtype=torch.float32, device=device
        )
        workspace["scratch"] = ensure_workspace_buffer(
            workspace,
            "scratch",
            (tokens, heads, half),
            dtype=torch.float32,
            device=device,
        )
        workspace.pop("k", None)
        workspace.pop("q", None)
        return workspace

    @torch.no_grad()
    def rerotate_(self, key_out, scratch):
        """Pre-RoPE keys -> native post-RoPE keys, in place, without allocation."""
        for layer in range(key_out.shape[1]):
            view = key_out[:, layer]
            self._apply(view, view, inverse=False, work=scratch)


class NativeRoPEDecodePlugin(KVCompressionPlugin):
    """Wrap a codec: canonical pre-RoPE keys -> native post-RoPE keys.

    RoPE is model-owned infrastructure, not an algorithm's responsibility. The
    wrapper accepts one transform for the full ``key`` tensor, or a mapping for
    both ``key`` and ``swa_key``. Forward rotations run inside restoration and
    reuse the store-owned workspace.
    """

    def __init__(self, inner, transform):
        self.inner = inner
        self.transforms = (
            dict(transform) if isinstance(transform, dict) else {"key": transform}
        )

    @property
    def configuration(self):
        return {
            "scheme": "pre-rope-inverse-wrapper",
            "version": 2,
            "input_key_space": "pre_rope",
            "output_key_space": "post_rope",
            "rope": "fp32-model-table-inverse",
            "key_tensors": sorted(self.transforms),
            "inner": self.inner.configuration,
        }

    def compress(self, tensors, *, context):
        if context.get("key_space") != "pre_rope":
            raise ValueError("Pre-RoPE codec must receive de-rotated keys")
        payload = self.inner.compress(tensors, context=context)
        positions = {}
        specs = context.get("tensor_specs", {})
        for name in self.transforms:
            if name not in tensors:
                continue
            spec = specs.get(name, {})
            start = (
                context.get("start_position")
                if name == "key"
                else spec.get("start_position")
            )
            if start is None:
                raise ValueError(f"Missing absolute start position for {name}")
            positions[name] = {
                "start_position": int(start),
                "layer_ids": list(spec.get("layer_ids", context.get("layer_ids", ()))),
            }
        return CompressedPayload(
            payload.tensors,
            {
                "inner": payload.metadata,
                "key_positions": positions,
            },
            payload.tensor_roles,
        )

    def ensure_decompression_workspace(self, payload, *, out, workspace):
        inner = CompressedPayload(
            payload.tensors, payload.metadata["inner"], payload.tensor_roles
        )
        workspace = {} if workspace is None else workspace
        previous_rope = workspace.get("rope", {})
        rope = {}
        for name, transform in self.transforms.items():
            if name not in out:
                continue
            position = payload.metadata["key_positions"][name]
            if len(position["layer_ids"]) != out[name].shape[1]:
                raise ValueError(
                    f"Layer metadata for {name} does not match the reconstruction"
                )
            rope[name] = transform.ensure_rerotation_workspace(
                out[name],
                start_position=position["start_position"],
                workspace=previous_rope.get(name),
            )
        workspace["inner"] = self.inner.ensure_decompression_workspace(
            inner, out=out, workspace=workspace.get("inner")
        )
        workspace["rope"] = rope
        return workspace

    def decompress(self, payload, *, out, workspace):
        inner = CompressedPayload(
            payload.tensors, payload.metadata["inner"], payload.tensor_roles
        )
        self.inner.decompress(
            inner, out=out, workspace=workspace["inner"]
        )
        for name, transform in self.transforms.items():
            if name in out:
                transform.rerotate_(out[name], workspace["rope"][name])
