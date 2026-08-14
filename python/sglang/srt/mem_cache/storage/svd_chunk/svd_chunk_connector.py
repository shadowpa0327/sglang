# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Joint-head SVD connector for coarse-grained HiCache chunks.

This module is the boundary between token-prefix identity, the live MHA NHD
KV pool, the byte-budgeted L2 chunk pool, and an optional opaque-blob L3
backend.  It deliberately does not depend on the radix cache or controller.

One K or V matrix has the logical contract ``[C, H * D]`` where ``C`` is the
fixed chunk size (at least 4096 tokens).  K and V are decomposed independently,
but all local KV heads participate in each decomposition jointly.  The direct
encoder reads physical pages without first compacting them into a dense
``[C, H * D]`` input.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import struct
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch
from sglang.srt.mem_cache.hicache_svd_chunk import (
    ParsedSVDChunkBlob,
    QuantizedKVChunk,
    make_svd_chunk_key,
    parse_svd_chunk_blob,
    serialize_svd_chunk,
)
from sglang.srt.mem_cache.hicache_svd_codec import (
    SUPPORTED_FACTOR_BITS,
    QuantizedFactor,
    QuantizedSVDFactors,
    page_ids_from_device_indices,
    quantize_svd_factors,
    view_nhd_as_joint_pages,
)
from sglang.srt.mem_cache.hicache_svd_restore import (
    SUPPORTED_SVD_RESTORE_COMPUTE,
    fused_reconstruct_svd_batch_into,
    fused_reconstruct_svd_into,
    fused_svd_restore_available,
)
from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_l3_tasks import (
    SVDChunkL3TaskManager,
)

_PREFIX_HASH_DOMAIN = b"sglang-hicache-svd-prefix-v1\0"
_SEED_DOMAIN = b"sglang-hicache-svd-omega-v1\0"
_MAX_DIRECT_RANK = 128
_RESTORE_WORKSPACE_ALIGNMENT = 64
# Triton specializes scalar launch arguments by equality/divisibility.  These
# values cover the full-chunk, special ``1``, and generic unaligned classes
# observed for ``first_row_start`` without compiling every possible row.
_BATCH_TMA_PREWARM_FIRST_ROWS = (0, 1, 13)

logger = logging.getLogger(__name__)


def _align_restore_workspace_nbytes(nbytes: int) -> int:
    """Round one serialized blob stride to the factor-section alignment."""

    return (
        (nbytes + _RESTORE_WORKSPACE_ALIGNMENT - 1)
        // _RESTORE_WORKSPACE_ALIGNMENT
        * _RESTORE_WORKSPACE_ALIGNMENT
    )


class ChunkPoolProtocol(Protocol):
    """Small surface required from the byte-budgeted L2 pool."""

    def put(self, key: str, blob: torch.Tensor) -> bool: ...

    def acquire(self, keys: Sequence[str]) -> Any: ...

    def acquire_parsed(
        self,
        keys: Sequence[str],
        *,
        expected_codec_metadata: Optional[Mapping] = None,
        validator: Any = None,
    ) -> Any: ...

    def put_parsed(
        self,
        key: str,
        parsed: ParsedSVDChunkBlob,
        *,
        contract_validated: bool = False,
    ) -> bool: ...

    def exists(self, key: str) -> bool: ...

    def clear(self) -> int: ...


@dataclass(frozen=True)
class SVDChunkConnectorConfig:
    """Experimental SVD/quantization contract encoded into every chunk key."""

    chunk_tokens: int = 4096
    key_rank: int = 32
    value_rank: int = 32
    bits_u: int = 4
    bits_r: int = 4
    scale_mode: str = "matrix"
    group_size_u: Optional[int] = None
    group_size_r: Optional[int] = None
    niter: int = 2
    backend: str = "triton"
    precision: str = "ieee"
    restore_schedule: str = "synchronous"
    restore_compute: str = "fp16"
    restore_io: str = "pointer"
    restore_prewarm_chunks: int = 8
    seed: int = 0
    l2_capacity_gb: Optional[float] = None
    l3_path: Optional[str] = None
    max_pending_writes: int = 1
    encode_batch_size: int = 8
    max_deferred_writes: int = 64
    write_retry_limit: int = 3
    l3_retry_limit: int = 3
    log_every: int = 100

    def __post_init__(self) -> None:
        _require_plain_int("chunk_tokens", self.chunk_tokens, minimum=4096)
        _require_plain_int("key_rank", self.key_rank, minimum=1)
        _require_plain_int("value_rank", self.value_rank, minimum=1)
        if self.key_rank > _MAX_DIRECT_RANK or self.value_rank > _MAX_DIRECT_RANK:
            raise ValueError("key_rank and value_rank must be at most 128")
        if self.bits_u not in SUPPORTED_FACTOR_BITS:
            raise ValueError(
                f"bits_u must be one of {SUPPORTED_FACTOR_BITS}, got {self.bits_u}"
            )
        if self.bits_r not in SUPPORTED_FACTOR_BITS:
            raise ValueError(
                f"bits_r must be one of {SUPPORTED_FACTOR_BITS}, got {self.bits_r}"
            )
        if self.scale_mode not in ("matrix", "row"):
            raise ValueError("scale_mode must be 'matrix' or 'row'")
        if self.scale_mode == "matrix":
            if self.group_size_u is not None or self.group_size_r is not None:
                raise ValueError("group sizes are only valid for row scale mode")
        else:
            _require_plain_int("group_size_u", self.group_size_u, minimum=1)
            _require_plain_int("group_size_r", self.group_size_r, minimum=1)
        _require_plain_int("niter", self.niter, minimum=0)
        if self.backend not in ("triton", "cute_bf16"):
            raise ValueError("backend must be 'triton' or 'cute_bf16'")
        if self.precision not in ("ieee", "tf32"):
            raise ValueError("precision must be 'ieee' or 'tf32'")
        if self.backend == "cute_bf16" and self.precision != "ieee":
            raise ValueError("cute_bf16 uses fixed BF16 inputs; select ieee")
        if self.restore_schedule not in ("full", "layerwise", "synchronous"):
            raise ValueError(
                "restore_schedule must be 'full', 'layerwise', or 'synchronous'"
            )
        if self.restore_compute not in SUPPORTED_SVD_RESTORE_COMPUTE:
            raise ValueError(
                "restore_compute must be one of "
                f"{SUPPORTED_SVD_RESTORE_COMPUTE}, got {self.restore_compute!r}"
            )
        if self.restore_io not in ("pointer", "tma"):
            raise ValueError("restore_io must be 'pointer' or 'tma'")
        if self.restore_io == "tma" and self.restore_compute != "fp8_e4m3":
            raise ValueError("TMA restore requires restore_compute='fp8_e4m3'")
        if self.restore_io == "tma" and (
            self.key_rank != 128
            or self.value_rank != 128
            or self.bits_u != 4
            or self.bits_r != 4
        ):
            raise ValueError("TMA restore currently requires rank-128 INT4 U/R factors")
        _require_plain_int(
            "restore_prewarm_chunks", self.restore_prewarm_chunks, minimum=1
        )
        _require_plain_int("seed", self.seed, minimum=0)
        if self.seed >= 1 << 63:
            raise ValueError("seed must fit in a signed 63-bit integer")
        if self.l2_capacity_gb is not None:
            if (
                isinstance(self.l2_capacity_gb, bool)
                or not isinstance(self.l2_capacity_gb, (int, float))
                or not math.isfinite(self.l2_capacity_gb)
                or self.l2_capacity_gb < 0
            ):
                raise ValueError("l2_capacity_gb must be a non-negative finite number")
        if self.l3_path is not None and (
            not isinstance(self.l3_path, str) or not self.l3_path
        ):
            raise ValueError("l3_path must be a nonempty string or null")
        _require_plain_int("max_pending_writes", self.max_pending_writes, minimum=1)
        _require_plain_int("encode_batch_size", self.encode_batch_size, minimum=1)
        _require_plain_int("max_deferred_writes", self.max_deferred_writes, minimum=1)
        _require_plain_int("write_retry_limit", self.write_retry_limit, minimum=1)
        _require_plain_int("l3_retry_limit", self.l3_retry_limit, minimum=1)
        _require_plain_int("log_every", self.log_every, minimum=1)


@dataclass
class SVDChunkConnectorStats:
    """Lifetime counters for the experimental compressed tier."""

    store_calls: int = 0
    lookup_calls: int = 0
    encoded_chunks: int = 0
    encode_batches: int = 0
    batched_encoded_chunks: int = 0
    encoded_tokens: int = 0
    raw_nbytes: int = 0
    encoded_nbytes: int = 0
    l2_puts: int = 0
    l3_puts: int = 0
    l2_rejections: int = 0
    l3_rejections: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    lookup_hit_chunks: int = 0
    lookup_misses: int = 0
    invalid_l2_blobs: int = 0
    invalid_l3_blobs: int = 0
    l3_io_errors: int = 0
    l3_retry_queued: int = 0
    l3_retry_succeeded: int = 0
    l3_retry_exhausted: int = 0
    restored_chunks: int = 0
    restored_tokens: int = 0
    restored_blob_nbytes: int = 0
    restore_failures: int = 0
    encode_wall_ms: float = 0.0
    restore_enqueue_cpu_ms: float = 0.0
    restore_h2d_ms: float = 0.0
    restore_gpu_ms: float = 0.0
    restore_timed_batches: int = 0
    restore_fp8_kernel_launches: int = 0
    restore_tma_kernel_launches: int = 0

    def snapshot(self) -> dict[str, int | float]:
        result = dict(vars(self))
        result["compression_ratio"] = (
            self.raw_nbytes / self.encoded_nbytes if self.encoded_nbytes else 0.0
        )
        return result


def _require_plain_int(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _config_from_mapping(values: Mapping[str, Any]) -> SVDChunkConnectorConfig:
    values = dict(values)
    rank = values.pop("rank", None)
    if rank is not None:
        values.setdefault("key_rank", rank)
        values.setdefault("value_rank", rank)
    bits = values.pop("bits", None)
    if bits is not None:
        values.setdefault("bits_u", bits)
        values.setdefault("bits_r", bits)
    group_size = values.pop("group_size", None)
    if group_size is not None:
        values.setdefault("group_size_u", group_size)
        values.setdefault("group_size_r", group_size)
    valid = set(SVDChunkConnectorConfig.__dataclass_fields__)
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"unknown HiCache SVD chunk fields: {unknown}")
    return SVDChunkConnectorConfig(**values)


def resolve_svd_chunk_connector_config(
    value: Any,
) -> Optional[SVDChunkConnectorConfig]:
    """Parse a config object, mapping, JSON object, or boolean spelling.

    ``None``/``False`` disables the connector and ``True`` selects defaults.
    Aliases ``rank``, ``bits``, and ``group_size`` set both corresponding
    K/V or U/R fields unless an explicit field is also present.
    """

    if value is None or value is False:
        return None
    if isinstance(value, SVDChunkConnectorConfig):
        return value
    if value is True:
        return SVDChunkConnectorConfig()
    if isinstance(value, Mapping):
        return _config_from_mapping(value)
    if not isinstance(value, str):
        raise TypeError(
            "HiCache SVD chunk config must be bool, string, mapping, or "
            "SVDChunkConnectorConfig"
        )
    stripped = value.strip()
    if stripped.lower() in ("", "0", "false", "off", "no", "none"):
        return None
    if stripped.lower() in ("1", "true", "on", "yes"):
        return SVDChunkConnectorConfig()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("HiCache SVD chunk JSON config must be an object")
    return _config_from_mapping(parsed)


def _dtype_name(dtype: torch.dtype | str) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    if not isinstance(dtype, str) or not dtype:
        raise TypeError("raw_dtype must be a torch.dtype or nonempty string")
    return dtype.removeprefix("torch.")


def build_svd_codec_metadata(
    config: SVDChunkConnectorConfig,
    *,
    model_name: Optional[str],
    page_size: int,
    local_layer_ids: Sequence[int],
    kv_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    raw_dtype: torch.dtype | str,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
) -> dict[str, Any]:
    """Build the canonical model/parallel/geometry/codec key namespace."""

    if not isinstance(config, SVDChunkConnectorConfig):
        raise TypeError("config must be SVDChunkConnectorConfig")
    for name, value in (
        ("page_size", page_size),
        ("kv_heads", kv_heads),
        ("key_head_dim", key_head_dim),
        ("value_head_dim", value_head_dim),
        ("tp_size", tp_size),
        ("pp_size", pp_size),
        ("attn_cp_size", attn_cp_size),
    ):
        _require_plain_int(name, value, minimum=1)
    for name, rank, size in (
        ("tp", tp_rank, tp_size),
        ("pp", pp_rank, pp_size),
        ("attn_cp", attn_cp_rank, attn_cp_size),
    ):
        _require_plain_int(f"{name}_rank", rank, minimum=0)
        if rank >= size:
            raise ValueError(f"{name}_rank must be smaller than {name}_size")
    if config.chunk_tokens % page_size:
        raise ValueError("chunk_tokens must be divisible by page_size")
    layers = tuple(local_layer_ids)
    if not layers:
        raise ValueError("local_layer_ids must not be empty")
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise TypeError("local_layer_ids must contain integers")
    if len(set(layers)) != len(layers):
        raise ValueError("local_layer_ids must not contain duplicates")

    return {
        "format": "hicache-svdq1",
        "model": model_name or "",
        "parallel": {
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": pp_rank,
            "pp_size": pp_size,
            "attn_cp_rank": attn_cp_rank,
            "attn_cp_size": attn_cp_size,
        },
        "geometry": {
            "page_size": page_size,
            "chunk_tokens": config.chunk_tokens,
            "local_layer_ids": list(layers),
            "kv_heads": kv_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "key_features": kv_heads * key_head_dim,
            "value_features": kv_heads * value_head_dim,
            "raw_dtype": _dtype_name(raw_dtype),
            "layout": "NHD-joint-head",
        },
        "codec": {
            "algorithm": "direct-randomized-svd",
            "key_rank": config.key_rank,
            "value_rank": config.value_rank,
            "bits_u": config.bits_u,
            "bits_r": config.bits_r,
            "scale_mode": config.scale_mode,
            "group_size_u": config.group_size_u,
            "group_size_r": config.group_size_r,
            "scale_dtype": "float16",
            "sigma_dtype": "float16",
            "niter": config.niter,
            "backend": config.backend,
            "precision": config.precision,
            "seed": config.seed,
            "packing": "rank-low-bits-first-offset-binary",
        },
    }


def build_svd_codec_metadata_from_pool(
    config: SVDChunkConnectorConfig,
    device_pool: Any,
    *,
    model_name: Optional[str],
    page_size: Optional[int] = None,
    local_layer_ids: Optional[Sequence[int]] = None,
    **parallel: int,
) -> dict[str, Any]:
    """Convenience metadata builder for an ``MHATokenToKVPool`` instance."""

    layer_count = len(getattr(device_pool, "k_buffer", ()))
    if local_layer_ids is None:
        start_layer = int(getattr(device_pool, "start_layer", 0) or 0)
        local_layer_ids = range(start_layer, start_layer + layer_count)
    return build_svd_codec_metadata(
        config,
        model_name=model_name,
        page_size=int(page_size or device_pool.page_size),
        local_layer_ids=local_layer_ids,
        kv_heads=int(device_pool.head_num),
        key_head_dim=int(device_pool.head_dim),
        value_head_dim=int(device_pool.v_head_dim),
        raw_dtype=device_pool.k_buffer[0].dtype,
        **parallel,
    )


@dataclass(frozen=True)
class SVDChunkIdentity:
    """Identity of one fixed chunk in a cumulative token prefix."""

    chunk_index: int
    token_start: int
    token_end: int
    prefix_hash: str
    key: str


def _normalize_token_ids(token_ids: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1 or token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be a one-dimensional INT32/INT64 tensor")
        values = token_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        if isinstance(token_ids, (str, bytes)):
            raise TypeError("token_ids must be an integer sequence")
        try:
            values = list(token_ids)
        except TypeError as exc:
            raise TypeError("token_ids must be an integer sequence") from exc
    normalized = []
    for token in values:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("token_ids must contain integers")
        if not 0 <= token < 1 << 63:
            raise ValueError("token IDs must fit in a nonnegative signed INT64")
        normalized.append(token)
    return tuple(normalized)


def derive_svd_chunk_identities(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    namespace: Any,
    codec_metadata: Mapping[str, Any],
    chunk_tokens: int,
) -> tuple[SVDChunkIdentity, ...]:
    """Derive cumulative keys for every complete fixed-size prefix chunk.

    The hash for chunk ``i`` commits to the previous cumulative hash and the
    current chunk's token IDs.  Consequently an identical 4K tail reached
    through a different earlier prefix receives a different key.  An
    incomplete tail is intentionally omitted.
    """

    _require_plain_int("chunk_tokens", chunk_tokens, minimum=4096)
    tokens = _normalize_token_ids(token_ids)
    complete_chunks = len(tokens) // chunk_tokens
    cumulative = hashlib.sha256(_PREFIX_HASH_DOMAIN).digest()
    identities = []
    for chunk_index in range(complete_chunks):
        start = chunk_index * chunk_tokens
        end = start + chunk_tokens
        digest = hashlib.sha256()
        digest.update(_PREFIX_HASH_DOMAIN)
        digest.update(cumulative)
        digest.update(struct.pack("<QQ", chunk_index, chunk_tokens))
        # Pack the complete token block in C.  Per-token ``struct.pack`` and
        # ``hash.update`` calls showed up directly in warm-hit scheduler time.
        digest.update(struct.pack(f"<{chunk_tokens}Q", *tokens[start:end]))
        cumulative = digest.digest()
        prefix_hash = cumulative.hex()
        identities.append(
            SVDChunkIdentity(
                chunk_index=chunk_index,
                token_start=start,
                token_end=end,
                prefix_hash=prefix_hash,
                key=make_svd_chunk_key(
                    codec_metadata=codec_metadata,
                    namespace=namespace,
                    end_page_hash=prefix_hash,
                ),
            )
        )
    return tuple(identities)


def _layout_index_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify one immutable index view for an ephemeral chunk session."""

    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors do not expose a version counter.  The cache retains
        # them below, which still prevents allocator pointer reuse.
        version = None
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor.storage_offset(),
        tensor.numel(),
        tuple(tensor.stride()),
        version,
    )


class DirectPagedSVDEncoder:
    """Lazy adapter around ``paged_svd.paged_svd_lowrank_direct``."""

    def __init__(self, paged_svd_module: Any = None) -> None:
        self._paged_svd = paged_svd_module
        self._chunk_cache: Optional[dict[str, dict[Any, Any]]] = None

    @contextmanager
    def chunk_session(self):
        """Reuse page layout, scratch, and Omega allocations within one chunk."""

        if self._chunk_cache is not None:
            raise RuntimeError("nested direct-paged SVD chunk sessions are invalid")
        self._chunk_cache = {
            "layouts": {},
            "layout_sources": {},
            "workspaces": {},
            "omegas": {},
        }
        try:
            yield
        finally:
            self._chunk_cache = None

    def _module(self) -> Any:
        if self._paged_svd is None:
            try:
                self._paged_svd = importlib.import_module("paged_svd")
            except ModuleNotFoundError as exc:
                if exc.name != "paged_svd":
                    raise
                raise ImportError(
                    "HiCache SVD compression requires the optional 'paged_svd' "
                    "package. Install /home/ccchang/svd_on_paged or add that "
                    "checkout to PYTHONPATH before starting the server."
                ) from exc
        return self._paged_svd

    def preflight(self) -> None:
        """Fail early when the optional direct paged-SVD API is unavailable."""

        module = self._module()
        missing = [
            name
            for name in (
                "paged_svd_lowrank_direct",
                "prepare_paged_layout",
                "prepare_direct_workspace",
            )
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise ImportError(
                "paged_svd is missing required direct API functions: "
                + ", ".join(missing)
            )

    def factorize(
        self,
        *,
        pages: torch.Tensor,
        page_ids: torch.Tensor,
        rank: int,
        niter: int,
        backend: str,
        precision: str,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        module = self._module()
        page_ids = page_ids.to(device=pages.device, dtype=torch.int64)
        layout_key = (
            "single",
            tuple(pages.shape),
            pages.dtype,
            pages.device,
            _layout_index_signature(page_ids),
        )
        cache = self._chunk_cache
        layout = None if cache is None else cache["layouts"].get(layout_key)
        if layout is None:
            indptr = torch.tensor([0, page_ids.numel()], device=pages.device)
            last_page_lens = torch.tensor([pages.shape[1]], device=pages.device)
            layout = module.prepare_paged_layout(
                pages, indptr, page_ids, last_page_lens
            )
            if cache is not None:
                cache["layouts"][layout_key] = layout
                cache["layout_sources"][layout_key] = page_ids
        workspace_key = (layout_key, rank)
        workspace = None if cache is None else cache["workspaces"].get(workspace_key)
        if workspace is None:
            workspace = module.prepare_direct_workspace(pages, layout, rank)
            if cache is not None:
                cache["workspaces"][workspace_key] = workspace
        generator = torch.Generator(device=pages.device)
        generator.manual_seed(seed)
        omega = None if cache is None else cache["omegas"].get(workspace_key)
        if omega is None:
            omega = torch.empty(
                1,
                pages.shape[2],
                rank,
                dtype=torch.float32,
                device=pages.device,
            )
            if cache is not None:
                cache["omegas"][workspace_key] = omega
        torch.randn(
            omega.shape,
            dtype=omega.dtype,
            device=omega.device,
            generator=generator,
            out=omega,
        )
        result = module.paged_svd_lowrank_direct(
            pages,
            layout=layout,
            q=rank,
            niter=niter,
            workspace=workspace,
            omega=omega,
            precision=precision,
            backend=backend,
        )
        if hasattr(result, "sequence"):
            result = result.sequence(0)
        return _normalize_factorizer_result(
            result,
            rank=rank,
            rows=int(page_ids.numel() * pages.shape[1]),
            features=int(pages.shape[2]),
            device=pages.device,
            check_finite=False,
        )

    def factorize_batch(
        self,
        *,
        pages: torch.Tensor,
        page_ids: Sequence[torch.Tensor],
        rank: int,
        niter: int,
        backend: str,
        precision: str,
        seeds: Sequence[int],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Factor equal-shaped logical matrices from one physical page pool.

        Each batch entry remains an independent joint-head SVD.  Batching only
        shares launches for the paged products and dense QR/SVD/lift stages; it
        never concatenates prefix chunks into one mathematical matrix.
        """

        if not page_ids:
            raise ValueError("page_ids batch must not be empty")
        if len(page_ids) != len(seeds):
            raise ValueError("page_ids and seeds must have the same batch size")
        if backend == "cute_bf16" and len(page_ids) != 1:
            # The optional CuTe page-product backend deliberately supports B=1.
            # Preserve correctness for this execution mode without pretending
            # that it has a batched kernel.
            return tuple(
                self.factorize(
                    pages=pages,
                    page_ids=ids,
                    rank=rank,
                    niter=niter,
                    backend=backend,
                    precision=precision,
                    seed=seed,
                )
                for ids, seed in zip(page_ids, seeds)
            )

        module = self._module()
        normalized_ids = tuple(
            ids.to(device=pages.device, dtype=torch.int64) for ids in page_ids
        )
        page_counts = tuple(int(ids.numel()) for ids in normalized_ids)
        if not page_counts or any(count <= 0 for count in page_counts):
            raise ValueError("every page_ids batch entry must be nonempty")
        if len(set(page_counts)) != 1:
            raise ValueError("SVD chunk encode batches require equal page counts")

        batch_size = len(normalized_ids)
        pages_per_matrix = page_counts[0]
        indices_signature = tuple(
            _layout_index_signature(ids) for ids in normalized_ids
        )
        layout_key = (
            "batch",
            tuple(pages.shape),
            pages.dtype,
            pages.device,
            batch_size,
            pages_per_matrix,
            indices_signature,
        )
        cache = self._chunk_cache
        layout = None if cache is None else cache["layouts"].get(layout_key)
        if layout is None:
            indptr = (
                torch.arange(
                    batch_size + 1,
                    dtype=torch.int64,
                    device=pages.device,
                )
                * pages_per_matrix
            )
            indices = torch.cat(normalized_ids)
            last_page_lens = torch.full(
                (batch_size,),
                int(pages.shape[1]),
                dtype=torch.int64,
                device=pages.device,
            )
            layout = module.prepare_paged_layout(pages, indptr, indices, last_page_lens)
            if cache is not None:
                cache["layouts"][layout_key] = layout
                cache["layout_sources"][layout_key] = normalized_ids

        workspace_key = (layout_key, rank)
        workspace = None if cache is None else cache["workspaces"].get(workspace_key)
        if workspace is None:
            # Keep the package's canonical split count.  Changing it with the
            # batch size changes FP32 reduction order and can produce different
            # serialized bytes for the same content-addressed prefix key.
            workspace = module.prepare_direct_workspace(pages, layout, rank)
            if cache is not None:
                cache["workspaces"][workspace_key] = workspace

        omega = None if cache is None else cache["omegas"].get(workspace_key)
        if omega is None:
            omega = torch.empty(
                batch_size,
                pages.shape[2],
                rank,
                dtype=torch.float32,
                device=pages.device,
            )
            if cache is not None:
                cache["omegas"][workspace_key] = omega
        for batch_index, seed in enumerate(seeds):
            generator = torch.Generator(device=pages.device)
            generator.manual_seed(seed)
            torch.randn(
                omega[batch_index].shape,
                dtype=omega.dtype,
                device=omega.device,
                generator=generator,
                out=omega[batch_index],
            )

        result = module.paged_svd_lowrank_direct(
            pages,
            layout=layout,
            q=rank,
            niter=niter,
            workspace=workspace,
            omega=omega,
            precision=precision,
            backend=backend,
        )
        rows = pages_per_matrix * int(pages.shape[1])
        return tuple(
            _normalize_factorizer_result(
                result.sequence(batch_index),
                rank=rank,
                rows=rows,
                features=int(pages.shape[2]),
                device=pages.device,
                check_finite=False,
            )
            for batch_index in range(batch_size)
        )


def _normalize_factorizer_result(
    result: Any,
    *,
    rank: int,
    rows: Optional[int] = None,
    features: Optional[int] = None,
    device: Optional[torch.device] = None,
    check_finite: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if all(hasattr(result, name) for name in ("U", "S", "V")):
        u, sigma, right = result.U, result.S, result.V
    else:
        try:
            u, sigma, right = result
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "SVD encoder must return U, Sigma, and right factor"
            ) from exc
    if u.ndim == 3 and u.shape[0] == 1:
        u = u[0]
    if sigma.ndim == 2 and sigma.shape[0] == 1:
        sigma = sigma[0]
    if right.ndim == 3 and right.shape[0] == 1:
        right = right[0]
    if u.ndim != 2 or sigma.ndim != 1 or right.ndim != 2:
        raise ValueError("SVD encoder returned invalid factor ranks")
    if u.shape[1] != rank or sigma.shape[0] != rank or right.shape[1] != rank:
        raise ValueError("SVD encoder returned a different retained rank")
    if rows is not None and u.shape[0] != rows:
        raise ValueError(
            f"SVD encoder U must have shape [{rows}, {rank}], got {tuple(u.shape)}"
        )
    if features is not None and right.shape[0] != features:
        raise ValueError(
            "SVD encoder R must have shape "
            f"[{features}, {rank}], got {tuple(right.shape)}"
        )
    if not all(tensor.is_floating_point() for tensor in (u, sigma, right)):
        raise TypeError("SVD encoder factors must use floating dtypes")
    if device is not None and any(
        tensor.device != device for tensor in (u, sigma, right)
    ):
        raise ValueError("SVD encoder factors must remain on the input device")
    if check_finite and not all(
        bool(torch.isfinite(tensor).all()) for tensor in (u, sigma, right)
    ):
        raise ValueError("SVD encoder factors must contain only finite values")
    return u, sigma, right


def _matrix_seed(
    config_seed: int, identity: SVDChunkIdentity, layer_index: int, kind: str
) -> int:
    digest = hashlib.sha256(
        _SEED_DOMAIN
        + struct.pack("<Q", config_seed)
        + bytes.fromhex(identity.prefix_hash)
        + struct.pack("<Q", layer_index)
        + kind.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class SVDChunkL3Adapter:
    """Opaque ``uint8`` blob adapter for ``HiCacheFile`` or a like backend."""

    def __init__(self, backend: Any) -> None:
        for name in ("get", "set", "exists"):
            if not callable(getattr(backend, name, None)):
                raise TypeError(f"L3 backend must provide {name}()")
        self.backend = backend

    def exists(self, key: str) -> bool:
        return bool(self.backend.exists(key))

    def get(self, key: str) -> Optional[torch.Tensor]:
        blob = self.backend.get(key)
        if blob is None:
            return None
        _validate_blob(blob)
        return blob

    def put(self, key: str, blob: torch.Tensor) -> bool:
        _validate_blob(blob)
        return bool(self.backend.set(key, blob))

    def delete(self, key: str) -> bool:
        delete = getattr(self.backend, "delete", None)
        if not callable(delete):
            return False
        result = delete(key)
        return True if result is None else bool(result)


def _validate_blob(blob: torch.Tensor) -> None:
    if not isinstance(blob, torch.Tensor):
        raise TypeError("SVD chunk blob must be a torch.Tensor")
    if blob.device.type != "cpu" or blob.dtype != torch.uint8 or blob.ndim != 1:
        raise ValueError("SVD chunk blob must be one-dimensional CPU uint8")


@dataclass
class LoadMarker:
    """Pinned/owned compressed chunks covering a restorable prefix suffix."""

    rid: Optional[str]
    restore_start: int
    matched_end: int
    first_chunk_index: int
    chunk_tokens: int
    chunk_keys: tuple[str, ...]
    chunks: tuple[ParsedSVDChunkBlob, ...] = field(repr=False)
    _leases: tuple[Any, ...] = field(default=(), repr=False)
    _released: bool = field(default=False, init=False, repr=False)
    _release_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.restore_start < 0 or self.matched_end <= self.restore_start:
            raise ValueError("LoadMarker must cover a nonempty suffix")
        if len(self.chunk_keys) != len(self.chunks) or not self.chunk_keys:
            raise ValueError(
                "LoadMarker keys and chunks must have equal nonzero length"
            )

    @property
    def blobs(self) -> tuple[torch.Tensor, ...]:
        """Compatibility view of the marker-owned serialized allocations."""

        return tuple(chunk.blob for chunk in self.chunks)

    @property
    def matched_tokens(self) -> int:
        return self.matched_end - self.restore_start

    @property
    def chunk_start(self) -> int:
        return self.first_chunk_index * self.chunk_tokens

    @property
    def chunk_end(self) -> int:
        return self.matched_end

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
            leases = self._leases
            self._leases = ()
        for lease in leases:
            lease.release()


@dataclass
class _RestoreOperation:
    """One validated marker queued until the scheduler starts its load batch."""

    marker: LoadMarker
    destination_slots: torch.Tensor


@dataclass
class _RestoreBatch:
    """Resources retained until the final per-layer restore event completes."""

    producer_index: int
    operations: tuple[_RestoreOperation, ...]
    device_blobs: tuple[torch.Tensor, ...]
    device_slots: tuple[torch.Tensor, ...]
    finish_event: Any
    timing_start_event: Any
    timing_h2d_event: Any
    timing_done_event: Any


@dataclass
class _L3Prefetch:
    """Coalesced background reads for one request's next chunk suffix."""

    rid: str
    first_chunk_index: int
    keys: tuple[str, ...]
    futures: tuple[Future, ...]


@dataclass
class _PendingL3Write:
    """Pinned L2 source and the bounded number of L3 submissions made."""

    lease: Any
    attempts: int = 0
    had_failure: bool = False


def _slice_quantized_factor(
    factor: QuantizedFactor, start: int, end: int
) -> QuantizedFactor:
    scale = factor.scale
    if factor.scale_mode == "row":
        scale = scale[start:end]
    return QuantizedFactor(
        qdata=factor.qdata[start:end],
        scale=scale,
        bits=factor.bits,
        rows=end - start,
        rank=factor.rank,
        scale_mode=factor.scale_mode,
        group_size=factor.group_size,
    )


def _factor_to_device(factor: QuantizedFactor, device: torch.device) -> QuantizedFactor:
    return QuantizedFactor(
        qdata=factor.qdata.to(device=device),
        scale=factor.scale.to(device=device),
        bits=factor.bits,
        rows=factor.rows,
        rank=factor.rank,
        scale_mode=factor.scale_mode,
        group_size=factor.group_size,
    )


def _svd_rows_to_device(
    factors: QuantizedSVDFactors,
    row_start: int,
    row_end: int,
    device: torch.device,
) -> QuantizedSVDFactors:
    if not 0 <= row_start < row_end <= factors.chunk_tokens:
        raise ValueError("requested SVD row slice is outside the encoded chunk")
    return QuantizedSVDFactors(
        u=_factor_to_device(
            _slice_quantized_factor(factors.u, row_start, row_end), device
        ),
        sigma=factors.sigma.to(device=device),
        right=_factor_to_device(factors.right, device),
    )


def reconstruct_svd_rows(
    factors: QuantizedSVDFactors,
    row_start: int,
    row_end: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode only ``U[row_start:row_end]`` and reconstruct those matrix rows."""

    device = torch.device(device)
    device_factors = _svd_rows_to_device(factors, row_start, row_end, device)
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device_factors.reconstruct(dtype=compute_dtype).to(dtype=dtype)


def reconstruct_svd_rows_into(
    factors: QuantizedSVDFactors,
    row_start: int,
    row_end: int,
    *,
    destination_slots: torch.Tensor,
    destination: torch.Tensor,
    compute_mode: str = "fp16",
    use_tma: bool = False,
) -> None:
    """Reconstruct selected rows directly into a paged NHD KV buffer."""

    rows = row_end - row_start
    if destination_slots.numel() != rows:
        raise ValueError("destination slot count does not match restored rows")
    if destination.ndim != 3 or not destination.is_contiguous():
        raise ValueError("destination must be a contiguous NHD KV buffer")
    if compute_mode not in SUPPORTED_SVD_RESTORE_COMPUTE:
        raise ValueError(
            f"compute_mode must be one of {SUPPORTED_SVD_RESTORE_COMPUTE}, "
            f"got {compute_mode!r}"
        )
    device = destination.device
    slots = destination_slots.to(device=device, dtype=torch.int64)
    device_factors = _svd_rows_to_device(factors, row_start, row_end, device)
    if (
        device.type == "cuda"
        and destination.dtype in (torch.float16, torch.bfloat16)
        and fused_svd_restore_available()
    ):
        fused_reconstruct_svd_into(
            device_factors,
            slots,
            destination,
            compute_mode=compute_mode,
            use_tma=use_tma,
        )
        return

    if use_tma:
        raise RuntimeError("TMA SVD restore requires the fused CUDA implementation")
    if compute_mode != "fp16":
        raise RuntimeError(
            f"{compute_mode} SVD restore requires the fused CUDA implementation"
        )

    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
    reconstructed = device_factors.reconstruct(dtype=compute_dtype).to(
        dtype=destination.dtype
    )
    destination.view(destination.shape[0], -1).index_copy_(0, slots, reconstructed)


class SVDChunkConnector:
    """Encode, publish, look up, and restore compressed MHA prefix chunks."""

    MAX_L3_RETRIES_PER_TICK = 8

    def __init__(
        self,
        config: SVDChunkConnectorConfig,
        *,
        device_pool: Any,
        page_size: int,
        pool: ChunkPoolProtocol,
        codec_metadata: Mapping[str, Any],
        storage: Any = None,
        encoder: Any = None,
        buffer_layer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(config, SVDChunkConnectorConfig):
            raise TypeError("config must be SVDChunkConnectorConfig")
        _require_plain_int("page_size", page_size, minimum=1)
        if config.chunk_tokens % page_size:
            raise ValueError("chunk_tokens must be divisible by page_size")
        self.config = config
        self.device_pool = device_pool
        self.page_size = page_size
        self.pool = pool
        self.codec_metadata = dict(codec_metadata)
        self.storage = (
            storage
            if storage is None or isinstance(storage, SVDChunkL3Adapter)
            else SVDChunkL3Adapter(storage)
        )
        self.encoder = encoder if encoder is not None else DirectPagedSVDEncoder()
        layer_count = len(getattr(device_pool, "k_buffer", ()))
        self.buffer_layer_indices = tuple(
            range(layer_count) if buffer_layer_indices is None else buffer_layer_indices
        )
        self._validate_device_contract()
        first_buffer = self.device_pool.k_buffer[self.buffer_layer_indices[0]]
        self._restore_device = first_buffer.device
        self._load_stream = None
        self._write_stream = None
        self.layer_done_counter = None
        self._load_workspaces: list[Optional[torch.Tensor]] = []
        self._load_workspace_mappings: list[dict[Any, Any]] = []
        self._restore_workspace_lock = threading.Lock()
        if self._restore_device.type == "cuda":
            from sglang.srt.managers.cache_controller import LayerDoneCounter

            with torch.cuda.device(self._restore_device):
                self._load_stream = torch.cuda.Stream(device=self._restore_device)
                self._write_stream = torch.cuda.Stream(device=self._restore_device)
                self.layer_done_counter = LayerDoneCounter(
                    len(self.buffer_layer_indices)
                )
                self._load_workspaces = [
                    None for _ in range(self.layer_done_counter.num_counters)
                ]
                self._load_workspace_mappings = [
                    {} for _ in range(self.layer_done_counter.num_counters)
                ]
        self._load_queue: list[_RestoreOperation] = []
        self._inflight_loads: list[_RestoreBatch] = []
        self._load_lock = threading.RLock()
        self._write_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="SVDChunkWrite",
        )
        self._write_futures: set[Future] = set()
        self._write_lock = threading.Lock()
        self._l3_tasks = (
            None
            if self.storage is None
            else SVDChunkL3TaskManager(
                self.storage,
                max_workers=2,
                max_pending_tasks=64,
                read_transform=self._validate_prefetched_l3_blob,
            )
        )
        self._prefetches: dict[str, _L3Prefetch] = {}
        self._prefetched_chunks: dict[str, dict[str, ParsedSVDChunkBlob]] = {}
        self._prefetch_loaded_tokens: dict[str, int] = {}
        self._prefetch_lock = threading.RLock()
        self._pending: dict[str, LoadMarker] = {}
        self._pending_lock = threading.Lock()
        # Failed durable writes retain a byte-accounted L2 lease until a
        # scheduler tick retries them.  This avoids losing the only compressed
        # copy immediately after its L1 source has been evicted.
        self._pending_l3: dict[str, _PendingL3Write] = {}
        self._pending_l3_puts: set[str] = set()
        self._pending_l3_lock = threading.Lock()
        self._stats = SVDChunkConnectorStats()
        self._stats_lock = threading.Lock()
        self._last_logged_activity = 0
        self._batch_tma_warm_signatures: set[tuple[Any, ...]] = set()
        self._prewarm_restore_kernels()

    def _make_warmup_factors(
        self,
        *,
        rows: int,
        features: int,
        rank: int,
    ) -> QuantizedSVDFactors:
        """Create shape-correct synthetic device factors for Triton JIT warmup."""

        packed_u = math.ceil(rank * self.config.bits_u / 8)
        packed_r = math.ceil(rank * self.config.bits_r / 8)
        if self.config.scale_mode == "matrix":
            u_scale_shape = (1, 1)
            r_scale_shape = (1, 1)
            group_u = None
            group_r = None
        else:
            assert self.config.group_size_u is not None
            assert self.config.group_size_r is not None
            group_u = self.config.group_size_u
            group_r = self.config.group_size_r
            u_scale_shape = (rows, math.ceil(rank / group_u))
            r_scale_shape = (features, math.ceil(rank / group_r))
        device = self._restore_device
        return QuantizedSVDFactors(
            u=QuantizedFactor(
                qdata=torch.zeros((rows, packed_u), dtype=torch.uint8, device=device),
                scale=torch.ones(u_scale_shape, dtype=torch.float16, device=device),
                bits=self.config.bits_u,
                rows=rows,
                rank=rank,
                scale_mode=self.config.scale_mode,
                group_size=group_u,
            ),
            sigma=torch.ones((rank,), dtype=torch.float16, device=device),
            right=QuantizedFactor(
                qdata=torch.zeros(
                    (features, packed_r), dtype=torch.uint8, device=device
                ),
                scale=torch.ones(r_scale_shape, dtype=torch.float16, device=device),
                bits=self.config.bits_r,
                rows=features,
                rank=rank,
                scale_mode=self.config.scale_mode,
                group_size=group_r,
            ),
        )

    def _make_batched_tma_warmup_factors(
        self,
        *,
        rows: int,
        features: int,
        rank: int,
        chunk_count: int = 2,
    ) -> tuple[tuple[QuantizedSVDFactors, ...], torch.Tensor]:
        """Build serialized-layout-like factors over one padded CUDA slab."""

        if rank != 128 or self.config.bits_u != 4 or self.config.bits_r != 4:
            raise ValueError("batched TMA warmup requires rank-128 INT4 factors")
        if self.config.scale_mode == "matrix":
            u_scale_shape = (1, 1)
            r_scale_shape = (1, 1)
            group_u = None
            group_r = None
        else:
            assert self.config.group_size_u is not None
            assert self.config.group_size_r is not None
            group_u = self.config.group_size_u
            group_r = self.config.group_size_r
            u_scale_shape = (rows, math.ceil(rank / group_u))
            r_scale_shape = (features, math.ceil(rank / group_r))

        specs = (
            ("u_qdata", (rows, 64), torch.uint8),
            ("u_scale", u_scale_shape, torch.float16),
            ("sigma", (rank,), torch.float16),
            ("right_qdata", (features, 64), torch.uint8),
            ("right_scale", r_scale_shape, torch.float16),
        )
        offsets = {}
        cursor = 0
        for name, shape, dtype in specs:
            cursor = _align_restore_workspace_nbytes(cursor)
            offsets[name] = cursor
            cursor += math.prod(shape) * dtype.itemsize
        blob_stride = _align_restore_workspace_nbytes(cursor)
        slab = torch.zeros(
            chunk_count * blob_stride,
            dtype=torch.uint8,
            device=self._restore_device,
        )

        def bind(chunk_index: int, name: str, shape: tuple[int, ...], dtype):
            byte_offset = chunk_index * blob_stride + offsets[name]
            nbytes = math.prod(shape) * dtype.itemsize
            view = slab.narrow(0, byte_offset, nbytes)
            if dtype != torch.uint8:
                view = view.view(dtype)
            return view.reshape(shape)

        factors = []
        for chunk_index in range(chunk_count):
            tensors = {
                name: bind(chunk_index, name, shape, dtype)
                for name, shape, dtype in specs
            }
            factors.append(
                QuantizedSVDFactors(
                    u=QuantizedFactor(
                        qdata=tensors["u_qdata"],
                        scale=tensors["u_scale"],
                        bits=4,
                        rows=rows,
                        rank=rank,
                        scale_mode=self.config.scale_mode,
                        group_size=group_u,
                    ),
                    sigma=tensors["sigma"],
                    right=QuantizedFactor(
                        qdata=tensors["right_qdata"],
                        scale=tensors["right_scale"],
                        bits=4,
                        rows=features,
                        rank=rank,
                        scale_mode=self.config.scale_mode,
                        group_size=group_r,
                    ),
                )
            )
        return tuple(factors), slab

    def _prewarm_restore_kernels(self) -> None:
        """Compile all K/V restore specializations before the first cache hit."""

        if self._restore_device.type != "cuda" or not fused_svd_restore_available():
            return
        assert self._load_stream is not None
        first_layer = self.buffer_layer_indices[0]
        key_buffer = self.device_pool.k_buffer[first_layer]
        value_buffer = self.device_pool.v_buffer[first_layer]
        contracts = (
            (
                int(key_buffer.shape[1] * key_buffer.shape[2]),
                self.config.key_rank,
                key_buffer.dtype,
            ),
            (
                int(value_buffer.shape[1] * value_buffer.shape[2]),
                self.config.value_rank,
                value_buffer.dtype,
            ),
        )
        seen = set()
        retained_allocations = []
        with (
            torch.inference_mode(),
            torch.cuda.device(self._restore_device),
            torch.cuda.stream(self._load_stream),
        ):
            slots = torch.arange(
                self.config.chunk_tokens,
                dtype=torch.int64,
                device=self._restore_device,
            )
            for features, rank, output_dtype in contracts:
                signature = (features, rank, output_dtype)
                if signature in seen:
                    continue
                seen.add(signature)
                factors = self._make_warmup_factors(
                    rows=self.config.chunk_tokens,
                    features=features,
                    rank=rank,
                )
                destination = torch.empty(
                    (self.config.chunk_tokens, features),
                    dtype=output_dtype,
                    device=self._restore_device,
                )
                fused_reconstruct_svd_into(
                    factors,
                    slots,
                    destination,
                    compute_mode=self.config.restore_compute,
                    use_tma=self.config.restore_io == "tma",
                )
                if self.config.restore_io == "tma":
                    batch_signature = (
                        features,
                        self.config.scale_mode,
                        self.config.group_size_u,
                        self.config.group_size_r,
                        destination.dtype,
                    )
                    if batch_signature not in self._batch_tma_warm_signatures:
                        batch_factors, slab = self._make_batched_tma_warmup_factors(
                            rows=self.config.chunk_tokens,
                            features=features,
                            rank=rank,
                        )
                        batch_slots = torch.arange(
                            2 * self.config.chunk_tokens,
                            dtype=torch.int64,
                            device=self._restore_device,
                        )
                        batch_destination = torch.empty(
                            (2 * self.config.chunk_tokens, features),
                            dtype=destination.dtype,
                            device=self._restore_device,
                        )
                        for first_row_start in _BATCH_TMA_PREWARM_FIRST_ROWS:
                            restored_rows = (
                                2 * self.config.chunk_tokens - first_row_start
                            )
                            if not fused_reconstruct_svd_batch_into(
                                batch_factors,
                                batch_slots[:restored_rows],
                                batch_destination,
                                first_row_start=first_row_start,
                                compute_mode=self.config.restore_compute,
                            ):
                                raise RuntimeError(
                                    "synthetic batched TMA warmup violated its "
                                    "layout contract"
                                )
                        self._batch_tma_warm_signatures.add(batch_signature)
                        retained_allocations.extend(
                            (slab, batch_slots, batch_destination)
                        )
        self._load_stream.synchronize()

    def _ensure_restore_workspace(
        self,
        producer_index: int,
        required_nbytes: int,
    ) -> torch.Tensor:
        """Return one producer-ring slab, growing it outside steady-state hits."""

        if self._restore_device.type != "cuda":
            raise RuntimeError("restore workspaces are CUDA-only")
        if required_nbytes < 1:
            raise ValueError("restore workspace size must be positive")
        with self._restore_workspace_lock:
            workspace = self._load_workspaces[producer_index]
            if workspace is None or workspace.numel() < required_nbytes:
                capacity = required_nbytes
                if workspace is not None:
                    capacity = max(capacity, workspace.numel() * 2)
                workspace = torch.empty(
                    (capacity,),
                    dtype=torch.uint8,
                    device=self._restore_device,
                )
                self._load_workspaces[producer_index] = workspace
                self._load_workspace_mappings[producer_index].clear()
            return workspace.narrow(0, 0, required_nbytes)

    def _map_restore_workspace_blob(
        self,
        producer_index: int,
        parsed: ParsedSVDChunkBlob,
        device_blob: torch.Tensor,
    ) -> Any:
        """Reuse factor-view objects when a producer slab layout repeats."""

        # The connector validates one immutable serialization/shape contract
        # for every admitted entry.  Within that contract, byte length and slab
        # offset uniquely determine the factor layout; rebuilding a 360-view
        # tuple on every warm hit was measurable scheduler work.
        layout_key = (
            int(device_blob.storage_offset()),
            int(device_blob.numel()),
        )
        with self._restore_workspace_lock:
            mappings = self._load_workspace_mappings[producer_index]
            mapped = mappings.get(layout_key)
            if mapped is None:
                mapped = parsed.map_blob(device_blob)
                mappings[layout_key] = mapped
            return mapped

    def _preallocate_restore_workspaces(self, blob_nbytes: int) -> None:
        """Provision one-chunk slabs when compression/prefetch reveals its size."""

        if self._restore_device.type != "cuda":
            return
        required_nbytes = _align_restore_workspace_nbytes(blob_nbytes)
        with torch.cuda.device(self._restore_device):
            for producer_index in range(len(self._load_workspaces)):
                self._ensure_restore_workspace(producer_index, required_nbytes)

    def _prewarm_restore_mappings(
        self,
        parsed: ParsedSVDChunkBlob,
        *,
        contiguous_chunks: int = 1,
    ) -> bool:
        """Best-effort build of bounded prefix slabs and mappings before TTFT."""

        if self._restore_device.type != "cuda":
            return False
        if contiguous_chunks < 1:
            raise ValueError("contiguous_chunks must be positive")
        blob_nbytes = int(parsed.blob.numel())
        blob_stride = _align_restore_workspace_nbytes(blob_nbytes)
        required_nbytes = blob_stride * contiguous_chunks
        try:
            with torch.cuda.device(self._restore_device):
                first_ring_chunks = []
                for producer_index in range(len(self._load_workspaces)):
                    workspace = self._ensure_restore_workspace(
                        producer_index, required_nbytes
                    )
                    for chunk_index in range(contiguous_chunks):
                        device_blob = workspace.narrow(
                            0,
                            chunk_index * blob_stride,
                            blob_nbytes,
                        )
                        mapped = self._map_restore_workspace_blob(
                            producer_index,
                            parsed,
                            device_blob,
                        )
                        if producer_index == 0:
                            first_ring_chunks.append(mapped)
                self._prewarm_batched_tma_restore(tuple(first_ring_chunks))
        except torch.OutOfMemoryError:
            # Prewarming is an optimization, not an L2-admission requirement.
            # Any successfully grown rings remain valid; missing mappings and
            # capacity are constructed lazily by start_loading().
            logger.warning(
                "Skipping SVD restore prewarm for %d chunks (%d bytes per ring) "
                "after a CUDA allocation failure",
                contiguous_chunks,
                required_nbytes,
            )
            return False
        return True

    def _prewarm_batched_tma_restore(self, mapped_chunks: Sequence[Any]) -> None:
        """Compile 3-D TMA K/V variants while prefix write-through is off-TTFT."""

        config = getattr(self, "config", None)
        if config is None or config.restore_io != "tma" or len(mapped_chunks) < 2:
            return
        first_layer = self.buffer_layer_indices[0]
        variants = (
            (
                tuple(mapped.chunk.key_layers[0] for mapped in mapped_chunks),
                self.device_pool.k_buffer[first_layer],
            ),
            (
                tuple(mapped.chunk.value_layers[0] for mapped in mapped_chunks),
                self.device_pool.v_buffer[first_layer],
            ),
        )
        launched = False
        assert self._write_stream is not None
        with torch.cuda.stream(self._write_stream):
            for factors, buffer in variants:
                first = factors[0]
                signature = (
                    first.feature_dim,
                    first.u.scale_mode,
                    first.u.group_size,
                    first.right.group_size,
                    buffer.dtype,
                )
                if signature in self._batch_tma_warm_signatures:
                    continue
                restored_rows = len(factors) * first.chunk_tokens
                slots = torch.arange(
                    restored_rows,
                    dtype=torch.int64,
                    device=self._restore_device,
                )
                destination = torch.empty(
                    (restored_rows, *buffer.shape[1:]),
                    dtype=buffer.dtype,
                    device=self._restore_device,
                )
                if fused_reconstruct_svd_batch_into(
                    factors,
                    slots,
                    destination,
                    compute_mode=config.restore_compute,
                ):
                    self._batch_tma_warm_signatures.add(signature)
                    launched = True
        if launched:
            # This runs during compression, never on cache-hit TTFT, and keeps
            # temporary slots/output alive through the asynchronous kernels.
            self._write_stream.synchronize()

    def _record_stats(self, **increments: int | float) -> None:
        should_log = False
        with self._stats_lock:
            for name, increment in increments.items():
                setattr(self._stats, name, getattr(self._stats, name) + increment)
            activity = self._stats.encoded_chunks + self._stats.restored_chunks
            if activity >= self._last_logged_activity + self.config.log_every:
                self._last_logged_activity = activity
                should_log = True
        if should_log:
            snapshot = self.metrics_snapshot()
            logger.info(
                "HiCache SVD chunks encoded=%d restored=%d raw_bytes=%d "
                "encoded_bytes=%d ratio=%.2fx restore_compute=%s restore_io=%s "
                "fp8_launches=%d tma_launches=%d l2_hits=%d l3_hits=%d "
                "misses=%d invalid_l2=%d invalid_l3=%d "
                "restore_h2d_ms=%.3f restore_gpu_ms=%.3f",
                snapshot["encoded_chunks"],
                snapshot["restored_chunks"],
                snapshot["raw_nbytes"],
                snapshot["encoded_nbytes"],
                snapshot["compression_ratio"],
                self.config.restore_compute,
                self.config.restore_io,
                snapshot["restore_fp8_kernel_launches"],
                snapshot["restore_tma_kernel_launches"],
                snapshot["l2_hits"],
                snapshot["l3_hits"],
                snapshot["lookup_misses"],
                snapshot["invalid_l2_blobs"],
                snapshot["invalid_l3_blobs"],
                snapshot["restore_h2d_ms"] / max(snapshot["restore_timed_batches"], 1),
                snapshot["restore_gpu_ms"] / max(snapshot["restore_timed_batches"], 1),
            )

    def metrics_snapshot(self) -> dict[str, int | float]:
        with self._stats_lock:
            snapshot = self._stats.snapshot()
        with self._pending_l3_lock:
            snapshot["pending_l3_writes"] = len(self._pending_l3)
        if self._l3_tasks is not None:
            l3_stats = self._l3_tasks.stats_snapshot()
            snapshot["l3_pending_tasks"] = l3_stats.pending_tasks
            snapshot["l3_read_requests_coalesced"] = l3_stats.read_requests_coalesced
            snapshot["l3_put_requests_coalesced"] = l3_stats.put_requests_coalesced
        return snapshot

    def _queue_l3_retry(self, key: str, *, record_retry: bool = True) -> bool:
        """Pin an L2 blob until its durable L3 publication is acknowledged."""

        lease = self.pool.acquire((key,))
        if len(lease) != 1:
            lease.release()
            return False
        with self._pending_l3_lock:
            if key in self._pending_l3:
                keep = False
            else:
                self._pending_l3[key] = _PendingL3Write(
                    lease=lease,
                    had_failure=record_retry,
                )
                keep = True
        if keep and record_retry:
            self._record_stats(l3_retry_queued=1)
        if not keep:
            lease.release()
        return keep

    def _complete_l3_retry(self, key: str, *, retried: bool = False) -> None:
        with self._pending_l3_lock:
            state = self._pending_l3.pop(key, None)
        if state is not None:
            state.lease.release()
            if retried and state.had_failure:
                self._record_stats(l3_retry_succeeded=1)

    def _release_l3_retries(self) -> None:
        with self._pending_l3_lock:
            states = tuple(self._pending_l3.values())
            self._pending_l3.clear()
        for state in states:
            state.lease.release()

    def _defer_l3_retry(self, key: str, state: _PendingL3Write) -> None:
        """Move a failed retry to the queue tail for bounded round-robin drain."""

        with self._pending_l3_lock:
            if self._pending_l3.get(key) is state:
                self._pending_l3.pop(key)
                self._pending_l3[key] = state

    def _fail_l3_attempt(self, key: str) -> None:
        """Retain a retryable source or release it after the configured bound."""

        exhausted = None
        retryable = False
        with self._pending_l3_lock:
            state = self._pending_l3.get(key)
            if state is not None:
                state.had_failure = True
                if state.attempts >= self.config.l3_retry_limit:
                    exhausted = self._pending_l3.pop(key)
                else:
                    retryable = True
        if retryable:
            self._record_stats(l3_retry_queued=1)
            return
        if exhausted is None:
            return
        exhausted.lease.release()
        self._record_stats(l3_retry_exhausted=1)
        logger.error(
            "dropping durable SVD L3 publication for %s after %d attempts",
            key,
            exhausted.attempts,
        )

    def _schedule_l3_put(
        self,
        key: str,
        blob: torch.Tensor,
        *,
        retry_from_l2: bool,
        wait_for_ack: bool = False,
    ) -> bool:
        """Submit one immutable publication and account for its eventual ACK."""

        if self._l3_tasks is None:
            return False
        if retry_from_l2:
            # Pin before submission, not only after failure.  Otherwise a fast
            # sequence of L2 admissions can evict the sole compressed copy
            # while its L3 write is still in flight.
            if not self._queue_l3_retry(key, record_retry=False):
                with self._pending_l3_lock:
                    if key not in self._pending_l3:
                        return False
            with self._pending_l3_lock:
                if key in self._pending_l3_puts:
                    return True
                state = self._pending_l3.get(key)
                if state is None:
                    return False
                if state.attempts >= self.config.l3_retry_limit:
                    return False
                state.attempts += 1
                self._pending_l3_puts.add(key)
        future = self._l3_tasks.submit_put(key, blob)

        def completed(done: Future) -> None:
            if retry_from_l2:
                with self._pending_l3_lock:
                    self._pending_l3_puts.discard(key)
            try:
                accepted = bool(done.result())
            except Exception:  # noqa: BLE001
                accepted = False
                self._record_stats(l3_io_errors=1)
                logger.warning(
                    "background SVD L3 put failed for %s", key, exc_info=True
                )
            self._record_stats(
                l3_puts=int(accepted),
                l3_rejections=int(not accepted),
            )
            if accepted:
                self._complete_l3_retry(key, retried=retry_from_l2)
            elif retry_from_l2:
                self._fail_l3_attempt(key)

        future.add_done_callback(completed)
        if wait_for_ack:
            try:
                return bool(future.result())
            except Exception:  # noqa: BLE001
                return False
        if not future.done():
            return True
        try:
            return bool(future.result())
        except Exception:  # noqa: BLE001
            return False

    def check_events(self) -> None:
        """Retry failed L3 publications from pinned, byte-accounted L2 blobs."""

        self._reap_loads()
        if self.storage is None:
            return
        with self._pending_l3_lock:
            pending = []
            for item in self._pending_l3.items():
                pending.append(item)
                if len(pending) == self.MAX_L3_RETRIES_PER_TICK:
                    break
        for key, state in pending:
            # The manager coalesces duplicate keyed puts, so scheduler ticks can
            # safely resubmit without blocking or multiplying backend work.
            self._schedule_l3_put(
                key,
                state.lease.blobs[0],
                retry_from_l2=True,
            )
            self._defer_l3_retry(key, state)

    def synchronize_source_reads(self) -> None:
        """Fence every CUDA stream that may still be reading evicted KV rows."""

        devices = {
            tensor.device
            for layer in self.buffer_layer_indices
            for tensor in (
                self.device_pool.k_buffer[layer],
                self.device_pool.v_buffer[layer],
            )
            if tensor.device.type == "cuda"
        }
        for device in devices:
            torch.cuda.synchronize(device)

    def preflight_encoder(self) -> None:
        """Validate optional encoder dependencies before serving requests."""

        preflight = getattr(self.encoder, "preflight", None)
        if callable(preflight):
            preflight()

    def _raw_chunk_nbytes(self) -> int:
        per_token = 0
        for layer in self.buffer_layer_indices:
            key = self.device_pool.k_buffer[layer]
            value = self.device_pool.v_buffer[layer]
            per_token += (
                key.shape[1] * key.shape[2] * key.element_size()
                + value.shape[1] * value.shape[2] * value.element_size()
            )
        return self.config.chunk_tokens * per_token

    def _validate_device_contract(self) -> None:
        if getattr(self.device_pool, "kv_cache_layout", "nhd") != "nhd":
            raise ValueError("SVD chunk connector requires an NHD device KV layout")
        if getattr(self.device_pool, "is_quantized_kv_cache", False):
            raise ValueError("quantized device KV caches are not supported")
        keys = getattr(self.device_pool, "k_buffer", None)
        values = getattr(self.device_pool, "v_buffer", None)
        if not keys or not values or len(keys) != len(values):
            raise ValueError("device_pool must expose equal nonempty K/V layer buffers")
        if not self.buffer_layer_indices:
            raise ValueError("buffer_layer_indices must not be empty")
        if len(set(self.buffer_layer_indices)) != len(self.buffer_layer_indices):
            raise ValueError("buffer_layer_indices must not contain duplicates")
        for layer in self.buffer_layer_indices:
            if isinstance(layer, bool) or not isinstance(layer, int):
                raise TypeError("buffer_layer_indices must contain integers")
            if not 0 <= layer < len(keys):
                raise ValueError("buffer_layer_indices contains an invalid layer")
            for tensor in (keys[layer], values[layer]):
                if tensor.ndim != 3 or not tensor.is_contiguous():
                    raise ValueError(
                        "device KV buffers must be contiguous [slots, heads, dim]"
                    )
                if not tensor.is_floating_point():
                    raise TypeError("device KV buffers must have a floating dtype")
                if tensor.shape[0] % self.page_size:
                    raise ValueError("device slot count must be divisible by page_size")
        first_key = keys[self.buffer_layer_indices[0]]
        first_value = values[self.buffer_layer_indices[0]]
        for layer in self.buffer_layer_indices:
            if (
                keys[layer].shape != first_key.shape
                or keys[layer].dtype != first_key.dtype
                or keys[layer].device != first_key.device
            ):
                raise ValueError("all selected K layers must share one tensor contract")
            if (
                values[layer].shape != first_value.shape
                or values[layer].dtype != first_value.dtype
                or values[layer].device != first_value.device
            ):
                raise ValueError("all selected V layers must share one tensor contract")
        if first_key.shape[0] != first_value.shape[0]:
            raise ValueError("K and V buffers must have the same slot count")
        if first_key.shape[1] != first_value.shape[1]:
            raise ValueError("K and V must have the same local KV-head count")
        key_features = int(first_key.shape[1] * first_key.shape[2])
        value_features = int(first_value.shape[1] * first_value.shape[2])
        if self.config.chunk_tokens < max(key_features, value_features):
            raise ValueError("direct paged SVD requires chunk_tokens >= H*D")
        if self.config.key_rank > min(self.config.chunk_tokens, key_features):
            raise ValueError("key_rank exceeds the K matrix dimensions")
        if self.config.value_rank > min(self.config.chunk_tokens, value_features):
            raise ValueError("value_rank exceeds the V matrix dimensions")
        if self.config.backend == "cute_bf16" and (
            key_features % 128
            or value_features % 128
            or self.config.key_rank % 8
            or self.config.value_rank % 8
        ):
            raise ValueError(
                "cute_bf16 requires H*D divisible by 128 and ranks divisible by 8"
            )

    def identities(
        self, token_ids: Sequence[int] | torch.Tensor, namespace: Any
    ) -> tuple[SVDChunkIdentity, ...]:
        return derive_svd_chunk_identities(
            token_ids,
            namespace=namespace,
            codec_metadata=self.codec_metadata,
            chunk_tokens=self.config.chunk_tokens,
        )

    def _factorize(
        self,
        pages: torch.Tensor,
        page_ids: torch.Tensor,
        *,
        rank: int,
        seed: int,
    ) -> QuantizedSVDFactors:
        factorize = getattr(self.encoder, "factorize", None)
        if factorize is None:
            if not callable(self.encoder):
                raise TypeError("encoder must be callable or provide factorize()")
            factorize = self.encoder
        result = factorize(
            pages=pages,
            page_ids=page_ids,
            rank=rank,
            niter=self.config.niter,
            backend=self.config.backend,
            precision=self.config.precision,
            seed=seed,
        )
        u, sigma, right = _normalize_factorizer_result(
            result,
            rank=rank,
            rows=int(page_ids.numel() * pages.shape[1]),
            features=int(pages.shape[2]),
            device=pages.device,
            # The production direct encoder already enforces its tensor
            # contract.  Avoid three GPU scalar fences per K/V/layer; injected
            # experimental encoders retain strict finite-value validation.
            check_finite=not isinstance(self.encoder, DirectPagedSVDEncoder),
        )
        return quantize_svd_factors(
            u,
            sigma,
            right,
            bits_u=self.config.bits_u,
            bits_r=self.config.bits_r,
            scale_mode=self.config.scale_mode,
            group_size_u=self.config.group_size_u,
            group_size_r=self.config.group_size_r,
        )

    def _factorize_batch(
        self,
        pages: torch.Tensor,
        page_ids: Sequence[torch.Tensor],
        *,
        rank: int,
        seeds: Sequence[int],
    ) -> tuple[QuantizedSVDFactors, ...]:
        """Factor independent chunks together when the encoder supports it."""

        if len(page_ids) != len(seeds):
            raise ValueError("page_ids and seeds must have the same batch size")
        factorize_batch = getattr(self.encoder, "factorize_batch", None)
        if not callable(factorize_batch) or len(page_ids) == 1:
            return tuple(
                self._factorize(pages, ids, rank=rank, seed=seed)
                for ids, seed in zip(page_ids, seeds)
            )
        results = tuple(
            factorize_batch(
                pages=pages,
                page_ids=page_ids,
                rank=rank,
                niter=self.config.niter,
                backend=self.config.backend,
                precision=self.config.precision,
                seeds=seeds,
            )
        )
        if len(results) != len(page_ids):
            raise ValueError("batched SVD encoder returned the wrong batch size")
        quantized = []
        for result, ids in zip(results, page_ids):
            u, sigma, right = _normalize_factorizer_result(
                result,
                rank=rank,
                rows=int(ids.numel() * pages.shape[1]),
                features=int(pages.shape[2]),
                device=pages.device,
                check_finite=not isinstance(self.encoder, DirectPagedSVDEncoder),
            )
            quantized.append(
                quantize_svd_factors(
                    u,
                    sigma,
                    right,
                    bits_u=self.config.bits_u,
                    bits_r=self.config.bits_r,
                    scale_mode=self.config.scale_mode,
                    group_size_u=self.config.group_size_u,
                    group_size_r=self.config.group_size_r,
                )
            )
        return tuple(quantized)

    def encode_chunks(
        self,
        identities: Sequence[SVDChunkIdentity],
        device_indices: Sequence[torch.Tensor],
        *,
        namespaces: Sequence[Any],
    ) -> tuple[torch.Tensor, ...]:
        """Encode independent prefix chunks with layer-wise SVD microbatching."""

        identities = tuple(identities)
        device_indices = tuple(device_indices)
        namespaces = tuple(namespaces)
        batch_size = len(identities)
        if batch_size == 0:
            raise ValueError("encode_chunks requires at least one chunk")
        if len(device_indices) != batch_size or len(namespaces) != batch_size:
            raise ValueError("batched identities, indices, and namespaces must align")
        if batch_size > self.config.encode_batch_size:
            raise ValueError(
                f"encode batch size {batch_size} exceeds configured bound "
                f"{self.config.encode_batch_size}"
            )
        for indices in device_indices:
            if indices.numel() != self.config.chunk_tokens:
                raise ValueError(
                    "every device_indices entry must contain one full chunk"
                )

        wall_started = time.perf_counter()
        page_ids = tuple(
            page_ids_from_device_indices(indices, page_size=self.page_size).to(
                device=indices.device,
                dtype=torch.int64,
            )
            for indices in device_indices
        )
        key_layers = [[] for _ in identities]
        value_layers = [[] for _ in identities]
        chunk_session = getattr(self.encoder, "chunk_session", None)
        session = chunk_session() if callable(chunk_session) else nullcontext()
        with torch.inference_mode(), session:
            for layer in self.buffer_layer_indices:
                key_pages = view_nhd_as_joint_pages(
                    self.device_pool.k_buffer[layer], page_size=self.page_size
                )
                value_pages = view_nhd_as_joint_pages(
                    self.device_pool.v_buffer[layer], page_size=self.page_size
                )
                key_batch = self._factorize_batch(
                    key_pages,
                    page_ids,
                    rank=self.config.key_rank,
                    seeds=tuple(
                        _matrix_seed(self.config.seed, identity, layer, "K")
                        for identity in identities
                    ),
                )
                value_batch = self._factorize_batch(
                    value_pages,
                    page_ids,
                    rank=self.config.value_rank,
                    seeds=tuple(
                        _matrix_seed(self.config.seed, identity, layer, "V")
                        for identity in identities
                    ),
                )
                for batch_index in range(batch_size):
                    key_layers[batch_index].append(key_batch[batch_index])
                    value_layers[batch_index].append(value_batch[batch_index])

        blobs = []
        for batch_index, identity in enumerate(identities):
            chunk = QuantizedKVChunk.from_sequences(
                key_layers[batch_index], value_layers[batch_index]
            )
            blob = serialize_svd_chunk(
                chunk,
                codec_metadata=self.codec_metadata,
                namespace=namespaces[batch_index],
                end_page_hash=identity.prefix_hash,
            )
            expected_key = make_svd_chunk_key(
                codec_metadata=self.codec_metadata,
                namespace=namespaces[batch_index],
                end_page_hash=identity.prefix_hash,
            )
            if identity.key != expected_key:
                raise ValueError(
                    "chunk identity does not match namespace/codec metadata"
                )
            blobs.append(blob)

        # Provision the restore ring once at the common fixed blob size.
        self._preallocate_restore_workspaces(max(blob.numel() for blob in blobs))
        self._record_stats(
            encode_batches=1,
            batched_encoded_chunks=batch_size if batch_size > 1 else 0,
            encoded_chunks=batch_size,
            encoded_tokens=batch_size * self.config.chunk_tokens,
            raw_nbytes=batch_size * self._raw_chunk_nbytes(),
            encoded_nbytes=sum(blob.numel() * blob.element_size() for blob in blobs),
            encode_wall_ms=(time.perf_counter() - wall_started) * 1000,
        )
        return tuple(blobs)

    def encode_chunk(
        self,
        identity: SVDChunkIdentity,
        device_indices: torch.Tensor,
        *,
        namespace: Any,
    ) -> torch.Tensor:
        """Encode one page-aligned physical L1 selection into a canonical blob."""
        return self.encode_chunks(
            (identity,),
            (device_indices,),
            namespaces=(namespace,),
        )[0]

    def _acquire_pool_blob(self, key: str) -> tuple[Optional[torch.Tensor], Any]:
        lease = self.pool.acquire((key,))
        if len(lease) == 1:
            return lease.blobs[0], lease
        lease.release()
        return None, None

    def _admit_parsed(self, key: str, parsed: ParsedSVDChunkBlob) -> bool:
        put_parsed = getattr(self.pool, "put_parsed", None)
        if callable(put_parsed):
            return bool(put_parsed(key, parsed, contract_validated=True))
        return bool(self.pool.put(key, parsed.blob))

    def _acquire_pool_chunk(self, key: str) -> tuple[Optional[ParsedSVDChunkBlob], Any]:
        """Acquire one L2 chunk and reuse its cached parse when supported."""

        acquire_parsed = getattr(self.pool, "acquire_parsed", None)
        if callable(acquire_parsed):
            lease = acquire_parsed(
                (key,),
                expected_codec_metadata=self.codec_metadata,
                validator=self._validate_chunk_contract,
            )
            if len(lease) == 1:
                chunk = lease.parsed_chunks[0]
                return chunk, lease
            lease.release()
            return None, None

        blob, lease = self._acquire_pool_blob(key)
        if blob is None:
            return None, None
        try:
            return self._validated_blob(key, blob), lease
        except Exception:
            lease.release()
            raise

    def _validated_blob(self, key: str, blob: torch.Tensor) -> ParsedSVDChunkBlob:
        _validate_blob(blob)
        parsed = parse_svd_chunk_blob(
            blob,
            expected_key=key,
            expected_codec_metadata=self.codec_metadata,
            copy_blob=False,
        )
        self._validate_chunk_contract(parsed.chunk)
        return parsed

    def _validate_chunk_contract(self, chunk: QuantizedKVChunk) -> None:
        """Cross-check decoded factor shapes against the live KV pool/config."""

        if chunk.chunk_tokens != self.config.chunk_tokens:
            raise ValueError("stored chunk token geometry does not match connector")
        if chunk.layer_count != len(self.buffer_layer_indices):
            raise ValueError("stored chunk layer count does not match device pool")

        def validate_factors(
            factors: QuantizedSVDFactors,
            *,
            features: int,
            rank: int,
            label: str,
        ) -> None:
            expected_group_u = (
                None if self.config.scale_mode == "matrix" else self.config.group_size_u
            )
            expected_group_r = (
                None if self.config.scale_mode == "matrix" else self.config.group_size_r
            )
            actual = (
                factors.chunk_tokens,
                factors.feature_dim,
                factors.rank,
                factors.u.bits,
                factors.right.bits,
                factors.u.scale_mode,
                factors.right.scale_mode,
                factors.u.group_size,
                factors.right.group_size,
            )
            expected = (
                self.config.chunk_tokens,
                features,
                rank,
                self.config.bits_u,
                self.config.bits_r,
                self.config.scale_mode,
                self.config.scale_mode,
                expected_group_u,
                expected_group_r,
            )
            if actual != expected:
                raise ValueError(
                    f"stored {label} factor contract does not match connector"
                )
            if not all(
                bool(torch.isfinite(tensor).all())
                for tensor in (factors.u.scale, factors.sigma, factors.right.scale)
            ):
                raise ValueError(f"stored {label} factors contain non-finite values")

        for encoded_layer, buffer_layer in enumerate(self.buffer_layer_indices):
            key_buffer = self.device_pool.k_buffer[buffer_layer]
            value_buffer = self.device_pool.v_buffer[buffer_layer]
            validate_factors(
                chunk.key_layers[encoded_layer],
                features=int(key_buffer.shape[1] * key_buffer.shape[2]),
                rank=self.config.key_rank,
                label="K",
            )
            validate_factors(
                chunk.value_layers[encoded_layer],
                features=int(value_buffer.shape[1] * value_buffer.shape[2]),
                rank=self.config.value_rank,
                label="V",
            )

    def _read_l3_blob(self, key: str) -> tuple[Optional[ParsedSVDChunkBlob], bool]:
        """Return a validated L3 chunk and whether a missing entry is writable."""

        if self.storage is None:
            return None, False
        try:
            exists = self.storage.exists(key)
        except Exception:  # noqa: BLE001
            logger.warning("SVD L3 exists() failed for %s", key, exc_info=True)
            self._record_stats(l3_io_errors=1)
            return None, False
        if not exists:
            return None, True
        try:
            blob = self.storage.get(key)
        except Exception:  # noqa: BLE001
            logger.warning("SVD L3 get() failed for %s", key, exc_info=True)
            self._record_stats(l3_io_errors=1)
            return None, False
        if blob is None:
            # The entry may have disappeared after exists(); a new immutable
            # write is safe.
            return None, True
        try:
            parsed = self._validated_blob(key, blob)
        except Exception:  # noqa: BLE001
            self._record_stats(invalid_l3_blobs=1)
            try:
                deleted = self.storage.delete(key)
            except Exception:  # noqa: BLE001
                logger.warning("SVD L3 delete() failed for %s", key, exc_info=True)
                self._record_stats(l3_io_errors=1)
                deleted = False
            return None, deleted
        self._record_stats(l3_hits=1)
        return parsed, False

    def _validate_prefetched_l3_blob(
        self,
        key: str,
        blob: Optional[torch.Tensor],
    ) -> Optional[ParsedSVDChunkBlob]:
        """Parse/checksum an L3 read on its worker, never on the scheduler."""

        if blob is None:
            return None
        try:
            return self._validated_blob(key, blob)
        except Exception:  # noqa: BLE001
            self._record_stats(invalid_l3_blobs=1)
            try:
                self.storage.delete(key)
            except Exception:  # noqa: BLE001
                logger.warning("SVD L3 delete() failed for %s", key, exc_info=True)
                self._record_stats(l3_io_errors=1)
            return None

    def _load_for_marker(
        self,
        key: str,
        *,
        allow_l3: bool,
        prefetched: Optional[dict[str, ParsedSVDChunkBlob]] = None,
    ) -> tuple[Optional[ParsedSVDChunkBlob], Any]:
        invalid_l2 = False
        try:
            parsed, lease = self._acquire_pool_chunk(key)
        except Exception:
            invalid_l2 = True
            self._record_stats(invalid_l2_blobs=1)
        else:
            if parsed is not None:
                self._record_stats(l2_hits=1)
                return parsed, lease
        if prefetched is not None:
            parsed = prefetched.get(key)
            if parsed is not None:
                self._record_stats(l3_hits=1)
                return parsed, None
        if self.storage is None or not allow_l3:
            self._record_stats(lookup_misses=1)
            return None, None
        parsed, _ = self._read_l3_blob(key)
        if parsed is None:
            self._record_stats(lookup_misses=1)
            return None, None
        # A corrupt immutable entry cannot be replaced through the pool's
        # content-binding API.  Keep the validated L3 bytes marker-owned for
        # this request and allow normal LRU eviction to remove the poison later.
        if not invalid_l2:
            if self._admit_parsed(key, parsed):
                self._record_stats(l2_puts=1)
                pooled, lease = self._acquire_pool_chunk(key)
                if pooled is not None:
                    return pooled, lease
            else:
                self._record_stats(l2_rejections=1)
        # A chunk larger than L2, or a temporarily fully-pinned L2, can still be
        # restored from this marker-owned L3 tensor.  This is bounded by the
        # number of chunks in a single in-flight lookup.
        return parsed, None

    def lookup(
        self,
        *,
        token_ids: Sequence[int] | torch.Tensor,
        namespace: Any,
        device_len: int,
        rid: Optional[str] = None,
        allow_l3: bool = True,
    ) -> Optional[LoadMarker]:
        """Acquire the longest complete compressed suffix following an L1 hit."""

        self._record_stats(lookup_calls=1)
        _require_plain_int("device_len", device_len, minimum=0)
        tokens = _normalize_token_ids(token_ids)
        if device_len > len(tokens):
            raise ValueError("device_len cannot exceed token_ids length")
        identities = self.identities(tokens, namespace)
        first_chunk = device_len // self.config.chunk_tokens
        if first_chunk >= len(identities):
            return None

        keys = []
        chunks = []
        leases = []
        with self._prefetch_lock:
            prefetched = self._prefetched_chunks.get(rid) if rid is not None else None
        for identity in identities[first_chunk:]:
            chunk, lease = self._load_for_marker(
                identity.key,
                allow_l3=allow_l3,
                prefetched=prefetched,
            )
            if chunk is None:
                break
            keys.append(identity.key)
            chunks.append(chunk)
            if lease is not None:
                leases.append(lease)
        if not chunks:
            return None
        self._record_stats(lookup_hit_chunks=len(chunks))
        matched_end = (first_chunk + len(chunks)) * self.config.chunk_tokens
        marker = LoadMarker(
            rid=rid,
            restore_start=device_len,
            matched_end=matched_end,
            first_chunk_index=first_chunk,
            chunk_tokens=self.config.chunk_tokens,
            chunk_keys=tuple(keys),
            chunks=tuple(chunks),
            _leases=tuple(leases),
        )
        if rid is not None:
            with self._prefetch_lock:
                self._prefetched_chunks.pop(rid, None)
                self._prefetch_loaded_tokens.pop(rid, None)
            with self._pending_lock:
                previous = self._pending.pop(rid, None)
                self._pending[rid] = marker
            if previous is not None:
                previous.release()
        return marker

    def prefetch(
        self,
        *,
        token_ids: Sequence[int] | torch.Tensor,
        namespace: Any,
        device_len: int,
        rid: str,
    ) -> None:
        """Start coalesced L3 reads while a request waits in the scheduler."""

        if self._l3_tasks is None:
            return
        _require_plain_int("device_len", device_len, minimum=0)
        if not isinstance(rid, str) or not rid:
            raise ValueError("rid must be a nonempty string")
        identities = self.identities(token_ids, namespace)
        first_chunk = device_len // self.config.chunk_tokens
        selected = identities[first_chunk:]
        if not selected:
            return
        self.cancel_prefetch(rid)
        keys = tuple(identity.key for identity in selected)
        futures = tuple(self._l3_tasks.submit_read(key) for key in keys)
        state = _L3Prefetch(
            rid=rid,
            first_chunk_index=first_chunk,
            keys=keys,
            futures=futures,
        )
        with self._prefetch_lock:
            self._prefetches[rid] = state

    def check_prefetch_progress(self, rid: str) -> bool:
        """Admit a completed contiguous L3 suffix without blocking on I/O."""

        with self._prefetch_lock:
            state = self._prefetches.get(rid)
        if state is None:
            return True

        ready: dict[str, ParsedSVDChunkBlob] = {}
        loaded_tokens = 0
        completed_prefix = 0
        for key, future in zip(state.keys, state.futures):
            if not future.done():
                # Waiting is useful only when no restorable prefix exists yet.
                # Once one or more leading chunks are ready, publish them and
                # detach this request from the slower tail.
                if completed_prefix == 0:
                    return False
                break
            try:
                parsed = future.result()
            except CancelledError:
                break
            except Exception:  # noqa: BLE001
                logger.warning("SVD L3 prefetch failed for %s", key, exc_info=True)
                self._record_stats(l3_io_errors=1)
                break
            if parsed is None:
                break
            if not isinstance(parsed, ParsedSVDChunkBlob):
                raise TypeError("SVD L3 read transform returned an invalid chunk")
            self._preallocate_restore_workspaces(parsed.blob.numel())
            try:
                admitted = self._admit_parsed(key, parsed)
            except ValueError:
                admitted = False
            if admitted:
                self._record_stats(l2_puts=1)
            else:
                self._record_stats(l2_rejections=1)
                if self._restore_device.type == "cuda" and not parsed.blob.is_pinned():
                    parsed = parse_svd_chunk_blob(
                        parsed.blob,
                        expected_key=key,
                        expected_codec_metadata=self.codec_metadata,
                        copy_blob=False,
                        pin_memory=True,
                    )
                ready[key] = parsed
            loaded_tokens += self.config.chunk_tokens
            completed_prefix += 1
            self._record_stats(l3_hits=1)

        with self._prefetch_lock:
            if self._prefetches.get(rid) is state:
                self._prefetches.pop(rid, None)
                self._prefetched_chunks[rid] = ready
                self._prefetch_loaded_tokens[rid] = loaded_tokens
                owns_request = True
            else:
                # A replacement prefetch for the same request must remain the
                # state observed by future scheduler polls.
                owns_request = rid not in self._prefetches

        # Task-manager futures are subscriber-owned.  Cancelling this request's
        # unconsumed suffix releases its ownership without cancelling a shared
        # backend read or another request's subscriber.
        for future in state.futures[completed_prefix:]:
            future.cancel()
        return owns_request

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        with self._prefetch_lock:
            return self._prefetch_loaded_tokens.pop(rid, 0)

    def cancel_prefetch(self, rid: str) -> None:
        with self._prefetch_lock:
            state = self._prefetches.pop(rid, None)
            self._prefetched_chunks.pop(rid, None)
            self._prefetch_loaded_tokens.pop(rid, None)
        if state is not None:
            for future in state.futures:
                future.cancel()

    def _restore_blob_rows(
        self,
        *,
        parsed: ParsedSVDChunkBlob,
        row_start: int,
        row_end: int,
        destination_slots: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        first_buffer = self.device_pool.k_buffer[self.buffer_layer_indices[0]]
        if first_buffer.device.type == "cuda":
            mapped = parsed.to(
                first_buffer.device,
                non_blocking=parsed.blob.is_pinned(),
            )
            chunk = mapped.chunk
            device_blob = mapped.blob
        else:
            chunk = parsed.chunk
            device_blob = None
        rows = row_end - row_start
        if destination_slots.numel() != rows:
            raise ValueError("destination slot count does not match restored rows")
        for encoded_layer, buffer_layer in enumerate(self.buffer_layer_indices):
            key_buffer = self.device_pool.k_buffer[buffer_layer]
            value_buffer = self.device_pool.v_buffer[buffer_layer]
            reconstruct_svd_rows_into(
                chunk.key_layers[encoded_layer],
                row_start,
                row_end,
                destination_slots=destination_slots,
                destination=key_buffer,
                compute_mode=self.config.restore_compute,
                use_tma=self.config.restore_io == "tma",
            )
            reconstruct_svd_rows_into(
                chunk.value_layers[encoded_layer],
                row_start,
                row_end,
                destination_slots=destination_slots,
                destination=value_buffer,
                compute_mode=self.config.restore_compute,
                use_tma=self.config.restore_io == "tma",
            )
        return device_blob

    @staticmethod
    def _validate_retrieve_inputs(marker: LoadMarker, dest_slots: torch.Tensor) -> None:
        if not isinstance(marker, LoadMarker):
            raise TypeError("marker must be a LoadMarker")
        if marker.released:
            raise RuntimeError("LoadMarker has already been released")
        if not isinstance(dest_slots, torch.Tensor):
            raise TypeError("dest_slots must be a torch.Tensor")
        if dest_slots.ndim != 1 or dest_slots.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("dest_slots must be a one-dimensional INT32/INT64 tensor")
        if dest_slots.numel() != marker.matched_tokens:
            raise ValueError("dest_slots length must equal marker.matched_tokens")

    def enqueue_retrieve(self, marker: LoadMarker, dest_slots: torch.Tensor) -> int:
        """Queue a validated restore and return its exact token count immediately.

        CUDA work starts only in :meth:`start_loading`, allowing the scheduler
        to merge every restore admitted to the same prefill batch.  CPU test and
        fallback pools retain the synchronous behavior.
        """

        self._validate_retrieve_inputs(marker, dest_slots)
        if self._restore_device.type != "cuda":
            return self.retrieve(marker, dest_slots)
        operation = _RestoreOperation(marker=marker, destination_slots=dest_slots)
        with self._load_lock:
            self._load_queue.append(operation)
        return marker.matched_tokens

    def start_loading(self) -> int:
        """Launch one layer-pipelined restore batch on the dedicated load stream."""

        enqueue_started = time.perf_counter()
        if self._restore_device.type != "cuda":
            return -1
        with self._load_lock:
            if not self._load_queue:
                return -1
            operations = tuple(self._load_queue)
            self._load_queue.clear()

        assert self.layer_done_counter is not None
        assert self._load_stream is not None
        producer_index = self.layer_done_counter.update_producer()
        producer_event = self.layer_done_counter.events[producer_index]
        producer_event.start_event.record()
        device_blobs: list[torch.Tensor] = []
        device_slots: list[torch.Tensor] = []
        prepared = []
        total_blob_nbytes = sum(
            parsed.blob.numel()
            for operation in operations
            for parsed in operation.marker.chunks
        )
        total_workspace_nbytes = sum(
            _align_restore_workspace_nbytes(parsed.blob.numel())
            for operation in operations
            for parsed in operation.marker.chunks
        )
        timing_start_event = torch.cuda.Event(enable_timing=True)
        timing_h2d_event = torch.cuda.Event(enable_timing=True)
        timing_done_event = torch.cuda.Event(enable_timing=True)

        try:
            with (
                torch.cuda.device(self._restore_device),
                torch.cuda.stream(self._load_stream),
            ):
                producer_event.start_event.wait(self._load_stream)
                workspace = self._ensure_restore_workspace(
                    producer_index, total_workspace_nbytes
                )
                blob_offset = 0
                blob_copies = []
                for operation in operations:
                    slots = operation.destination_slots.to(
                        device=self._restore_device,
                        dtype=torch.int64,
                        non_blocking=True,
                    ).contiguous()
                    device_slots.append(slots)
                    mapped_chunks = []
                    for parsed in operation.marker.chunks:
                        blob_nbytes = parsed.blob.numel()
                        device_blob = workspace.narrow(0, blob_offset, blob_nbytes)
                        mapped = self._map_restore_workspace_blob(
                            producer_index, parsed, device_blob
                        )
                        blob_copies.append((device_blob, parsed.blob))
                        blob_offset += _align_restore_workspace_nbytes(blob_nbytes)
                        mapped_chunks.append(mapped)
                        device_blobs.append(device_blob)
                    prepared.append((operation, slots, tuple(mapped_chunks)))

                # Keep Python layout preparation outside the CUDA interval so
                # the H2D counter reports transfer work rather than host-side
                # enqueue gaps.  Locally encoded cumulative-prefix mappings
                # are built on the L2 write path.
                timing_start_event.record(self._load_stream)
                for device_blob, host_blob in blob_copies:
                    device_blob.copy_(
                        host_blob,
                        non_blocking=host_blob.is_pinned(),
                    )
                timing_h2d_event.record(self._load_stream)

                fp8_kernel_launches = 0
                tma_kernel_launches = 0
                for encoded_layer, buffer_layer in enumerate(self.buffer_layer_indices):
                    key_buffer = self.device_pool.k_buffer[buffer_layer]
                    value_buffer = self.device_pool.v_buffer[buffer_layer]
                    for operation, slots, mapped_chunks in prepared:
                        marker = operation.marker
                        first_chunk_start = (
                            marker.first_chunk_index * marker.chunk_tokens
                        )
                        first_row_start = max(
                            marker.restore_start - first_chunk_start,
                            0,
                        )
                        key_batched = False
                        value_batched = False
                        if self.config.restore_io == "tma":
                            key_batched = fused_reconstruct_svd_batch_into(
                                tuple(
                                    mapped.chunk.key_layers[encoded_layer]
                                    for mapped in mapped_chunks
                                ),
                                slots,
                                key_buffer,
                                first_row_start=first_row_start,
                                compute_mode=self.config.restore_compute,
                            )
                            value_batched = fused_reconstruct_svd_batch_into(
                                tuple(
                                    mapped.chunk.value_layers[encoded_layer]
                                    for mapped in mapped_chunks
                                ),
                                slots,
                                value_buffer,
                                first_row_start=first_row_start,
                                compute_mode=self.config.restore_compute,
                            )
                            tma_kernel_launches += int(key_batched) + int(value_batched)
                            fp8_kernel_launches += int(key_batched) + int(value_batched)

                        destination_offset = 0
                        for relative_index, mapped in enumerate(mapped_chunks):
                            chunk_index = marker.first_chunk_index + relative_index
                            chunk_start = chunk_index * marker.chunk_tokens
                            row_start = max(marker.restore_start - chunk_start, 0)
                            row_end = marker.chunk_tokens
                            rows = row_end - row_start
                            chunk_slots = slots[
                                destination_offset : destination_offset + rows
                            ]
                            if not key_batched:
                                reconstruct_svd_rows_into(
                                    mapped.chunk.key_layers[encoded_layer],
                                    row_start,
                                    row_end,
                                    destination_slots=chunk_slots,
                                    destination=key_buffer,
                                    compute_mode=self.config.restore_compute,
                                    use_tma=self.config.restore_io == "tma",
                                )
                                fp8_kernel_launches += int(
                                    self.config.restore_compute == "fp8_e4m3"
                                )
                                tma_kernel_launches += int(
                                    self.config.restore_io == "tma"
                                )
                            if not value_batched:
                                reconstruct_svd_rows_into(
                                    mapped.chunk.value_layers[encoded_layer],
                                    row_start,
                                    row_end,
                                    destination_slots=chunk_slots,
                                    destination=value_buffer,
                                    compute_mode=self.config.restore_compute,
                                    use_tma=self.config.restore_io == "tma",
                                )
                                fp8_kernel_launches += int(
                                    self.config.restore_compute == "fp8_e4m3"
                                )
                                tma_kernel_launches += int(
                                    self.config.restore_io == "tma"
                                )
                            destination_offset += rows
                    if self.config.restore_schedule == "layerwise":
                        producer_event.complete(encoded_layer)
                if self.config.restore_schedule in ("full", "synchronous"):
                    # SVD reconstruction and the model both consume SMs.
                    # Releasing layer 0 early made them contend and increased
                    # TTFT; record every gate only after the compact restore
                    # finishes so forward runs without cross-stream compute
                    # interference.  Keep layerwise mode for controlled A/Bs.
                    for encoded_layer in range(len(self.buffer_layer_indices)):
                        producer_event.complete(encoded_layer)
                timing_done_event.record(self._load_stream)
        except Exception:
            self._record_stats(restore_failures=len(operations))
            # Partial launches may still reference marker-owned pinned blobs.
            self._load_stream.synchronize()
            for operation in operations:
                self._finish_marker(operation.marker)
            raise

        batch = _RestoreBatch(
            producer_index=producer_index,
            operations=operations,
            device_blobs=tuple(device_blobs),
            device_slots=tuple(device_slots),
            finish_event=producer_event.finish_event,
            timing_start_event=timing_start_event,
            timing_h2d_event=timing_h2d_event,
            timing_done_event=timing_done_event,
        )
        with self._load_lock:
            self._inflight_loads.append(batch)
        restored_chunks = sum(len(op.marker.chunks) for op in operations)
        self._record_stats(
            restored_chunks=restored_chunks,
            restored_tokens=sum(op.marker.matched_tokens for op in operations),
            restored_blob_nbytes=total_blob_nbytes,
            restore_enqueue_cpu_ms=(time.perf_counter() - enqueue_started) * 1000,
            restore_fp8_kernel_launches=fp8_kernel_launches,
            restore_tma_kernel_launches=tma_kernel_launches,
        )
        return producer_index

    def _reap_loads(self, *, synchronize: bool = False) -> int:
        """Release marker leases whose final layer event has completed."""

        with self._load_lock:
            batches = tuple(self._inflight_loads)
        completed = []
        for batch in batches:
            if synchronize:
                batch.timing_done_event.synchronize()
                completed.append(batch)
            elif batch.timing_done_event.query():
                completed.append(batch)
        if not completed:
            return 0
        completed_ids = {id(batch) for batch in completed}
        with self._load_lock:
            self._inflight_loads = [
                batch
                for batch in self._inflight_loads
                if id(batch) not in completed_ids
            ]
        for batch in completed:
            self._record_stats(
                restore_h2d_ms=batch.timing_start_event.elapsed_time(
                    batch.timing_h2d_event
                ),
                restore_gpu_ms=batch.timing_start_event.elapsed_time(
                    batch.timing_done_event
                ),
                restore_timed_batches=1,
            )
            for operation in batch.operations:
                self._finish_marker(operation.marker)
        return len(completed)

    def retrieve(self, marker: LoadMarker, dest_slots: torch.Tensor) -> int:
        """Reconstruct a marker's row suffix into allocated L1 destination slots."""

        self._validate_retrieve_inputs(marker, dest_slots)
        if self._restore_device.type == "cuda":
            count = self.enqueue_retrieve(marker, dest_slots)
            producer_index = self.start_loading()
            if producer_index < 0:
                raise RuntimeError("queued SVD restore did not start")
            self.layer_done_counter.events[producer_index].finish_event.synchronize()
            self._reap_loads()
            return count

        destination_offset = 0
        try:
            device_blobs = []
            for relative_index, parsed in enumerate(marker.chunks):
                chunk_index = marker.first_chunk_index + relative_index
                chunk_start = chunk_index * marker.chunk_tokens
                row_start = max(marker.restore_start - chunk_start, 0)
                row_end = marker.chunk_tokens
                rows = row_end - row_start
                chunk_slots = dest_slots[destination_offset : destination_offset + rows]
                device_blob = self._restore_blob_rows(
                    parsed=parsed,
                    row_start=row_start,
                    row_end=row_end,
                    destination_slots=chunk_slots,
                )
                if device_blob is not None:
                    device_blobs.append(device_blob)
                destination_offset += rows
                self._record_stats(restored_chunks=1, restored_tokens=rows)
            return destination_offset
        except Exception:
            self._record_stats(restore_failures=1)
            raise
        finally:
            self._finish_marker(marker)

    def _finish_marker(self, marker: LoadMarker) -> None:
        if marker.rid is not None:
            with self._pending_lock:
                if self._pending.get(marker.rid) is marker:
                    self._pending.pop(marker.rid, None)
        marker.release()

    def _store_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        _background_l3: bool,
    ) -> tuple[tuple[str, ...], ...]:
        """Store independent one-chunk requests with a shared encode pass."""

        requests = tuple(requests)
        if not requests:
            return ()
        if len(requests) > self.config.encode_batch_size:
            raise ValueError("store batch exceeds configured encode_batch_size")

        # Live radix write-through submits one complete chunk per request.  Keep
        # ``store`` as the fully general fallback, while pre-encoding all absent
        # one-chunk candidates together for the common path.
        candidates: dict[str, tuple[SVDChunkIdentity, torch.Tensor, Any]] = {}
        for request in requests:
            tokens = _normalize_token_ids(request["token_ids"])
            start_token = int(request.get("start_token", 0))
            indices = request["device_indices"]
            namespace = request["namespace"]
            if (
                len(tokens) - start_token != self.config.chunk_tokens
                or start_token < 0
                or start_token % self.config.chunk_tokens
                or not isinstance(indices, torch.Tensor)
                or indices.numel() != self.config.chunk_tokens
            ):
                continue
            identity = self.identities(tokens, namespace)[-1]
            if identity.token_start != start_token or self.pool.exists(identity.key):
                continue
            candidates.setdefault(identity.key, (identity, indices, namespace))

        preencoded: dict[str, torch.Tensor] = {}
        candidate_values = tuple(candidates.values())
        for offset in range(0, len(candidate_values), self.config.encode_batch_size):
            group = candidate_values[offset : offset + self.config.encode_batch_size]
            if not group:
                continue
            blobs = self.encode_chunks(
                tuple(item[0] for item in group),
                tuple(item[1] for item in group),
                namespaces=tuple(item[2] for item in group),
            )
            preencoded.update((item[0].key, blob) for item, blob in zip(group, blobs))

        # Radix submits chunks in logical prefix order.  Admit in reverse so a
        # constrained L2 leaves the earliest contiguous chunks most-recent and
        # therefore resident, matching ``store``'s multi-chunk policy.
        results: list[tuple[str, ...]] = [() for _ in requests]
        for request_index in range(len(requests) - 1, -1, -1):
            request = requests[request_index]
            results[request_index] = self.store(
                token_ids=request["token_ids"],
                device_indices=request["device_indices"],
                namespace=request["namespace"],
                start_token=int(request.get("start_token", 0)),
                _background_l3=_background_l3,
                _preencoded_blobs=preencoded,
            )

        if self.storage is None:
            # A later request in this same batch may have displaced an earlier
            # result.  Child futures must describe final residency, not the
            # intermediate state observed when that individual store returned.
            for request_index, request in enumerate(requests):
                tokens = _normalize_token_ids(request["token_ids"])
                first_chunk = (
                    int(request.get("start_token", 0)) // self.config.chunk_tokens
                )
                results[request_index] = tuple(
                    identity.key
                    for identity in self.identities(tokens, request["namespace"])[
                        first_chunk:
                    ]
                    if self.pool.exists(identity.key)
                )
        return tuple(results)

    def store_many_async(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> tuple[Future, ...]:
        """Atomically accept a group and return one source-lifetime ACK per item.

        On any raised setup error, no background worker can subsequently read
        the supplied source slots.
        """

        normalized_items = []
        for request in requests:
            tokens = _normalize_token_ids(request["token_ids"])
            start_token = request.get("start_token", 0)
            _require_plain_int("start_token", start_token, minimum=0)
            device_indices = request["device_indices"]
            if (
                not isinstance(device_indices, torch.Tensor)
                or device_indices.ndim != 1
                or device_indices.dtype not in (torch.int32, torch.int64)
            ):
                raise TypeError(
                    "device_indices must be a one-dimensional INT32/INT64 tensor"
                )
            if not tokens or len(tokens) % self.config.chunk_tokens:
                raise ValueError("store requires a nonempty chunk-aligned token prefix")
            if start_token >= len(tokens):
                raise ValueError("start_token must select at least one complete chunk")
            if start_token % self.config.chunk_tokens:
                raise ValueError("start_token must be chunk-aligned")
            if device_indices.numel() != len(tokens) - start_token:
                raise ValueError(
                    "device_indices must cover token_ids[start_token:] exactly"
                )
            normalized_items.append(
                {
                    "token_ids": tokens,
                    "device_indices": device_indices.detach(),
                    "namespace": request["namespace"],
                    "start_token": start_token,
                }
            )
        normalized = tuple(normalized_items)
        if not normalized:
            return ()
        if len(normalized) > self.config.encode_batch_size:
            raise ValueError("store batch exceeds configured encode_batch_size")

        children = tuple(Future() for _ in normalized)
        # These futures are ownership ACKs, not cancellable convenience tasks.
        # Construct and mark all of them before submitting work: if allocation
        # fails, no encoder can have started reading authoritative L1 slots.
        for child in children:
            if not child.set_running_or_notify_cancel():  # pragma: no cover
                raise RuntimeError(
                    "new batched store future was unexpectedly cancelled"
                )

        start_event = None
        if self._restore_device.type == "cuda":
            start_event = torch.cuda.Event()
            start_event.record(torch.cuda.current_stream(self._restore_device))

        def run() -> tuple[tuple[str, ...], ...]:
            if self._restore_device.type != "cuda":
                return self._store_many(normalized, _background_l3=True)
            assert self._write_stream is not None
            assert start_event is not None
            with (
                torch.cuda.device(self._restore_device),
                torch.cuda.stream(self._write_stream),
            ):
                start_event.wait(self._write_stream)
                try:
                    return self._store_many(normalized, _background_l3=True)
                finally:
                    # The child futures are source-slot lifetime ACKs.  Fence
                    # even on exceptions so radix locks cannot be released while
                    # a partially launched encoder still reads authoritative L1.
                    self._write_stream.synchronize()

        def distribute(completed: Future) -> None:
            try:
                results = tuple(completed.result())
                if len(results) != len(children):
                    raise RuntimeError("batched store returned the wrong result count")
            except BaseException as exc:  # propagate cancellation and codec failures
                for child in children:
                    child.set_exception(exc)
                return
            for child, result in zip(children, results):
                child.set_result(result)

        def finished(completed: Future) -> None:
            with self._write_lock:
                self._write_futures.discard(completed)

        def complete(completed: Future) -> None:
            try:
                distribute(completed)
            finally:
                # Keep the parent tracked until every child ownership ACK has
                # been resolved, not merely until its worker body returns.
                finished(completed)

        launch_gate = threading.Event()
        abort_gate = threading.Event()

        def gated_run() -> tuple[tuple[str, ...], ...]:
            launch_gate.wait()
            if abort_gate.is_set():
                raise CancelledError("batched store setup did not complete")
            return run()

        parent = None
        try:
            # The worker cannot touch source slots until every ACK and callback
            # is ready.  This also covers an executor implementation that
            # enqueues work and then raises before returning its parent Future.
            parent = self._write_executor.submit(gated_run)
            with self._write_lock:
                self._write_futures.add(parent)
            parent.add_done_callback(complete)
        except BaseException as exc:
            abort_gate.set()
            launch_gate.set()
            # For every post-submit setup failure, wait for the accepted parent
            # before letting radix release source locks.  A submit that queued
            # work without returning is still safe: gated_run observes abort.
            if parent is not None:
                try:
                    parent.result()
                except BaseException:
                    pass
                with self._write_lock:
                    self._write_futures.discard(parent)
            for child in children:
                if not child.done():
                    child.set_exception(exc)
            raise
        launch_gate.set()
        return children

    def store_async(
        self,
        *,
        token_ids: Sequence[int] | torch.Tensor,
        device_indices: torch.Tensor,
        namespace: Any,
        start_token: int = 0,
    ) -> Future:
        """Compress and publish a prefix on the dedicated write worker/stream.

        The returned future completes only after every GPU source read and CPU
        serialization has finished, so the radix adapter may use it as the
        source-slot lifetime ACK.  Blocking L3 I/O also stays on this worker.
        """

        return self.store_many_async(
            (
                {
                    "token_ids": token_ids,
                    "device_indices": device_indices,
                    "namespace": namespace,
                    "start_token": start_token,
                },
            )
        )[0]

    def _drain_writes(self) -> None:
        """Wait for accepted compression jobs without holding bookkeeping locks."""

        while True:
            with self._write_lock:
                futures = tuple(self._write_futures)
            if not futures:
                return
            for future in futures:
                try:
                    future.result()
                except Exception:  # noqa: BLE001
                    logger.exception("background SVD write-through failed")

    def store(
        self,
        *,
        token_ids: Sequence[int] | torch.Tensor,
        device_indices: torch.Tensor,
        namespace: Any,
        start_token: int = 0,
        _background_l3: bool = False,
        _preencoded_blobs: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[str, ...]:
        """Encode selected authoritative L1 chunks from a complete prefix.

        The input is strict by design: ``token_ids`` is the complete,
        chunk-aligned token prefix needed to derive cumulative identities, while
        ``device_indices`` covers only ``[start_token, len(token_ids))``.
        Existing L2/L3 chunks are reused and never recompressed.  When L3 is
        configured it is the durable admission tier: an L2-only copy is retried
        until the L3 write succeeds.  ``start_token`` lets the radix adapter
        publish exactly the chunks touched by an L1 eviction without gathering
        indices for earlier resident chunks.
        """

        self._record_stats(store_calls=1)
        tokens = _normalize_token_ids(token_ids)
        if not tokens or len(tokens) % self.config.chunk_tokens:
            raise ValueError("store requires a nonempty chunk-aligned token prefix")
        if (
            not isinstance(device_indices, torch.Tensor)
            or device_indices.ndim != 1
            or device_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise TypeError(
                "device_indices must be a one-dimensional INT32/INT64 tensor"
            )
        _require_plain_int("start_token", start_token, minimum=0)
        if start_token >= len(tokens):
            raise ValueError("start_token must select at least one complete chunk")
        if start_token % self.config.chunk_tokens:
            raise ValueError("start_token must be chunk-aligned")
        expected_indices = len(tokens) - start_token
        if device_indices.numel() != expected_indices:
            raise ValueError(
                "device_indices must cover token_ids[start_token:] exactly"
            )

        identities = self.identities(tokens, namespace)
        first_chunk = start_token // self.config.chunk_tokens
        selected_identities = identities[first_chunk:]
        durable_keys = set()
        retry_keys = set()
        # Admit in reverse logical order.  If an L2-only byte budget cannot fit
        # every selected blob, later insertions then displace unreachable suffix
        # chunks and leave the earliest contiguous chunks useful to lookup.
        for identity in reversed(selected_identities):
            l2_blob, lease = self._acquire_pool_blob(identity.key)
            if l2_blob is not None:
                try:
                    self._validated_blob(identity.key, l2_blob)
                except Exception:  # noqa: BLE001
                    self._record_stats(invalid_l2_blobs=1)
                    l2_blob = None
                finally:
                    lease.release()

            l3_writable = False
            if self.storage is not None and not _background_l3:
                l3_chunk, l3_writable = self._read_l3_blob(identity.key)
                if l3_chunk is not None:
                    self._complete_l3_retry(identity.key)
                    if l2_blob is None:
                        try:
                            l2_ok = self._admit_parsed(identity.key, l3_chunk)
                        except ValueError:
                            # A validated L3 copy remains usable even if a
                            # corrupt immutable-by-key L2 entry cannot be
                            # replaced until normal LRU eviction.
                            l2_ok = False
                        if l2_ok:
                            self._record_stats(l2_puts=1)
                        else:
                            self._record_stats(l2_rejections=1)
                    durable_keys.add(identity.key)
                    continue
            elif self.storage is None and l2_blob is not None:
                continue

            begin = identity.token_start - start_token
            end = identity.token_end - start_token
            blob = l2_blob
            if blob is None:
                blob = (
                    None
                    if _preencoded_blobs is None
                    else _preencoded_blobs.get(identity.key)
                )
                if blob is None:
                    blob = self.encode_chunk(
                        identity,
                        device_indices[begin:end],
                        namespace=namespace,
                    )
                parsed = self._validated_blob(identity.key, blob)
                self._prewarm_restore_mappings(
                    parsed,
                    contiguous_chunks=min(
                        self.config.restore_prewarm_chunks,
                        int(identity.token_end) // self.config.chunk_tokens,
                    ),
                )
                try:
                    l2_ok = self._admit_parsed(identity.key, parsed)
                except ValueError:
                    l2_ok = False
                self._record_stats(
                    l2_puts=int(l2_ok),
                    l2_rejections=int(not l2_ok),
                )
            else:
                l2_ok = True

            if self.storage is None:
                continue

            if _background_l3:
                scheduled = self._schedule_l3_put(
                    identity.key,
                    blob,
                    retry_from_l2=l2_ok,
                    # If L2 rejected the blob, keep the radix source locked
                    # until L3 itself ACKs; an accepted queue slot is not yet a
                    # durable copy.
                    wait_for_ack=not l2_ok,
                )
                if scheduled:
                    durable_keys.add(identity.key)
                elif l2_ok:
                    retry_keys.add(identity.key)
                continue

            l3_ok = False
            if l3_writable:
                try:
                    l3_ok = self.storage.put(identity.key, blob)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "SVD L3 put() failed for %s", identity.key, exc_info=True
                    )
                    self._record_stats(l3_io_errors=1)
            self._record_stats(
                l3_puts=int(l3_ok),
                l3_rejections=int(not l3_ok),
            )
            if l3_ok:
                durable_keys.add(identity.key)
                self._complete_l3_retry(identity.key)
            elif l2_ok:
                retry_keys.add(identity.key)

        # Queue only blobs that survived the complete reverse-order admission.
        # Their leases make future L2 puts respect the configured byte budget.
        for identity in selected_identities:
            if identity.key in retry_keys and self.pool.exists(identity.key):
                self._queue_l3_retry(identity.key)

        if self.storage is None:
            # Report current residency, not historical put success: a later
            # blob in this same call may have evicted an earlier one.
            return tuple(
                identity.key
                for identity in selected_identities
                if self.pool.exists(identity.key)
            )
        return tuple(
            identity.key
            for identity in selected_identities
            if identity.key in durable_keys
        )

    def release_pending(self, rid: str) -> None:
        queued_markers = []
        with self._load_lock:
            kept = []
            for operation in self._load_queue:
                if operation.marker.rid == rid:
                    queued_markers.append(operation.marker)
                else:
                    kept.append(operation)
            self._load_queue = kept
            inflight = any(
                operation.marker.rid == rid
                for batch in self._inflight_loads
                for operation in batch.operations
            )
        with self._pending_lock:
            marker = self._pending.pop(rid, None)
        for queued in queued_markers:
            queued.release()
        if (
            marker is not None
            and not inflight
            and all(marker is not queued for queued in queued_markers)
        ):
            marker.release()

    def reset(self) -> None:
        self._drain_writes()
        with self._prefetch_lock:
            prefetches = tuple(self._prefetches.values())
            self._prefetches.clear()
            self._prefetched_chunks.clear()
            self._prefetch_loaded_tokens.clear()
        for state in prefetches:
            for future in state.futures:
                future.cancel()
        with self._load_lock:
            queued = tuple(self._load_queue)
            self._load_queue.clear()
        for operation in queued:
            self._finish_marker(operation.marker)
        self._reap_loads(synchronize=True)
        if self.layer_done_counter is not None:
            self.layer_done_counter.reset()
        self._release_l3_retries()
        with self._pending_lock:
            markers = tuple(self._pending.values())
            self._pending.clear()
        for marker in markers:
            marker.release()
        self.pool.clear()

    def shutdown(self) -> None:
        # One bounded best-effort durability pass before releasing retry leases.
        self._drain_writes()
        self._write_executor.shutdown(wait=True, cancel_futures=False)
        self.check_events()
        if self._l3_tasks is not None:
            failures = self._l3_tasks.shutdown(wait=True)
            for failure in failures:
                logger.warning(
                    "SVD L3 %s failed for %s during shutdown: %s",
                    failure.operation,
                    failure.key,
                    failure.exception,
                )
        self.reset()


def create_svd_chunk_connector(
    config_value: Any,
    *,
    device_pool: Any,
    model_name: Optional[str],
    page_size: Optional[int] = None,
    l2_capacity_bytes: Optional[int] = None,
    storage: Any = None,
    storage_config: Any = None,
    local_layer_ids: Optional[Sequence[int]] = None,
    encoder: Any = None,
    buffer_layer_indices: Optional[Sequence[int]] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    preflight_encoder: bool = False,
) -> Optional[SVDChunkConnector]:
    """Construct the default L2/L3 connector used by the radix integration.

    The caller may supply an already-created storage backend.  Otherwise a
    configured ``l3_path`` creates a rank-namespaced ``HiCacheFile`` backend.
    ``l2_capacity_bytes`` takes precedence over ``l2_capacity_gb`` so the live
    server can derive the byte budget from its ordinary HiCache size/ratio.
    """

    config = resolve_svd_chunk_connector_config(config_value)
    if config is None:
        return None
    resolved_page_size = int(
        getattr(device_pool, "page_size") if page_size is None else page_size
    )
    if l2_capacity_bytes is None:
        if config.l2_capacity_gb is None:
            raise ValueError("l2_capacity_bytes or config.l2_capacity_gb is required")
        # Match ServerArgs.hicache_size, whose public unit is decimal GB.
        l2_capacity_bytes = int(config.l2_capacity_gb * 1_000_000_000)
    _require_plain_int("l2_capacity_bytes", l2_capacity_bytes, minimum=0)
    if l2_capacity_bytes == 0 and storage is None and config.l3_path is None:
        raise ValueError(
            "zero-capacity SVD L2 requires l3_path or an injected L3 backend"
        )

    parallel = {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "attn_cp_rank": attn_cp_rank,
        "attn_cp_size": attn_cp_size,
    }
    codec_metadata = build_svd_codec_metadata_from_pool(
        config,
        device_pool,
        model_name=model_name,
        page_size=resolved_page_size,
        local_layer_ids=local_layer_ids,
        **parallel,
    )

    if storage is None and config.l3_path is not None:
        from sglang.srt.mem_cache.hicache_storage import (
            HiCacheFile,
            HiCacheStorageConfig,
        )

        if storage_config is None:
            storage_config = HiCacheStorageConfig(
                **parallel,
                is_mla_model=False,
                enable_storage_metrics=False,
                is_page_first_layout=False,
                model_name=model_name,
            )
        elif not isinstance(storage_config, HiCacheStorageConfig):
            raise TypeError("storage_config must be HiCacheStorageConfig")
        storage = HiCacheFile(storage_config, file_path=config.l3_path)

    from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_pool import SVDChunkPool

    first_buffer = device_pool.k_buffer[
        0 if buffer_layer_indices is None else int(buffer_layer_indices[0])
    ]
    connector = SVDChunkConnector(
        config,
        device_pool=device_pool,
        page_size=resolved_page_size,
        pool=SVDChunkPool(
            l2_capacity_bytes,
            pin_memory=first_buffer.device.type == "cuda",
        ),
        codec_metadata=codec_metadata,
        storage=storage,
        encoder=encoder,
        buffer_layer_indices=buffer_layer_indices,
    )
    if preflight_encoder:
        connector.preflight_encoder()
    return connector
