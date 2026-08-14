"""Opt-in live shadow runner for joint-head SVD HiCache experiments.

The shadow observes L1->L2 writes after the normal raw transfer has been
enqueued.  It never owns host slots, publishes factors, or participates in
lookups.  Its only durable state is aggregate instrumentation.

``paged_svd`` is deliberately imported only when the feature is enabled.  A
normal SGLang installation therefore does not acquire a dependency on the
research package.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Optional

import torch

from sglang.srt.mem_cache.hicache_svd_codec import (
    page_ids_from_device_indices,
    quantize_svd_factors,
    view_nhd_as_joint_pages,
)

logger = logging.getLogger(__name__)

HICACHE_SVD_SHADOW_ENV = "SGLANG_HICACHE_SVD_SHADOW"


@dataclass(frozen=True)
class HiCacheSVDShadowConfig:
    """Configuration for the non-owning live shadow path."""

    chunk_tokens: int = 4096
    rank: int = 32
    bits_u: int = 4
    bits_r: int = 4
    scale_mode: str = "matrix"
    group_size: Optional[int] = None
    niter: int = 2
    max_layers: Optional[int] = 1
    max_chunks_per_backup: int = 1
    log_every: int = 1
    backend: str = "triton"
    precision: str = "ieee"

    def __post_init__(self) -> None:
        if self.chunk_tokens < 4096:
            raise ValueError("chunk_tokens must be at least 4096")
        if not 1 <= self.rank <= 128:
            raise ValueError("rank must be in [1, 128]")
        if self.bits_u not in (2, 4, 8) or self.bits_r not in (2, 4, 8):
            raise ValueError("bits_u and bits_r must be one of 2, 4, or 8")
        if self.scale_mode not in ("matrix", "row"):
            raise ValueError("scale_mode must be 'matrix' or 'row'")
        if self.scale_mode == "matrix" and self.group_size is not None:
            raise ValueError("group_size is only valid for row scale mode")
        if self.scale_mode == "row" and (
            self.group_size is None or self.group_size < 1
        ):
            raise ValueError("row scale mode requires a positive group_size")
        if self.niter < 0:
            raise ValueError("niter must be nonnegative")
        if self.max_layers is not None and self.max_layers < 1:
            raise ValueError("max_layers must be positive or null")
        if self.max_chunks_per_backup < 1:
            raise ValueError("max_chunks_per_backup must be positive")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.backend not in ("triton", "cute_bf16"):
            raise ValueError("backend must be 'triton' or 'cute_bf16'")
        if self.precision not in ("ieee", "tf32"):
            raise ValueError("precision must be 'ieee' or 'tf32'")
        if self.backend == "cute_bf16" and self.precision != "ieee":
            raise ValueError("cute_bf16 uses fixed BF16 inputs; select ieee")


@dataclass(frozen=True)
class ShadowEncodeResult:
    raw_nbytes: int
    encoded_nbytes: int
    factor_matrices: int


@dataclass
class HiCacheSVDShadowMetrics:
    backup_calls: int = 0
    encoded_chunks: int = 0
    encoded_tokens: int = 0
    skipped_short_backups: int = 0
    skipped_backend_backups: int = 0
    skipped_tail_tokens: int = 0
    skipped_capped_tokens: int = 0
    failures: int = 0
    raw_nbytes: int = 0
    encoded_nbytes: int = 0
    factor_matrices: int = 0
    host_seconds: float = 0.0
    timed_chunks: int = 0
    gpu_milliseconds: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        result = asdict(self)
        result["capacity_ratio"] = (
            self.raw_nbytes / self.encoded_nbytes if self.encoded_nbytes else 0.0
        )
        return result


def _config_from_mapping(values: Mapping[str, Any]) -> HiCacheSVDShadowConfig:
    values = dict(values)
    bits = values.pop("bits", None)
    if bits is not None:
        values.setdefault("bits_u", bits)
        values.setdefault("bits_r", bits)
    valid_fields = set(HiCacheSVDShadowConfig.__dataclass_fields__)
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"unknown HiCache SVD shadow fields: {unknown}")
    return HiCacheSVDShadowConfig(**values)


def resolve_hicache_svd_shadow_config(
    value: Any = None,
    *,
    use_env_fallback: bool = True,
) -> Optional[HiCacheSVDShadowConfig]:
    """Normalize a config object/string, optionally falling back to the dev env.

    Accepted strings are a JSON object, or a conventional boolean spelling.
    ``True`` selects defaults; ``False`` disables the feature.
    """

    if value is None and use_env_fallback:
        value = os.environ.get(HICACHE_SVD_SHADOW_ENV)
    if value is None or value is False:
        return None
    if isinstance(value, HiCacheSVDShadowConfig):
        return value
    if value is True:
        return HiCacheSVDShadowConfig()
    if isinstance(value, Mapping):
        return _config_from_mapping(value)
    if not isinstance(value, str):
        raise TypeError(
            "HiCache SVD shadow config must be bool, string, mapping, or "
            "HiCacheSVDShadowConfig"
        )

    stripped = value.strip()
    if stripped.lower() in ("", "0", "false", "off", "no", "none"):
        return None
    if stripped.lower() in ("1", "true", "on", "yes"):
        return HiCacheSVDShadowConfig()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("HiCache SVD shadow JSON config must be an object")
    return _config_from_mapping(parsed)


class _PagedSVDChunkEncoder:
    """Direct-paged randomized SVD plus factor quantization for one chunk."""

    def __init__(
        self,
        config: HiCacheSVDShadowConfig,
        page_size: int,
        layer_ids: tuple[int, ...],
        paged_svd_module: Any,
    ) -> None:
        self.config = config
        self.page_size = page_size
        self.layer_ids = layer_ids
        self._paged_svd = paged_svd_module

    def encode_chunk(
        self, device_pool: Any, device_indices: torch.Tensor
    ) -> ShadowEncodeResult:
        page_ids = page_ids_from_device_indices(
            device_indices, page_size=self.page_size
        )
        first_pages = view_nhd_as_joint_pages(
            device_pool.k_buffer[self.layer_ids[0]], page_size=self.page_size
        )
        page_count = int(page_ids.numel())
        indptr = torch.tensor(
            [0, page_count], dtype=torch.int64, device=device_indices.device
        )
        last_page_lens = torch.tensor(
            [self.page_size], dtype=torch.int64, device=device_indices.device
        )
        layout = self._paged_svd.prepare_paged_layout(
            first_pages, indptr, page_ids, last_page_lens
        )
        workspace = self._paged_svd.prepare_direct_workspace(
            first_pages, layout, self.config.rank
        )

        raw_nbytes = 0
        encoded_nbytes = 0
        factor_matrices = 0
        group_size = self.config.group_size if self.config.scale_mode == "row" else None
        # K and V are decomposed separately, but every local KV head participates
        # jointly through the flattened feature dimension of each paged view.
        for layer_id in self.layer_ids:
            for values in (
                device_pool.k_buffer[layer_id],
                device_pool.v_buffer[layer_id],
            ):
                pages = view_nhd_as_joint_pages(values, page_size=self.page_size)
                result = self._paged_svd.paged_svd_lowrank_direct(
                    pages,
                    layout=layout,
                    q=self.config.rank,
                    niter=self.config.niter,
                    workspace=workspace,
                    precision=self.config.precision,
                    backend=self.config.backend,
                )
                factors = quantize_svd_factors(
                    result.U,
                    result.S,
                    result.V,
                    bits_u=self.config.bits_u,
                    bits_r=self.config.bits_r,
                    scale_mode=self.config.scale_mode,
                    group_size_u=group_size,
                    group_size_r=group_size,
                )
                raw_nbytes += (
                    device_indices.numel()
                    * values.shape[1]
                    * values.shape[2]
                    * values.element_size()
                )
                encoded_nbytes += factors.storage_nbytes
                factor_matrices += 1

        return ShadowEncodeResult(
            raw_nbytes=raw_nbytes,
            encoded_nbytes=encoded_nbytes,
            factor_matrices=factor_matrices,
        )


class HiCacheSVDShadowObserver:
    """Non-owning observer invoked synchronously from the HiCache write stream."""

    def __init__(
        self,
        config: HiCacheSVDShadowConfig,
        encoder: Any,
        *,
        pool_label: str = "kv",
        timing_event_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.pool_label = pool_label
        self.metrics = HiCacheSVDShadowMetrics()
        self._timing_event_factory = timing_event_factory
        self._pending_timings: list[tuple[Any, Any]] = []
        self.active = True

    def _drain_timings(self) -> None:
        pending = []
        for start, finish in self._pending_timings:
            if not finish.query():
                pending.append((start, finish))
                continue
            self.metrics.gpu_milliseconds += start.elapsed_time(finish)
            self.metrics.timed_chunks += 1
        self._pending_timings = pending

    def _log_metrics(self) -> None:
        snapshot = self.metrics.snapshot()
        logger.info(
            "HiCache SVD shadow pool=%s chunks=%d tokens=%d raw_bytes=%d "
            "encoded_bytes=%d capacity_ratio=%.2fx host_seconds=%.4f "
            "gpu_ms=%.3f timed_chunks=%d pending_timings=%d failures=%d",
            self.pool_label,
            snapshot["encoded_chunks"],
            snapshot["encoded_tokens"],
            snapshot["raw_nbytes"],
            snapshot["encoded_nbytes"],
            snapshot["capacity_ratio"],
            snapshot["host_seconds"],
            snapshot["gpu_milliseconds"],
            snapshot["timed_chunks"],
            len(self._pending_timings),
            snapshot["failures"],
        )

    def observe_backup(
        self,
        device_pool: Any,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        if not self.active:
            return
        self._drain_timings()
        self.metrics.backup_calls += 1
        if io_backend != "kernel":
            self.metrics.skipped_backend_backups += 1
            return

        token_count = int(device_indices.numel())
        chunk_tokens = self.config.chunk_tokens
        available_chunks = token_count // chunk_tokens
        if available_chunks == 0:
            self.metrics.skipped_short_backups += 1
            self.metrics.skipped_tail_tokens += token_count
            return

        selected_chunks = min(available_chunks, self.config.max_chunks_per_backup)
        selected_tokens = selected_chunks * chunk_tokens
        self.metrics.skipped_tail_tokens += token_count % chunk_tokens
        self.metrics.skipped_capped_tokens += (
            token_count - selected_tokens - (token_count % chunk_tokens)
        )

        for chunk_index in range(selected_chunks):
            begin = chunk_index * chunk_tokens
            end = begin + chunk_tokens
            chunk_indices = device_indices[begin:end]
            start_event = finish_event = None
            if self._timing_event_factory is not None:
                start_event = self._timing_event_factory()
                finish_event = self._timing_event_factory()
                start_event.record()
            host_start = time.perf_counter()
            try:
                result = self.encoder.encode_chunk(device_pool, chunk_indices)
            except Exception:
                self.metrics.failures += 1
                self.active = False
                logger.exception(
                    "HiCache SVD shadow encode failed; disabling the observer"
                )
                return
            finally:
                self.metrics.host_seconds += time.perf_counter() - host_start

            if finish_event is not None:
                finish_event.record()
                self._pending_timings.append((start_event, finish_event))
            self.metrics.encoded_chunks += 1
            self.metrics.encoded_tokens += chunk_tokens
            self.metrics.raw_nbytes += result.raw_nbytes
            self.metrics.encoded_nbytes += result.encoded_nbytes
            self.metrics.factor_matrices += result.factor_matrices
            if self.metrics.encoded_chunks % self.config.log_every == 0:
                self._drain_timings()
                self._log_metrics()

    def metrics_snapshot(self) -> dict[str, Any]:
        self._drain_timings()
        return self.metrics.snapshot()


def _compatibility_error(
    device_pool: Any,
    config: HiCacheSVDShadowConfig,
    page_size: int,
    has_mtp_draft: bool,
) -> Optional[str]:
    if has_mtp_draft:
        return "packed MTP draft layers are not supported by the first shadow slice"
    if getattr(device_pool, "kv_cache_layout", None) != "nhd":
        return "the shadow requires an NHD device KV layout"
    if getattr(device_pool, "is_quantized_kv_cache", False):
        return "quantized device KV caches are not supported"
    if getattr(device_pool, "head_dim", None) != getattr(
        device_pool, "v_head_dim", None
    ):
        return "asymmetric K/V head dimensions are not supported yet"
    if config.chunk_tokens % page_size:
        return "chunk_tokens must be divisible by the device page size"
    buffers = getattr(device_pool, "k_buffer", None)
    value_buffers = getattr(device_pool, "v_buffer", None)
    if not buffers or not value_buffers:
        return "the device pool does not expose K/V buffers"
    first = buffers[0]
    if not first.is_cuda:
        return "the live direct-paged shadow requires CUDA tensors"
    if torch.version.hip is not None:
        return "the first live shadow slice is CUDA-only"
    if not first.is_floating_point():
        return "the live shadow requires floating-point device KV tensors"
    if first.ndim != 3 or not first.is_contiguous():
        return "the live shadow requires contiguous [slots,H,D] buffers"
    if first.shape[0] % page_size:
        return "the device slot count must be divisible by page_size"
    feature_dim = int(first.shape[1] * first.shape[2])
    if config.chunk_tokens < feature_dim:
        return "direct paged SVD requires chunk_tokens >= H*D"
    if config.rank > feature_dim:
        return "rank cannot exceed H*D"
    if config.backend == "cute_bf16" and (feature_dim % 128 or config.rank % 8):
        return "cute_bf16 requires H*D divisible by 128 and rank divisible by 8"
    return None


def maybe_create_hicache_svd_shadow(
    *,
    device_pool: Any,
    page_size: int,
    pool_label: str,
    config_value: Any = None,
    use_env_fallback: bool = True,
    has_mtp_draft: bool = False,
) -> Optional[HiCacheSVDShadowObserver]:
    """Build the live observer, returning ``None`` for every disabled path."""

    try:
        config = resolve_hicache_svd_shadow_config(
            config_value, use_env_fallback=use_env_fallback
        )
    except Exception:
        logger.exception("Invalid HiCache SVD shadow configuration; feature disabled")
        return None
    if config is None:
        return None
    if pool_label != "kv":
        logger.info(
            "HiCache SVD shadow is disabled for non-primary pool %s", pool_label
        )
        return None

    error = _compatibility_error(device_pool, config, page_size, has_mtp_draft)
    if error is not None:
        logger.warning("HiCache SVD shadow disabled: %s", error)
        return None

    try:
        import paged_svd
    except ImportError:
        logger.warning(
            "HiCache SVD shadow requested but optional package 'paged_svd' is "
            "unavailable. Install /home/ccchang/svd_on_paged into the runtime "
            "environment; raw HiCache remains active."
        )
        return None
    if config.backend == "triton" and not paged_svd.has_triton():
        logger.warning(
            "HiCache SVD shadow requested with backend=triton, but Triton is "
            "unavailable; raw HiCache remains active."
        )
        return None

    layer_count = int(device_pool.layer_num)
    selected_layers = (
        layer_count
        if config.max_layers is None
        else min(layer_count, config.max_layers)
    )
    layer_ids = tuple(range(selected_layers))
    encoder = _PagedSVDChunkEncoder(config, page_size, layer_ids, paged_svd)

    def timing_event_factory():
        return torch.cuda.Event(enable_timing=True)

    observer = HiCacheSVDShadowObserver(
        config,
        encoder,
        pool_label=pool_label,
        timing_event_factory=timing_event_factory,
    )
    logger.info(
        "Enabled non-owning HiCache SVD shadow: pool=%s chunk_tokens=%d "
        "rank=%d bits_u=%d bits_r=%d layers=%s max_chunks_per_backup=%d",
        pool_label,
        config.chunk_tokens,
        config.rank,
        config.bits_u,
        config.bits_r,
        layer_ids,
        config.max_chunks_per_backup,
    )
    return observer
