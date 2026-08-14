"""Experimental radix adapter for chunk-compressed SVD KV storage.

This module deliberately contains only the scheduler/radix integration.  The
connector owns compression, byte-capacity accounting, and L2/L3 persistence.
Keeping that boundary explicit avoids representing a variable-sized SVD blob
as ordinary token-addressed HiCache host pages.

The external tier has a coarser contract than the L1 radix cache:

* only complete ``chunk_size``-token prefixes are looked up or stored;
* lookup is metadata-only and returns a marker with ``matched_end``;
* retrieve restores the suffix ``[device_len, matched_end)`` into
  caller-allocated L1 slots, synchronizing before forward by default while
  retaining event-gated ``full`` and ``layerwise`` experiment modes; and
* complete chunks are compressed asynchronously after radix insertion, with
  their source nodes locked until the write future acknowledges consumption.

Eviction never performs codec work for the live write-through connector.  A
legacy connector without ``store_async`` retains the synchronous eviction
fallback so allocator reclamation remains compatible with the smaller test
and extension protocol.

This is an MHA-only prototype.  Eagle/bigram keys are rejected because a
4K-token compression chunk and a 4K-bigram radix span do not have the same
shape contract.  ``chunk_size`` must also be divisible by the device radix
``page_size``.
"""

from __future__ import annotations

import logging
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Tuple, runtime_checkable

import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


@runtime_checkable
class SVDChunkLookupMarker(Protocol):
    """Minimum result returned by :meth:`SVDChunkConnector.lookup`."""

    matched_end: int


@runtime_checkable
class SVDChunkConnectorProtocol(Protocol):
    """Connector surface consumed by :class:`SVDChunkRadixCache`.

    ``token_ids`` always describes the complete chunk-aligned prefix, while
    ``device_len`` may fall inside its final compressed chunk.  In that case
    retrieve reconstructs the whole chunk internally but writes only the
    missing suffix to ``dest``.  Async store admission is atomic with respect
    to source ownership: a successful batched call returns exactly one ACK
    future per request, while a raised call leaves no worker able to read the
    supplied source indices.
    """

    def lookup(
        self,
        *,
        token_ids,
        namespace: Optional[str],
        device_len: int,
        rid: str,
        allow_l3: bool = True,
    ) -> Optional[SVDChunkLookupMarker]: ...

    def retrieve(self, marker: SVDChunkLookupMarker, dest: torch.Tensor) -> int: ...

    def store(
        self,
        *,
        token_ids,
        device_indices: torch.Tensor,
        namespace: Optional[str],
        start_token: int = 0,
    ) -> Any: ...

    def store_async(
        self,
        *,
        token_ids,
        device_indices: torch.Tensor,
        namespace: Optional[str],
        start_token: int = 0,
    ) -> Any: ...

    def store_many_async(self, requests) -> tuple[Any, ...]: ...

    def release_pending(self, rid: str) -> None: ...

    def synchronize_source_reads(self) -> None: ...

    def reset(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True)
class _LoadBackMarker:
    """Detached state carried from lookup to scheduler-time restore."""

    connector_marker: SVDChunkLookupMarker
    key: RadixKey
    device_len: int
    matched_end: int


@dataclass(frozen=True)
class _PrefetchCandidate:
    """Full prefix snapshot needed to derive content-addressed L3 keys."""

    token_ids: tuple[int, ...]
    namespace: Any
    device_len: int


@dataclass(frozen=True)
class _DeferredWrite:
    """A bounded, unlocked retry candidate for a still-resident L1 chunk."""

    chunk_key: str
    token_ids: array
    namespace: Any
    token_start: int
    token_end: int
    attempts: int = 0


@dataclass(frozen=True)
class _WriteThroughTask:
    """One background compression job and the radix lock protecting its source."""

    chunk_key: str
    future: Any
    lock_node: TreeNode
    request: _DeferredWrite


class SVDChunkRadixCache(RadixCache):
    """RadixCache with a coarse, SVD-compressed external prefix tier.

    Registry construction passes server/model configuration and lets this
    class lazily create the connector.  Unit tests and experiments may inject
    a connector directly.
    """

    DEFAULT_CHUNK_SIZE = 4096
    # The write worker owns a radix source lock and a completion Future.  It is
    # safe for an otherwise idle scheduler to block in its event poll; the next
    # request wakes the loop and reaps the ACK before cache admission.
    hicache_writes_allow_idle_sleep = True

    def __init__(
        self,
        params: CacheInitParams,
        server_args: ServerArgs,
        model_config: Optional[ModelConfig] = None,
        connector: Optional[SVDChunkConnectorProtocol] = None,
        *,
        chunk_size: Optional[int] = None,
    ) -> None:
        if connector is None:
            connector = self._create_connector(params, server_args, model_config)
        if connector is None:
            raise ValueError("SVD chunk connector configuration is disabled")

        connector_config = getattr(connector, "config", None)
        configured_chunk_size = getattr(
            connector_config,
            "chunk_tokens",
            self.DEFAULT_CHUNK_SIZE,
        )
        chunk_size = (
            int(configured_chunk_size) if chunk_size is None else int(chunk_size)
        )
        if connector_config is not None and chunk_size != int(configured_chunk_size):
            raise ValueError(
                "radix chunk_size must match connector config: "
                f"{chunk_size=} connector_chunk_size={configured_chunk_size}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if params.page_size <= 0 or chunk_size % params.page_size != 0:
            raise ValueError(
                "chunk_size must be divisible by the radix page_size: "
                f"{chunk_size=} {params.page_size=}"
            )
        if params.is_eagle:
            raise ValueError("SVDChunkRadixCache does not support Eagle/bigram keys")

        # RadixCache.__init__ dynamically calls self.reset(), so all subclass
        # reset state must tolerate the connector not being installed yet.
        super().__init__(params)
        self.server_args = server_args
        self.model_config = model_config
        self.connector = connector
        self.chunk_size = chunk_size
        self.layerwise_restore = self._configure_layerwise_restore(params, connector)
        self._pending_restore_acks: list[tuple[str, TreeNode]] = []
        self._inflight_restore_acks: dict[int, tuple[tuple[str, TreeNode], ...]] = {}
        connector_storage = getattr(connector, "storage", None)
        self.enable_storage = bool(
            getattr(connector, "enable_storage", connector_storage is not None)
        )
        self._load_markers: dict[str, _LoadBackMarker] = {}
        self._prefetch_candidates: dict[str, _PrefetchCandidate] = {}
        # Scheduler compatibility.  An event-capable connector owns the CUDA
        # restore streams/events while this adapter owns radix locks until the
        # corresponding final-layer ACK.  The default schedule synchronizes
        # that final event before forward; full/layerwise modes retain gates.
        self.ongoing_write_through: dict[Any, Any] = {}
        self.ongoing_load_back: dict[Any, Any] = {}
        self.ongoing_prefetch: dict[Any, Any] = {}
        self.ongoing_backup: dict[Any, Any] = {}
        self._write_through_enabled = callable(
            getattr(connector, "store_async", None)
        ) and callable(getattr(connector, "identities", None))
        self._write_task_counter = 0
        self._write_chunk_tasks: dict[str, int] = {}
        self._write_through_complete: set[str] = set()
        # Compression is much slower than a D2D/L1 eviction.  Bound the number
        # of radix nodes whose source slots can be lock-protected at once; the
        # remaining candidates are metadata-only retries and stay evictable.
        self._max_pending_writes = int(
            getattr(connector_config, "max_pending_writes", 1 << 30)
        )
        self._batched_write_through = callable(
            getattr(connector, "store_many_async", None)
        )
        self._encode_batch_size = max(
            1,
            min(
                self._max_pending_writes,
                (
                    int(getattr(connector_config, "encode_batch_size", 1))
                    if self._batched_write_through
                    else 1
                ),
            ),
        )
        # The concrete connector has one write worker.  Once microbatching is
        # enabled, queuing several independent parent jobs only freezes their
        # batch size at one and makes the worker execute those singletons in
        # order.  Keep later candidates metadata-only until every child ACK from
        # the active parent has been reaped, then capture one new microbatch.
        # Configurations without a real multi-item path retain their previous
        # bounded, independently queued write behavior.
        self._coalesce_write_through = (
            self._batched_write_through and self._encode_batch_size > 1
        )
        self._max_deferred_writes = int(
            getattr(connector_config, "max_deferred_writes", 64)
        )
        self._write_retry_limit = int(getattr(connector_config, "write_retry_limit", 3))
        self._deferred_write_through: OrderedDict[str, _DeferredWrite] = OrderedDict()
        self.write_through_deferred_count = 0
        self.write_through_dropped_count = 0
        self._shutdown = False

    def _configure_layerwise_restore(
        self, params: CacheInitParams, connector: Any
    ) -> bool:
        """Register an opt-in connector's stock HiCache layer-event counter."""

        enqueue = callable(getattr(connector, "enqueue_retrieve", None))
        start = callable(getattr(connector, "start_loading", None))
        counter = getattr(connector, "layer_done_counter", None)
        # The concrete connector deliberately exposes the queue methods on CPU
        # too, where they fall back to synchronous retrieve and have no event
        # counter.  A counter is therefore the capability advertisement.
        if counter is None:
            return False
        if not enqueue or not start:
            raise TypeError(
                "layerwise SVD restore requires enqueue_retrieve(), "
                "start_loading(), and layer_done_counter"
            )
        device_pool = params.token_to_kv_pool_allocator.get_kvcache()
        register_counter = getattr(device_pool, "register_layer_transfer_counter", None)
        if not callable(register_counter):
            raise TypeError(
                "layerwise SVD restore requires a KV pool transfer-counter hook"
            )
        register_counter(counter)
        return True

    def runtime_metrics_snapshot(self) -> dict[str, Any]:
        """Return a read-only, JSON-safe snapshot for server introspection."""

        connector_snapshot_fn = getattr(self.connector, "metrics_snapshot", None)
        connector_snapshot = (
            dict(connector_snapshot_fn()) if callable(connector_snapshot_fn) else {}
        )

        pool_snapshot = {}
        pool_stats_fn = getattr(getattr(self.connector, "pool", None), "stats", None)
        if callable(pool_stats_fn):
            pool_stats = pool_stats_fn()
            for field_name in (
                "capacity_bytes",
                "used_bytes",
                "available_bytes",
                "entry_count",
                "pinned_entry_count",
                "pin_count",
                "hits",
                "misses",
                "evictions",
                "rejected_puts",
            ):
                pool_snapshot[field_name] = getattr(pool_stats, field_name)

        return {
            "schema_version": 1,
            "radix_eviction_policy": self.eviction_policy,
            "connector": connector_snapshot,
            "l2_pool": pool_snapshot,
            "write_through": {
                "inflight": len(self.ongoing_write_through),
                "deferred": len(self._deferred_write_through),
                "deferred_events": self.write_through_deferred_count,
                "dropped": self.write_through_dropped_count,
                "max_inflight": self._max_pending_writes,
                "max_deferred": self._max_deferred_writes,
            },
        }

    @classmethod
    def _create_connector(
        cls,
        params: CacheInitParams,
        server_args: ServerArgs,
        model_config: Optional[ModelConfig],
    ) -> Optional[SVDChunkConnectorProtocol]:
        """Lazily build the concrete connector for registry construction."""

        if server_args is None:
            raise ValueError("server_args is required when connector is not injected")
        allocator = params.token_to_kv_pool_allocator
        if allocator is None:
            raise ValueError("a device KV allocator is required for SVD chunks")
        device_pool = allocator.get_kvcache()

        from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector import (
            create_svd_chunk_connector,
            resolve_svd_chunk_connector_config,
        )

        config_value = getattr(server_args, "hicache_svd_config", None)
        resolved_config = resolve_svd_chunk_connector_config(config_value)
        if resolved_config is None:
            return None
        model_name = (
            getattr(model_config, "model_path", None)
            or getattr(server_args, "served_model_name", None)
            or getattr(server_args, "model_path", None)
            or "unknown-model"
        )
        model_revision = getattr(server_args, "revision", None)
        if model_revision:
            # L3 can outlive a server process, so weight revisions must not
            # share a prefix namespace even when the user-facing model path is
            # unchanged.
            model_name = f"{model_name}@{model_revision}"
        local_layer_count = len(getattr(device_pool, "k_buffer", ()))
        local_layer_start = int(getattr(device_pool, "start_layer", 0))
        local_layer_ids = tuple(
            range(local_layer_start, local_layer_start + local_layer_count)
        )
        return create_svd_chunk_connector(
            config_value,
            device_pool=device_pool,
            page_size=params.page_size,
            model_name=model_name,
            l2_capacity_bytes=(
                None
                if resolved_config.l2_capacity_gb is not None
                else cls._derive_l2_capacity_bytes(server_args, device_pool)
            ),
            local_layer_ids=local_layer_ids,
            tp_rank=0,
            tp_size=int(getattr(server_args, "tp_size", 1)),
            pp_rank=int(getattr(params, "pp_rank", 0)),
            pp_size=int(getattr(params, "pp_size", 1)),
            attn_cp_rank=int(getattr(params, "attn_cp_rank", 0)),
            attn_cp_size=int(getattr(params, "attn_cp_size", 1)),
            preflight_encoder=True,
        )

    @staticmethod
    def _derive_l2_capacity_bytes(server_args: ServerArgs, device_pool: Any) -> int:
        size_gb = int(getattr(server_args, "hicache_size", 0) or 0)
        if size_gb > 0:
            # Match HostKVCache's existing --hicache-size interpretation.
            return size_gb * 1_000_000_000

        ratio = float(getattr(server_args, "hicache_ratio", 0.0) or 0.0)
        if ratio < 0:
            raise ValueError("hicache_ratio must be non-negative")
        raw_l1_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                *getattr(device_pool, "k_buffer", ()),
                *getattr(device_pool, "v_buffer", ()),
            )
            if isinstance(tensor, torch.Tensor)
        )
        return int(raw_l1_bytes * ratio)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:  # type: ignore[override]
        if hasattr(self, "ongoing_write_through"):
            self._drain_write_through(synchronize=True)
        if getattr(self, "layerwise_restore", False):
            self._synchronize_layerwise_restores()
        if hasattr(self, "ongoing_prefetch"):
            for rid in tuple(self.ongoing_prefetch):
                self._cancel_connector_prefetch(rid)
        if hasattr(self, "_prefetch_candidates"):
            self._prefetch_candidates.clear()
        if hasattr(self, "_load_markers"):
            for rid in tuple(self._load_markers):
                self._release_pending(rid)
            self._load_markers.clear()
        for name in (
            "ongoing_load_back",
            "ongoing_prefetch",
            "ongoing_backup",
        ):
            if hasattr(self, name):
                getattr(self, name).clear()
        if hasattr(self, "_write_chunk_tasks"):
            self._write_chunk_tasks.clear()
            self._write_through_complete.clear()
        if hasattr(self, "_deferred_write_through"):
            self._deferred_write_through.clear()
        if hasattr(self, "connector"):
            self.connector.reset()
        super().reset()

    def shutdown(self) -> None:
        if not hasattr(self, "connector") or getattr(self, "_shutdown", False):
            return
        self._shutdown = True
        self._drain_write_through(synchronize=True)
        if getattr(self, "layerwise_restore", False):
            self._synchronize_layerwise_restores()
        for rid in tuple(self.ongoing_prefetch):
            self._cancel_connector_prefetch(rid)
        self._prefetch_candidates.clear()
        for rid in tuple(self._load_markers):
            self._release_pending(rid)
        self._load_markers.clear()
        self.connector.shutdown()

    def release_host_resources(self) -> None:
        """Release compressed-tier leases/resources during graceful shutdown."""

        self.shutdown()

    def release_aborted_request(self, rid: str) -> None:
        """Release a lookup marker when scheduling abandons a request."""

        self._prefetch_candidates.pop(rid, None)
        self._cancel_connector_prefetch(rid)
        if self._owns_layerwise_restore(rid):
            # The restore is already part of an admitted batch.  Its completed
            # rows remain a valid cache fill even if the originating request is
            # aborted; the ACK callback releases the marker and transfer lock.
            return
        self._load_markers.pop(rid, None)
        self._release_pending(rid)

    def terminate_prefetch(self, rid: str) -> None:
        """Cancel L3 staging and release any request-owned lookup marker."""

        self.release_aborted_request(rid)

    def prefetch_from_storage(
        self,
        rid: str,
        last_host_node: TreeNode,
        new_input_tokens,
        last_hash: Optional[str] = None,
        prefix_keys=None,
    ) -> None:
        """Stage the remaining content-addressed chunks from L3 into L2.

        Stock HiCache passes only the suffix after its current host hit.  SVD
        chunk identities, however, hash the complete prefix.  ``match_prefix``
        therefore snapshots the complete chunk-aligned key and this method
        consumes that snapshot; the host-node/hash arguments are intentionally
        unused compatibility inputs.
        """

        del last_host_node, new_input_tokens, last_hash, prefix_keys
        candidate = self._prefetch_candidates.pop(rid, None)
        prefetch = getattr(self.connector, "prefetch", None)
        if (
            candidate is None
            or not self.enable_storage
            or not callable(prefetch)
            or candidate.device_len >= len(candidate.token_ids)
        ):
            return
        self._cancel_connector_prefetch(rid)
        try:
            prefetch(
                token_ids=candidate.token_ids,
                namespace=candidate.namespace,
                device_len=candidate.device_len,
                rid=rid,
            )
        except Exception:  # noqa: BLE001
            logger.exception("SVD L3 prefetch failed open for rid=%s", rid)
            self._cancel_connector_prefetch(rid)
            return
        self.ongoing_prefetch[rid] = True

    def check_prefetch_progress(self, rid: str) -> bool:
        if rid not in self.ongoing_prefetch:
            return True
        check = getattr(self.connector, "check_prefetch_progress", None)
        if not callable(check):
            self.ongoing_prefetch.pop(rid, None)
            return True
        try:
            return bool(check(rid))
        except Exception:  # noqa: BLE001
            logger.exception("SVD L3 prefetch polling failed open for rid=%s", rid)
            self._cancel_connector_prefetch(rid)
            return True

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        if self.ongoing_prefetch.pop(rid, None) is None:
            return 0
        pop = getattr(self.connector, "pop_prefetch_loaded_tokens", None)
        if not callable(pop):
            return 0
        try:
            return int(pop(rid))
        except Exception:  # noqa: BLE001
            logger.exception("failed to collect SVD L3 prefetch for rid=%s", rid)
            self._cancel_connector_prefetch(rid)
            return 0

    def _cancel_connector_prefetch(self, rid: str) -> None:
        self.ongoing_prefetch.pop(rid, None)
        cancel = getattr(self.connector, "cancel_prefetch", None)
        if callable(cancel):
            try:
                cancel(rid)
            except Exception:  # noqa: BLE001
                logger.exception("failed to cancel SVD L3 prefetch for rid=%s", rid)

    def ready_to_load_host_cache(self) -> int:
        """Dispatch an event-capable restore batch, or report no async work."""

        if self.layerwise_restore:
            consumer_index = int(self.connector.start_loading())
            if consumer_index < 0:
                if self._pending_restore_acks:
                    raise RuntimeError(
                        "SVD connector did not start an admitted restore batch"
                    )
                return -1
            if consumer_index in self._inflight_restore_acks:
                raise RuntimeError(
                    f"SVD restore consumer slot {consumer_index} is still in flight"
                )
            self._inflight_restore_acks[consumer_index] = tuple(
                self._pending_restore_acks
            )
            self._pending_restore_acks.clear()
            if (
                getattr(
                    getattr(self.connector, "config", None),
                    "restore_schedule",
                    "layerwise",
                )
                == "synchronous"
            ):
                # Controlled experiment: finish the compact compute restore on
                # the scheduler before forward, then disable per-layer waits.
                # This isolates LayerDoneCounter/forward integration overhead.
                self.connector.layer_done_counter.events[
                    consumer_index
                ].finish_event.synchronize()
                drain = getattr(self.connector, "check_events", None)
                if callable(drain):
                    drain()
                self._drain_layerwise_restore_acks()
                return -1
            return consumer_index
        return -1

    def is_load_back_event_done(self, consumer_index: int) -> bool:
        if not self.layerwise_restore or consumer_index < 0:
            return True
        finish_event = self.connector.layer_done_counter.events[
            consumer_index
        ].finish_event
        if not finish_event.query():
            return False
        self._drain_layerwise_restore_acks()
        return True

    def check_hicache_events(self) -> None:
        """Give connectors with deferred housekeeping a scheduler-tick hook."""

        self._drain_write_through()
        self._schedule_deferred_writes()
        drain = getattr(self.connector, "check_events", None)
        if drain is None:
            drain = getattr(self.connector, "drain", None)
        if callable(drain):
            drain()
        if self.layerwise_restore:
            self._drain_layerwise_restore_acks()

    def _drain_write_through(self, *, synchronize: bool = False) -> int:
        """Consume completed compression ACKs and release their source locks."""

        completed = 0
        for task_id, task in tuple(getattr(self, "ongoing_write_through", {}).items()):
            if not synchronize and not task.future.done():
                continue
            admitted = ()
            try:
                admitted = tuple(task.future.result())
            except Exception:  # noqa: BLE001
                logger.exception(
                    "background SVD write-through failed for chunk=%s",
                    task.chunk_key,
                )
            finally:
                self.ongoing_write_through.pop(task_id, None)
                if self._write_chunk_tasks.get(task.chunk_key) == task_id:
                    self._write_chunk_tasks.pop(task.chunk_key, None)
                self.dec_lock_ref(task.lock_node)
            if task.chunk_key in admitted:
                self._write_through_complete.add(task.chunk_key)
                self._deferred_write_through.pop(task.chunk_key, None)
            elif not synchronize:
                self._defer_write(
                    _DeferredWrite(
                        chunk_key=task.request.chunk_key,
                        token_ids=task.request.token_ids,
                        namespace=task.request.namespace,
                        token_start=task.request.token_start,
                        token_end=task.request.token_end,
                        attempts=task.request.attempts + 1,
                    )
                )
            completed += 1
        return completed

    def _chunk_is_published(self, chunk_key: str) -> bool:
        """Return whether a chunk currently has an L2 or durable L3 copy."""

        pool_exists = getattr(getattr(self.connector, "pool", None), "exists", None)
        if callable(pool_exists) and pool_exists(chunk_key):
            self._write_through_complete.add(chunk_key)
            return True
        if chunk_key not in self._write_through_complete:
            return False
        if self.enable_storage or not callable(pool_exists):
            return True
        # L2-only completion is residency, not durability.  A later L2
        # displacement makes the still-resident L1 chunk eligible again.
        self._write_through_complete.discard(chunk_key)
        return False

    def _defer_write(self, request: _DeferredWrite) -> None:
        """Keep bounded retry metadata without pinning any L1 source slots."""

        if request.attempts >= self._write_retry_limit:
            self.write_through_dropped_count += 1
            return
        if self._chunk_is_published(request.chunk_key):
            return
        current = self._deferred_write_through.pop(request.chunk_key, None)
        if current is not None and current.attempts < request.attempts:
            request = _DeferredWrite(
                chunk_key=request.chunk_key,
                token_ids=request.token_ids,
                namespace=request.namespace,
                token_start=request.token_start,
                token_end=request.token_end,
                attempts=current.attempts,
            )
        if len(self._deferred_write_through) >= self._max_deferred_writes:
            self._deferred_write_through.popitem(last=False)
            self.write_through_dropped_count += 1
        self._deferred_write_through[request.chunk_key] = request
        self.write_through_deferred_count += 1

    @staticmethod
    def _retry_request(request: _DeferredWrite) -> _DeferredWrite:
        return _DeferredWrite(
            chunk_key=request.chunk_key,
            token_ids=request.token_ids,
            namespace=request.namespace,
            token_start=request.token_start,
            token_end=request.token_end,
            attempts=request.attempts + 1,
        )

    def _start_write_batch(self, requests: tuple[_DeferredWrite, ...]) -> int:
        """Capture, lock, and enqueue a bounded group of exact L1 chunks."""

        available = self._max_pending_writes - len(self.ongoing_write_through)
        if available <= 0 or not requests:
            return 0
        batch_limit = self._encode_batch_size if self._batched_write_through else 1
        captures: list[tuple[_DeferredWrite, TreeNode, torch.Tensor]] = []
        seen = set()
        for request in requests[: min(available, batch_limit)]:
            if request.chunk_key in seen:
                continue
            seen.add(request.chunk_key)
            if request.chunk_key in self._write_chunk_tasks:
                continue
            if self._chunk_is_published(request.chunk_key):
                continue

            radix_key = RadixKey(request.token_ids, request.namespace).page_aligned(
                self.page_size
            )
            current = super().match_prefix(MatchPrefixParams(key=radix_key))
            if int(current.device_indices.numel()) != request.token_end:
                # The unlocked candidate was evicted before a worker became free.
                self.write_through_dropped_count += 1
                continue

            lock_node = current.last_device_node
            source_indices = current.device_indices[
                request.token_start : request.token_end
            ].clone()
            self.inc_lock_ref(lock_node)
            captures.append((request, lock_node, source_indices))

        if not captures:
            return 0

        try:
            if len(captures) > 1:
                futures = tuple(
                    self.connector.store_many_async(
                        tuple(
                            {
                                "token_ids": request.token_ids,
                                "device_indices": source_indices,
                                "namespace": request.namespace,
                                "start_token": request.token_start,
                            }
                            for request, _, source_indices in captures
                        )
                    )
                )
                if len(futures) != len(captures):
                    raise RuntimeError(
                        "batched SVD write returned the wrong future count"
                    )
            else:
                request, _, source_indices = captures[0]
                futures = (
                    self.connector.store_async(
                        token_ids=request.token_ids,
                        device_indices=source_indices,
                        namespace=request.namespace,
                        start_token=request.token_start,
                    ),
                )
        except Exception:  # noqa: BLE001
            for request, lock_node, _ in captures:
                self.dec_lock_ref(lock_node)
                self._defer_write(self._retry_request(request))
            logger.exception(
                "failed to enqueue SVD write-through batch chunks=%s",
                [capture[0].chunk_key for capture in captures],
            )
            return 0

        for (request, lock_node, _), future in zip(captures, futures):
            task_id = self._write_task_counter
            self._write_task_counter += 1
            self._deferred_write_through.pop(request.chunk_key, None)
            self.ongoing_write_through[task_id] = _WriteThroughTask(
                chunk_key=request.chunk_key,
                future=future,
                lock_node=lock_node,
                request=request,
            )
            self._write_chunk_tasks[request.chunk_key] = task_id
        return len(captures)

    def _start_write(self, request: _DeferredWrite) -> bool:
        """Lock and enqueue one still-exact L1 chunk if a worker slot is free."""

        if request.chunk_key in self._write_chunk_tasks:
            return True
        if self._chunk_is_published(request.chunk_key):
            return True
        return self._start_write_batch((request,)) == 1

    def _schedule_deferred_writes(self) -> None:
        """Retry unlocked candidates while preserving the in-flight lock bound."""

        candidates = len(self._deferred_write_through)
        while (
            candidates > 0
            and self._deferred_write_through
            and len(self.ongoing_write_through) < self._max_pending_writes
            and (not self._coalesce_write_through or not self.ongoing_write_through)
        ):
            available = self._max_pending_writes - len(self.ongoing_write_through)
            group_size = min(candidates, available, self._encode_batch_size)
            group = []
            for _ in range(group_size):
                _, request = self._deferred_write_through.popitem(last=False)
                group.append(request)
                candidates -= 1
            self._start_write_batch(tuple(group))

    def _schedule_write_through_prefix(
        self,
        token_ids,
        namespace: Any,
    ) -> None:
        """Compress each newly-complete chunk away from the request/eviction path."""

        if not self._write_through_enabled or self.disable:
            return
        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size
        if aligned_end == 0:
            return
        # RadixKey intentionally rejects mixed token-container types.  The
        # live tree uses ``array('q')`` even when a caller supplies a list.
        radix_tokens = array("q", token_ids[:aligned_end])
        identities = self.connector.identities(radix_tokens, namespace)

        requests = []
        for identity in identities:
            chunk_key = identity.key
            if chunk_key in self._write_chunk_tasks:
                continue
            if self._chunk_is_published(chunk_key):
                continue
            prefix_end = int(identity.token_end)
            chunk_start = int(identity.token_start)
            request = _DeferredWrite(
                chunk_key=chunk_key,
                token_ids=radix_tokens[:prefix_end],
                namespace=namespace,
                token_start=chunk_start,
                token_end=prefix_end,
            )
            requests.append(request)

        next_request = 0
        while (
            next_request < len(requests)
            and len(self.ongoing_write_through) < self._max_pending_writes
            and (not self._coalesce_write_through or not self.ongoing_write_through)
        ):
            available = self._max_pending_writes - len(self.ongoing_write_through)
            group_size = min(
                len(requests) - next_request,
                available,
                self._encode_batch_size,
            )
            group = tuple(requests[next_request : next_request + group_size])
            next_request += group_size
            self._start_write_batch(group)
        for request in requests[next_request:]:
            self._defer_write(request)

    def _owns_layerwise_restore(self, rid: str) -> bool:
        return bool(
            getattr(self, "layerwise_restore", False)
            and rid in getattr(self, "ongoing_load_back", {})
        )

    def _finish_layerwise_restore(self, rid: str, node: TreeNode) -> None:
        owned_node = self.ongoing_load_back.get(rid)
        if owned_node is not node:
            return
        self.ongoing_load_back.pop(rid, None)
        self.dec_lock_ref(node)
        # The connector also releases its LoadMarker lease while reaping its
        # event.  Keep this idempotent release so alternate implementations do
        # not retain request-owned lookup state.
        self._release_pending(rid)

    def _drain_layerwise_restore_acks(self) -> int:
        completed = 0
        counter = self.connector.layer_done_counter
        for consumer_index, operations in tuple(self._inflight_restore_acks.items()):
            if not counter.events[consumer_index].finish_event.query():
                continue
            self._inflight_restore_acks.pop(consumer_index)
            for rid, node in operations:
                self._finish_layerwise_restore(rid, node)
                completed += 1
        return completed

    def _synchronize_layerwise_restores(self) -> None:
        """Drain every transfer lock before connector reset or shutdown."""

        counter = self.connector.layer_done_counter

        # Free producer-ring slots before dispatching a batch that was admitted
        # but not yet handed to ready_to_load_host_cache.
        for consumer_index in tuple(self._inflight_restore_acks):
            counter.events[consumer_index].finish_event.synchronize()
        check_events = getattr(self.connector, "check_events", None)
        if callable(check_events):
            check_events()
        self._drain_layerwise_restore_acks()

        if self._pending_restore_acks:
            consumer_index = self.ready_to_load_host_cache()
            if consumer_index >= 0:
                counter.events[consumer_index].finish_event.synchronize()
            if callable(check_events):
                check_events()
            self._drain_layerwise_restore_acks()

        if self._pending_restore_acks or self._inflight_restore_acks:
            raise RuntimeError("failed to drain layerwise SVD restore ownership")

    def clear_storage_backend(self) -> bool:
        """Clear connector-owned L3 when the experimental connector supports it."""

        clear = getattr(self.connector, "clear_storage", None)
        if clear is None:
            clear = getattr(self.connector, "clear", None)
        if clear is None:
            storage = getattr(self.connector, "storage", None)
            backend = getattr(storage, "backend", storage)
            clear = getattr(backend, "clear", None)
        if not callable(clear):
            return False
        try:
            result = clear()
        except Exception:  # noqa: BLE001
            logger.exception("failed to clear SVD chunk storage")
            return False
        return True if result is None else bool(result)

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:
        return False

    def is_backuped(self, node: TreeNode) -> bool:
        """Allow L3 staging after any L1 prefix when SVD storage is present.

        Unlike stock HiCache, SVD has no raw host node to mark ``backuped``;
        the full content-addressed key was captured during ``match_prefix``.
        """

        del node
        return self.enable_storage

    def _release_pending(self, rid: str) -> None:
        try:
            self.connector.release_pending(rid)
        except Exception:  # noqa: BLE001
            logger.exception("failed to release SVD chunk lookup for rid=%s", rid)

    # ------------------------------------------------------------------
    # L1 + external match
    # ------------------------------------------------------------------

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        """Match L1 first, then ask the connector for a longer 4K hit.

        No L1 slots are allocated here.  The marker is retained by request id
        until the scheduler calls :meth:`init_load_back`.
        """

        base_result = super().match_prefix(params)
        key = params.key
        if self.disable or not key or params.req is None:
            return base_result

        rid = params.req.rid
        self._prefetch_candidates.pop(rid, None)
        # Any later match for the same request supersedes an earlier lookup,
        # including a new all-L1 hit that returns before contacting L2/L3.
        if rid in self._load_markers:
            self._load_markers.pop(rid, None)
            self._release_pending(rid)

        lookup_end = (len(key) // self.chunk_size) * self.chunk_size
        device_len = int(base_result.device_indices.numel())
        if lookup_end <= device_len:
            return base_result

        lookup_key = key[:lookup_end]
        token_ids = lookup_key.raw_token_ids()
        self._prefetch_candidates[rid] = _PrefetchCandidate(
            token_ids=tuple(int(token_id) for token_id in token_ids),
            namespace=lookup_key.extra_key,
            device_len=device_len,
        )
        marker = self.connector.lookup(
            token_ids=token_ids,
            namespace=lookup_key.extra_key,
            device_len=device_len,
            rid=rid,
            # Scheduler prefetch owns the L3 -> L2 transition.  The live
            # radix match is intentionally metadata-only against resident L2.
            allow_l3=False,
        )
        if marker is None:
            return base_result

        matched_end = int(marker.matched_end)
        if (
            matched_end <= device_len
            or matched_end > lookup_end
            or matched_end % self.chunk_size != 0
        ):
            logger.warning(
                "ignoring invalid SVD chunk lookup result: rid=%s "
                "device_len=%d matched_end=%d lookup_end=%d",
                rid,
                device_len,
                matched_end,
                lookup_end,
            )
            self._release_pending(rid)
            return base_result

        # Snapshot the key: request fill_ids may keep growing after lookup.
        token_snapshot = token_ids[:]
        detached_key = RadixKey(
            token_snapshot,
            lookup_key.extra_key,
            is_bigram=lookup_key.is_bigram,
        )
        self._load_markers[rid] = _LoadBackMarker(
            connector_marker=marker,
            key=detached_key,
            device_len=device_len,
            matched_end=matched_end,
        )
        self._prefetch_candidates[rid] = _PrefetchCandidate(
            token_ids=tuple(int(token_id) for token_id in token_ids),
            namespace=lookup_key.extra_key,
            device_len=matched_end,
        )
        return MatchResult(
            device_indices=base_result.device_indices,
            last_device_node=base_result.last_device_node,
            last_host_node=base_result.last_device_node,
            best_match_node=base_result.last_device_node,
            host_hit_length=matched_end - device_len,
            full_kv_hit_length=matched_end,
        )

    # ------------------------------------------------------------------
    # External restore -> L1
    # ------------------------------------------------------------------

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:  # type: ignore[override]
        """Allocate destinations, restore a suffix, and commit it to radix.

        A connector without event support finishes all writes before
        ``retrieve`` returns.  The live connector commits exact destinations
        and batches them at ``ready_to_load_host_cache``.  Its default schedule
        synchronizes the completed batch before forward; ``full`` and
        ``layerwise`` retain per-layer event gates for A/B experiments.  A
        second radix match prevents a stale lookup from overwriting a prefix
        another request inserted meanwhile.
        """

        req = params.req
        if req is None:
            return self._empty_load_result(params.best_match_node)

        marker = self._load_markers.pop(req.rid, None)
        if marker is None:
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)

        expected = marker.matched_end - marker.device_len
        if params.host_hit_length != expected:
            logger.warning(
                "SVD chunk load length changed between match and restore: "
                "rid=%s reported=%d expected=%d",
                req.rid,
                params.host_hit_length,
                expected,
            )
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)

        available = self.token_to_kv_pool_allocator.available_size()
        if available < expected:
            self.evict(EvictParams(num_tokens=expected - available))
        slots = self.token_to_kv_pool_allocator.alloc(expected)
        if slots is None:
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)

        if self.layerwise_restore:
            queued = self._queue_layerwise_restore(
                rid=req.rid,
                marker=marker,
                slots=slots,
                expected=expected,
                priority=getattr(req, "priority", 0) or 0,
            )
            if queued is not None:
                return queued

        try:
            fetched = int(self.connector.retrieve(marker.connector_marker, slots))
        except Exception:  # noqa: BLE001
            self._free_slots(slots, marker.device_len)
            self._release_pending(req.rid)
            logger.exception(
                "SVD chunk restore failed open for rid=%s; falling back to prefill",
                req.rid,
            )
            return self._empty_load_result(params.best_match_node)

        if fetched <= 0 or fetched > expected:
            logger.warning(
                "invalid SVD chunk retrieve count: rid=%s fetched=%d expected=%d",
                req.rid,
                fetched,
                expected,
            )
            self._free_slots(slots, marker.device_len)
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)

        restored_end = marker.device_len + fetched
        if fetched % self.page_size != 0 or restored_end % self.chunk_size != 0:
            logger.warning(
                "SVD chunk retrieve did not end on a chunk boundary: "
                "rid=%s device_len=%d fetched=%d",
                req.rid,
                marker.device_len,
                fetched,
            )
            self._free_slots(slots, marker.device_len)
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)

        if fetched < expected:
            self._free_slots(
                slots[fetched:],
                marker.device_len + fetched,
            )
            slots = slots[:fetched]

        result = self._commit_restored_suffix(
            marker=marker,
            slots=slots,
            fetched=fetched,
            priority=getattr(req, "priority", 0) or 0,
        )
        if result is None:
            self._free_slots(slots, marker.device_len)
            self._release_pending(req.rid)
            return self._empty_load_result(params.best_match_node)
        # Concrete retrieve consumes its marker in a finally block.  Keep this
        # explicit idempotent release for injected/alternate connectors too.
        self._release_pending(req.rid)
        return result

    def _queue_layerwise_restore(
        self,
        *,
        rid: str,
        marker: _LoadBackMarker,
        slots: torch.Tensor,
        expected: int,
        priority: int,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Commit exact destinations and defer their writes to batch dispatch.

        Async publication is safe only while the lookup-time L1 prefix is still
        exact.  If another insertion extended that prefix, the synchronous path
        reconstructs first and performs its existing duplicate-row cleanup.
        """

        restored_key = marker.key[: marker.matched_end]
        current = super().match_prefix(MatchPrefixParams(key=restored_key))
        if int(current.device_indices.numel()) != marker.device_len:
            return None

        fetched = int(self.connector.enqueue_retrieve(marker.connector_marker, slots))
        if fetched != expected:
            raise RuntimeError(
                "layerwise SVD connector violated exact admission contract: "
                f"fetched={fetched} expected={expected}"
            )
        result = self._commit_restored_suffix(
            marker=marker,
            slots=slots,
            fetched=expected,
            priority=priority,
            current=current,
        )
        if result is None:
            return None
        _, restored_node = result

        # The request will take its normal radix lock later in admission.  This
        # additional ownership lock covers the gap through the final restore
        # event, including request rejection/abort after init_load_back.
        self.inc_lock_ref(restored_node)
        self.ongoing_load_back[rid] = restored_node
        self._pending_restore_acks.append((rid, restored_node))
        # The exact-prefix guard above and the single-threaded scheduler make
        # these caller-owned destinations authoritative.  Returning them
        # directly avoids ``torch.equal``'s device-to-host scalar fence on the
        # TTFT path; the insert still clones the indices into radix ownership.
        return slots, restored_node

    def _commit_restored_suffix(
        self,
        *,
        marker: _LoadBackMarker,
        slots: torch.Tensor,
        fetched: int,
        priority: int,
        current: Optional[MatchResult] = None,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Insert restored slots without trusting lookup-time tree state."""

        restored_end = marker.device_len + fetched
        restored_key = marker.key[:restored_end]
        if current is None:
            current = super().match_prefix(MatchPrefixParams(key=restored_key))
        current_len = int(current.device_indices.numel())
        if current_len < marker.device_len:
            # The original matched prefix should be request-locked by the
            # scheduler.  If it disappeared, inserting would leave a gap.
            logger.warning(
                "L1 prefix shrank during SVD restore: before=%d now=%d",
                marker.device_len,
                current_len,
            )
            return None

        restored_duplicate = min(current_len - marker.device_len, fetched)
        if current_len < restored_end:
            values = torch.cat(
                [current.device_indices, slots[restored_duplicate:fetched]]
            )
            insert_result = self.insert(
                InsertParams(
                    key=restored_key,
                    value=values,
                    priority=priority,
                )
            )
            # Any restored rows that raced with an already-present radix
            # suffix remain caller-owned; release only those duplicate rows.
            restored_duplicate = min(
                max(insert_result.prefix_len - marker.device_len, 0),
                fetched,
            )

        if restored_duplicate:
            self._free_slots(slots[:restored_duplicate], marker.device_len)

        authoritative = super().match_prefix(MatchPrefixParams(key=restored_key))
        if int(authoritative.device_indices.numel()) != restored_end:
            raise RuntimeError(
                "restored SVD suffix was not fully committed to the radix tree"
            )
        return (
            authoritative.device_indices[marker.device_len : restored_end],
            authoritative.last_device_node,
        )

    def _empty_load_result(
        self, last_node: Optional[TreeNode]
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def _free_slots(self, slots: torch.Tensor, start_pos: int) -> None:
        if slots.numel() == 0:
            return
        self.token_to_kv_pool_allocator.free_segment(
            slots,
            start_pos=start_pos,
        )

    # ------------------------------------------------------------------
    # L1 eviction -> external store
    # ------------------------------------------------------------------

    def _before_evict_leaf(self, node: TreeNode) -> None:  # type: ignore[override]
        """Legacy synchronous fallback for connectors without write-through.

        The live connector compresses complete chunks when they are inserted
        and holds their radix sources until its background future ACKs.  Such
        nodes cannot become evictable while compression is in flight, so this
        hook deliberately performs no codec, transfer, or synchronization.

        A radix node may begin or end inside a compression chunk.  The source
        snapshot therefore walks the full root-to-leaf path, while
        ``start_token`` tells the connector to encode only chunks overlapping
        the leaf being evicted.  Earlier chunks that remain resident in an
        ancestor are neither copied nor recompressed at this boundary.

        Eviction is fail-open: a codec or storage failure drops this lower-tier
        opportunity but must never prevent the allocator from reclaiming L1.
        """

        if self._write_through_enabled:
            # Live write-through retries only while an unlocked candidate is
            # still resident.  Never put slow SVD work or a device-wide fence
            # back onto the allocator's eviction critical path.
            return

        source_read_started = False
        try:
            chain = []
            current = node
            while current is not self.root_node:
                if current is None or current.parent is None:
                    raise RuntimeError("evicted radix leaf is detached from root")
                if current.value is None:
                    raise RuntimeError("evicted radix path contains missing L1 slots")
                chain.append(current)
                current = current.parent
            chain.reverse()

            node_start = sum(len(entry.key) for entry in chain[:-1])
            prefix_end = node_start + len(node.key)
            store_start = (node_start // self.chunk_size) * self.chunk_size
            store_end = (prefix_end // self.chunk_size) * self.chunk_size
            if store_end <= store_start:
                return

            namespace = chain[0].key.extra_key
            token_ids = []
            device_segments = []
            path_offset = 0
            for entry in chain:
                if entry.key.extra_key != namespace:
                    raise RuntimeError("radix path changed namespace")
                token_ids.extend(entry.key.raw_token_ids())
                entry_end = path_offset + len(entry.key)
                overlap_start = max(store_start, path_offset)
                overlap_end = min(store_end, entry_end)
                if overlap_start < overlap_end:
                    device_segments.append(
                        entry.value[
                            overlap_start - path_offset : overlap_end - path_offset
                        ]
                    )
                path_offset = entry_end

            token_ids = token_ids[:store_end]
            device_indices = torch.cat(device_segments, dim=0)
            if (
                len(token_ids) != store_end
                or device_indices.numel() != store_end - store_start
            ):
                raise RuntimeError("radix eviction snapshot is not prefix-complete")

            source_read_started = True
            stored = self.connector.store(
                token_ids=token_ids,
                device_indices=device_indices,
                namespace=namespace,
                start_token=store_start,
            )
            expected_chunks = (store_end - store_start) // self.chunk_size
            if isinstance(stored, (tuple, list, set)) and len(stored) < expected_chunks:
                logger.warning(
                    "SVD eviction admitted only %d/%d touched chunks for "
                    "node=%d token_range=[%d,%d)",
                    len(stored),
                    expected_chunks,
                    node.id,
                    store_start,
                    store_end,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "SVD eviction store failed open for node=%s",
                getattr(node, "id", "unknown"),
            )
        finally:
            # A factorizer may have launched CUDA work before raising.  L1
            # slots cannot return to the allocator until every potential source
            # read is quiescent.  A fence failure deliberately propagates so the
            # base cache does not free possibly-live storage.
            synchronize = getattr(self.connector, "synchronize_source_reads", None)
            if source_read_started and callable(synchronize):
                synchronize()

    def cache_finished_req(
        self,
        req: Req,
        is_insert: bool = True,
        *,
        kv_len_to_handle: int,
    ) -> None:  # type: ignore[override]
        """Insert into L1 and enqueue every newly complete compression chunk."""

        tokens = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        inserted = bool(is_insert and not self.disable_finished_insert)
        try:
            super().cache_finished_req(
                req,
                is_insert=is_insert,
                kv_len_to_handle=kv_len_to_handle,
            )
            if inserted:
                self._schedule_write_through_prefix(tokens, req.extra_key)
        finally:
            self._prefetch_candidates.pop(req.rid, None)
            self._cancel_connector_prefetch(req.rid)
            self._load_markers.pop(req.rid, None)
            self._release_pending(req.rid)

    def cache_unfinished_req(self, req: Req, chunked: bool = False) -> None:  # type: ignore[override]
        """Insert an intermediate prefix and enqueue newly complete chunks."""

        super().cache_unfinished_req(req, chunked=chunked)
        self._schedule_write_through_prefix(req.get_fill_ids(), req.extra_key)


__all__ = [
    "SVDChunkConnectorProtocol",
    "SVDChunkLookupMarker",
    "SVDChunkRadixCache",
]
