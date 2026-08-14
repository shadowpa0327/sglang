"""CPU contract tests for the experimental SVD chunk radix adapter."""

import unittest
from array import array
from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.svd_chunk import SVDChunkRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@dataclass
class _Marker:
    rid: str
    device_len: int
    matched_end: int


@dataclass(frozen=True)
class _Identity:
    key: str
    token_start: int
    token_end: int


class _FakeConnector:
    enable_storage = True

    def __init__(self):
        self.lookup_end = 0
        self.retrieve_count = None
        self.lookup_calls = []
        self.retrieve_calls = []
        self.store_calls = []
        self.released = []
        self.reset_count = 0
        self.shutdown_count = 0
        self.drain_count = 0
        self.clear_count = 0
        self.synchronize_count = 0
        self.raise_store = False
        self.raise_retrieve = False
        self.raise_synchronize = False
        self.on_store = None

    def lookup(self, token_ids, namespace, device_len, rid, allow_l3=True):
        self.lookup_calls.append(
            (list(token_ids), namespace, int(device_len), rid, bool(allow_l3))
        )
        if self.lookup_end <= device_len:
            return None
        return _Marker(rid, int(device_len), self.lookup_end)

    def retrieve(self, marker, dest):
        self.retrieve_calls.append((marker, dest.clone()))
        if self.raise_retrieve:
            raise RuntimeError("injected reconstruction failure")
        if self.retrieve_count is not None:
            return self.retrieve_count
        return marker.matched_end - marker.device_len

    def store(self, token_ids, device_indices, namespace, start_token=0):
        self.store_calls.append(
            (list(token_ids), device_indices.clone(), namespace, start_token)
        )
        if self.on_store is not None:
            self.on_store()
        if self.raise_store:
            raise RuntimeError("injected codec failure")
        return tuple(
            f"chunk-{index}" for index in range(start_token // 8, len(token_ids) // 8)
        )

    def release_pending(self, rid):
        self.released.append(rid)

    def reset(self):
        self.reset_count += 1

    def shutdown(self):
        self.shutdown_count += 1

    def drain(self):
        self.drain_count += 1

    def clear_storage(self):
        self.clear_count += 1
        return True

    def synchronize_source_reads(self):
        self.synchronize_count += 1
        if self.raise_synchronize:
            raise RuntimeError("injected CUDA fence failure")


class _FakeAsyncPool:
    def __init__(self):
        self.resident = set()

    def exists(self, key):
        return key in self.resident


class _FakeAsyncConnector(_FakeConnector):
    """Controllable write-through connector without a background worker."""

    def __init__(self):
        super().__init__()
        self.pool = _FakeAsyncPool()
        self.async_store_calls = []
        self.async_store_many_calls = []
        self.async_store_futures = []
        self.raise_store_async = False
        self.raise_store_many_async = False

    @staticmethod
    def identities(token_ids, namespace):
        del namespace
        return tuple(
            _Identity(
                key=f"chunk-{chunk_index}",
                token_start=chunk_index * 8,
                token_end=(chunk_index + 1) * 8,
            )
            for chunk_index in range(len(token_ids) // 8)
        )

    def store_async(
        self,
        *,
        token_ids,
        device_indices,
        namespace,
        start_token=0,
    ):
        self.async_store_calls.append(
            (list(token_ids), device_indices.clone(), namespace, start_token)
        )
        if self.raise_store_async:
            raise RuntimeError("injected async enqueue failure")
        future = Future()
        self.async_store_futures.append(future)
        return future

    def store_many_async(self, requests):
        copied = tuple(
            (
                list(request["token_ids"]),
                request["device_indices"].clone(),
                request["namespace"],
                request.get("start_token", 0),
            )
            for request in requests
        )
        self.async_store_many_calls.append(copied)
        if self.raise_store_many_async:
            raise RuntimeError("injected batched async enqueue failure")
        futures = tuple(Future() for _ in copied)
        self.async_store_futures.extend(futures)
        return futures


class _FakeAllocator:
    device = torch.device("cpu")

    def __init__(self):
        self.next_slot = 1000
        self.freed = []
        self.kvcache = None
        self.available = 1_000_000

    def available_size(self):
        return self.available

    def get_kvcache(self):
        return self.kvcache

    def alloc(self, count):
        result = torch.arange(
            self.next_slot,
            self.next_slot + count,
            dtype=torch.int64,
        )
        self.next_slot += count
        return result

    def free(self, slots):
        if slots.numel():
            self.freed.append(slots.clone())

    def free_segment(self, slots, *, start_pos):
        self.free(slots)

    def free_segments(self, segments):
        for slots, start_pos in segments:
            self.free_segment(slots, start_pos=start_pos)


class _FakeEvent:
    def __init__(self):
        self.ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class _FakeLayerDoneCounter:
    def __init__(self):
        self.events = [SimpleNamespace(finish_event=_FakeEvent()) for _ in range(3)]


class _FakeKVPool:
    def __init__(self):
        self.k_buffer = [torch.empty(1)]
        self.registered_counter = None

    def register_layer_transfer_counter(self, counter):
        self.registered_counter = counter


class _FakeLayerwiseConnector(_FakeConnector):
    def __init__(self):
        super().__init__()
        self.layer_done_counter = _FakeLayerDoneCounter()
        self.enqueued_retrieves = []
        self.started_batches = 0

    def enqueue_retrieve(self, marker, dest):
        self.enqueued_retrieves.append((marker, dest.clone()))
        return marker.matched_end - marker.device_len

    def start_loading(self):
        if not self.enqueued_retrieves:
            return -1
        consumer_index = self.started_batches % 3
        self.started_batches += 1
        return consumer_index

    def check_events(self):
        self.drain_count += 1


class _FakePrefetchConnector(_FakeConnector):
    def __init__(self):
        super().__init__()
        self.prefetch_calls = []
        self.prefetch_done = False
        self.prefetch_loaded_tokens = 0
        self.cancelled_prefetches = []

    def prefetch(self, *, token_ids, namespace, device_len, rid):
        self.prefetch_calls.append((list(token_ids), namespace, int(device_len), rid))

    def check_prefetch_progress(self, rid):
        return self.prefetch_done

    def pop_prefetch_loaded_tokens(self, rid):
        result = self.prefetch_loaded_tokens
        self.prefetch_loaded_tokens = 0
        return result

    def cancel_prefetch(self, rid):
        self.cancelled_prefetches.append(rid)


class _FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.full((2, 64), -1, dtype=torch.int64)

    def write(self, index, value):
        self.req_to_token[index] = value


class _FakeReq:
    def __init__(self, rid, token_ids, *, namespace="adapter-a"):
        self.rid = rid
        self.origin_input_ids = array("q", token_ids)
        self.output_ids = array("q")
        self.fill_ids = array("q", token_ids)
        self.extra_key = namespace
        self.req_pool_idx = 0
        self.cache_protected_len = 0
        self.last_node = None
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.priority = 0

    def get_fill_ids(self):
        return self.fill_ids


class TestSVDChunkRadixCache(unittest.TestCase):
    def setUp(self):
        self.allocator = _FakeAllocator()
        self.req_pool = _FakeReqToTokenPool()
        self.connector = _FakeConnector()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=4,
        )
        self.cache = SVDChunkRadixCache(
            params,
            server_args=None,
            connector=self.connector,
            chunk_size=8,
        )

    @staticmethod
    def _key(count, namespace="adapter-a"):
        return RadixKey(array("q", range(count)), namespace)

    def _insert_prefix(self, count, first_slot=100):
        self.cache.insert(
            InsertParams(
                key=self._key(count),
                value=torch.arange(
                    first_slot,
                    first_slot + count,
                    dtype=torch.int64,
                ),
            )
        )

    def test_runtime_metrics_snapshot_combines_connector_pool_and_backlog(self):
        self.connector.metrics_snapshot = mock.Mock(
            return_value={
                "restored_chunks": 8,
                "restore_fp8_kernel_launches": 72,
                "restore_tma_kernel_launches": 72,
            }
        )
        self.connector.pool = SimpleNamespace(
            stats=mock.Mock(
                return_value=SimpleNamespace(
                    capacity_bytes=1000,
                    used_bytes=750,
                    available_bytes=250,
                    entry_count=4,
                    pinned_entry_count=1,
                    pin_count=2,
                    hits=9,
                    misses=3,
                    evictions=5,
                    rejected_puts=2,
                )
            )
        )
        self.cache.ongoing_write_through = {1: object(), 2: object()}
        self.cache._deferred_write_through["pending"] = object()
        self.cache.write_through_deferred_count = 7
        self.cache.write_through_dropped_count = 3
        self.cache._max_pending_writes = 8
        self.cache._max_deferred_writes = 64

        snapshot = self.cache.runtime_metrics_snapshot()

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["radix_eviction_policy"], "lru")
        self.assertEqual(snapshot["connector"]["restored_chunks"], 8)
        self.assertEqual(snapshot["connector"]["restore_tma_kernel_launches"], 72)
        self.assertEqual(snapshot["l2_pool"]["used_bytes"], 750)
        self.assertEqual(snapshot["l2_pool"]["evictions"], 5)
        self.assertEqual(
            snapshot["write_through"],
            {
                "inflight": 2,
                "deferred": 1,
                "deferred_events": 7,
                "dropped": 3,
                "max_inflight": 8,
                "max_deferred": 64,
            },
        )

    def test_match_then_restore_inserts_missing_suffix(self):
        self._insert_prefix(4)
        self.connector.lookup_end = 8
        req = _FakeReq("restore", range(12))

        match = self.cache.match_prefix(MatchPrefixParams(key=self._key(12), req=req))
        self.assertEqual(match.device_indices.tolist(), [100, 101, 102, 103])
        self.assertEqual(match.host_hit_length, 4)
        self.assertEqual(match.full_kv_hit_length, 8)
        self.assertEqual(
            self.connector.lookup_calls,
            [(list(range(8)), "adapter-a", 4, "restore", False)],
        )

        loaded, last_node = self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded.tolist(), [1000, 1001, 1002, 1003])
        self.assertIsNotNone(last_node)
        self.assertIn("restore", self.connector.released)
        rematch = self.cache.match_prefix(MatchPrefixParams(key=self._key(8)))
        self.assertEqual(
            rematch.device_indices.tolist(),
            [100, 101, 102, 103, 1000, 1001, 1002, 1003],
        )

    def test_restore_deduplicates_suffix_inserted_after_lookup(self):
        self._insert_prefix(4)
        self.connector.lookup_end = 8
        req = _FakeReq("restore-race", range(8))
        match = self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))

        # Simulate another request completing this prefix before load-back.
        self.cache.insert(
            InsertParams(
                key=self._key(8),
                value=torch.arange(100, 108, dtype=torch.int64),
            )
        )
        loaded, _ = self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=4,
                req=req,
            )
        )
        self.assertEqual(loaded.tolist(), [104, 105, 106, 107])
        self.assertEqual(len(self.allocator.freed), 1)
        self.assertEqual(
            self.allocator.freed[0].tolist(),
            [1000, 1001, 1002, 1003],
        )

    def test_restore_failure_is_fail_open_and_frees_destinations(self):
        self._insert_prefix(4)
        self.connector.lookup_end = 8
        self.connector.raise_retrieve = True
        req = _FakeReq("restore-failure", range(8))
        match = self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))

        loaded, last_node = self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=4,
                req=req,
            )
        )

        self.assertEqual(loaded.numel(), 0)
        self.assertIs(last_node, match.best_match_node)
        self.assertEqual(self.allocator.freed[-1].tolist(), [1000, 1001, 1002, 1003])
        self.assertIn(req.rid, self.connector.released)

    def test_layerwise_restore_commits_then_releases_on_ack(self):
        connector = _FakeLayerwiseConnector()
        allocator = _FakeAllocator()
        allocator.kvcache = _FakeKVPool()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
        )
        cache = SVDChunkRadixCache(
            params,
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        cache.insert(
            InsertParams(
                key=self._key(4),
                value=torch.arange(100, 104, dtype=torch.int64),
            )
        )
        connector.lookup_end = 8
        req = _FakeReq("layerwise", range(8))
        match = cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))

        # The async exact-prefix contract must not validate device indices via
        # a scalar-producing torch.equal() fence on the scheduler/TTFT path.
        with mock.patch("torch.equal", side_effect=AssertionError("GPU fence")):
            loaded, last_node = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=match.best_match_node,
                    host_hit_length=match.host_hit_length,
                    req=req,
                )
            )

        self.assertEqual(loaded.tolist(), [1000, 1001, 1002, 1003])
        self.assertEqual(connector.retrieve_calls, [])
        self.assertEqual(connector.enqueued_retrieves[0][0].rid, req.rid)
        self.assertEqual(connector.enqueued_retrieves[0][1].numel(), 4)
        self.assertIs(
            allocator.kvcache.registered_counter, connector.layer_done_counter
        )
        self.assertEqual(last_node.lock_ref, 1)
        consumer_index = cache.ready_to_load_host_cache()
        self.assertEqual(consumer_index, 0)
        cache.release_aborted_request(req.rid)
        self.assertNotIn(req.rid, connector.released)

        cache.check_hicache_events()
        self.assertEqual(last_node.lock_ref, 1)
        connector.layer_done_counter.events[consumer_index].finish_event.ready = True
        self.assertTrue(cache.is_load_back_event_done(consumer_index))

        self.assertEqual(last_node.lock_ref, 0)
        self.assertIn(req.rid, connector.released)
        self.assertEqual(cache.ongoing_load_back, {})

    def test_synchronous_restore_finishes_before_forward_without_consumer_gate(self):
        connector = _FakeLayerwiseConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=1,
            max_deferred_writes=4,
            write_retry_limit=3,
            restore_schedule="synchronous",
        )
        allocator = _FakeAllocator()
        allocator.kvcache = _FakeKVPool()
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        cache.insert(
            InsertParams(
                key=self._key(4),
                value=torch.arange(100, 104, dtype=torch.int64),
            )
        )
        connector.lookup_end = 8
        req = _FakeReq("synchronous", range(8))
        match = cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        _, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )

        self.assertEqual(last_node.lock_ref, 1)
        self.assertEqual(cache.ready_to_load_host_cache(), -1)
        self.assertTrue(connector.layer_done_counter.events[0].finish_event.query())
        self.assertEqual(last_node.lock_ref, 0)
        self.assertIn(req.rid, connector.released)
        self.assertEqual(cache.ongoing_load_back, {})

    def test_l3_prefetch_uses_full_prefix_after_l2_hit(self):
        connector = _FakePrefetchConnector()
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        cache.insert(
            InsertParams(
                key=self._key(4),
                value=torch.arange(100, 104, dtype=torch.int64),
            )
        )
        connector.lookup_end = 8
        req = _FakeReq("prefetch", range(16))
        match = cache.match_prefix(MatchPrefixParams(key=self._key(16), req=req))

        cache.prefetch_from_storage(
            req.rid,
            match.last_host_node,
            list(range(8, 16)),
            None,
            None,
        )

        self.assertEqual(
            connector.prefetch_calls,
            [(list(range(16)), "adapter-a", 8, req.rid)],
        )
        self.assertFalse(cache.check_prefetch_progress(req.rid))
        connector.prefetch_done = True
        connector.prefetch_loaded_tokens = 8
        self.assertTrue(cache.check_prefetch_progress(req.rid))
        self.assertEqual(cache.pop_prefetch_loaded_tokens(req.rid), 8)
        self.assertTrue(cache.check_prefetch_progress(req.rid))

    def test_l3_prefetch_starts_after_partial_l1_when_l2_misses(self):
        connector = _FakePrefetchConnector()
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        cache.insert(
            InsertParams(
                key=self._key(4),
                value=torch.arange(100, 104, dtype=torch.int64),
            )
        )
        req = _FakeReq("prefetch-l2-miss", range(16))
        match = cache.match_prefix(MatchPrefixParams(key=self._key(16), req=req))
        self.assertIsNot(match.last_host_node, cache.root_node)
        self.assertTrue(cache.is_backuped(match.last_host_node))

        cache.prefetch_from_storage(
            req.rid,
            match.last_host_node,
            list(range(4, 16)),
        )

        self.assertEqual(
            connector.prefetch_calls,
            [(list(range(16)), "adapter-a", 4, req.rid)],
        )

    def test_aborted_request_cancels_l3_prefetch(self):
        connector = _FakePrefetchConnector()
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        req = _FakeReq("prefetch-abort", range(16))
        match = cache.match_prefix(MatchPrefixParams(key=self._key(16), req=req))
        cache.prefetch_from_storage(
            req.rid,
            match.last_host_node,
            list(range(16)),
        )

        cache.release_aborted_request(req.rid)

        self.assertIn(req.rid, connector.cancelled_prefetches)
        self.assertNotIn(req.rid, cache.ongoing_prefetch)

    def test_restore_evicts_only_allocator_shortfall(self):
        self._insert_prefix(4)
        self.connector.lookup_end = 8
        self.allocator.available = 3
        req = _FakeReq("restore-shortfall", range(8))
        match = self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        self.cache.evict = mock.MagicMock()

        self.cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=4,
                req=req,
            )
        )

        self.assertEqual(self.cache.evict.call_args.args[0].num_tokens, 1)

    def test_rematch_releases_marker_when_l1_now_covers_hit(self):
        self._insert_prefix(4)
        self.connector.lookup_end = 8
        req = _FakeReq("rematch", range(8))
        first = self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        self.assertEqual(first.host_hit_length, 4)

        self.cache.insert(
            InsertParams(
                key=self._key(8),
                value=torch.arange(100, 108, dtype=torch.int64),
            )
        )
        second = self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        self.assertEqual(second.host_hit_length, 0)
        self.assertIn("rematch", self.connector.released)

    def test_finished_insert_waits_for_eviction_before_store(self):
        self._insert_prefix(4, first_slot=100)
        req = _FakeReq("finished", range(12))
        self.req_pool.req_to_token[0, :12] = torch.arange(200, 212, dtype=torch.int64)

        self.cache.cache_finished_req(req, kv_len_to_handle=12)

        self.assertEqual(self.connector.store_calls, [])
        freed_before_store = []
        self.connector.on_store = lambda: freed_before_store.append(
            len(self.allocator.freed)
        )
        result = self.cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(len(self.connector.store_calls), 1)
        token_ids, indices, namespace, start_token = self.connector.store_calls[0]
        self.assertEqual(token_ids, list(range(8)))
        self.assertEqual(namespace, "adapter-a")
        self.assertEqual(start_token, 0)
        # The request row's first four slots were duplicate and freed by
        # RadixCache.cache_finished_req.  Eviction must reconstruct the path
        # from the pre-existing parent and the new leaf before freeing either.
        self.assertEqual(
            indices.tolist(),
            [100, 101, 102, 103, 204, 205, 206, 207],
        )
        self.assertEqual(freed_before_store, [1])
        self.assertEqual(self.connector.synchronize_count, 1)

    def test_unfinished_insert_does_not_write_through(self):
        req = _FakeReq("chunked", range(16))
        req.last_node = self.cache.root_node
        self.req_pool.req_to_token[0, :16] = torch.arange(300, 316, dtype=torch.int64)

        self.cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(self.connector.store_calls, [])

    def test_eviction_encodes_only_chunks_overlapping_evicted_leaf(self):
        self._insert_prefix(8, first_slot=100)
        self.cache.insert(
            InsertParams(
                key=self._key(16),
                value=torch.arange(100, 116, dtype=torch.int64),
            )
        )
        self.cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(len(self.connector.store_calls), 1)
        token_ids, indices, namespace, start_token = self.connector.store_calls[0]
        self.assertEqual(token_ids, list(range(16)))
        self.assertEqual(indices.tolist(), list(range(108, 116)))
        self.assertEqual(namespace, "adapter-a")
        self.assertEqual(start_token, 8)

    def test_subchunk_eviction_does_not_create_external_entry(self):
        self._insert_prefix(4)
        self.cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(self.connector.store_calls, [])
        self.assertEqual(self.connector.synchronize_count, 0)

    def test_eviction_store_failure_is_fail_open(self):
        self._insert_prefix(8, first_slot=400)
        self.connector.raise_store = True

        result = self.cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(len(self.connector.store_calls), 1)
        self.assertEqual(self.allocator.freed[-1].tolist(), list(range(400, 408)))
        self.assertEqual(self.connector.synchronize_count, 1)

    def test_source_fence_failure_prevents_slot_reuse(self):
        self._insert_prefix(8, first_slot=500)
        self.connector.raise_synchronize = True

        with self.assertRaisesRegex(RuntimeError, "fence failure"):
            self.cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(self.cache.total_size(), 8)
        self.assertEqual(self.allocator.freed, [])

    def test_lifecycle_and_scheduler_compatibility(self):
        self.connector.lookup_end = 8
        req = _FakeReq("cancel", range(8))
        self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        self.cache.release_aborted_request(req.rid)
        self.assertIn("cancel", self.connector.released)

        self.cache.match_prefix(MatchPrefixParams(key=self._key(8), req=req))
        self.cache.reset()
        self.assertGreaterEqual(self.connector.released.count("cancel"), 2)
        self.assertEqual(self.connector.reset_count, 1)
        self.assertEqual(self.cache.ready_to_load_host_cache(), -1)
        self.cache.check_hicache_events()
        self.assertEqual(self.connector.drain_count, 1)
        self.assertTrue(self.cache.clear_storage_backend())
        self.assertEqual(self.connector.clear_count, 1)
        self.assertEqual(self.cache.ongoing_write_through, {})
        self.assertEqual(self.cache.ongoing_load_back, {})
        self.assertEqual(self.cache.ongoing_prefetch, {})
        self.assertEqual(self.cache.ongoing_backup, {})

        self.cache.shutdown()
        self.assertEqual(self.connector.shutdown_count, 1)
        self.cache.release_host_resources()
        self.assertEqual(self.connector.shutdown_count, 1)

    def test_registry_constructor_lazily_builds_connector(self):
        device_pool = SimpleNamespace(
            k_buffer=[torch.empty(2, dtype=torch.float32)],
            v_buffer=[torch.empty(2, dtype=torch.float32)],
            start_layer=7,
        )
        allocator = _FakeAllocator()
        allocator.kvcache = device_pool
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=_FakeReqToTokenPool(),
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
        )
        server_args = SimpleNamespace(
            hicache_svd_config='{"chunk_tokens": 4096}',
            hicache_size=0,
            hicache_ratio=2.0,
            tp_size=1,
            served_model_name=None,
            model_path="fallback-model",
            revision="revision-a",
        )
        connector = _FakeConnector()
        factory_path = (
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector."
            "create_svd_chunk_connector"
        )
        with mock.patch(factory_path, return_value=connector) as factory:
            cache = SVDChunkRadixCache(
                params,
                server_args,
                model_config=SimpleNamespace(model_path="model-x"),
            )

        self.assertIs(cache.connector, connector)
        args, kwargs = factory.call_args
        self.assertEqual(args, ('{"chunk_tokens": 4096}',))
        self.assertIs(kwargs["device_pool"], device_pool)
        self.assertEqual(kwargs["model_name"], "model-x@revision-a")
        self.assertEqual(kwargs["l2_capacity_bytes"], 32)
        self.assertEqual(kwargs["local_layer_ids"], (7,))
        self.assertTrue(kwargs["preflight_encoder"])

        server_args.hicache_svd_config = (
            '{"chunk_tokens": 4096, "l2_capacity_gb": 0.25}'
        )
        with mock.patch(factory_path, return_value=connector) as factory:
            SVDChunkRadixCache(params, server_args)
        self.assertIsNone(factory.call_args.kwargs["l2_capacity_bytes"])

        server_args.hicache_size = 2
        self.assertEqual(
            SVDChunkRadixCache._derive_l2_capacity_bytes(
                server_args,
                device_pool,
            ),
            2_000_000_000,
        )


class TestSVDChunkAsyncWriteThrough(unittest.TestCase):
    def setUp(self):
        self.allocator = _FakeAllocator()
        self.req_pool = _FakeReqToTokenPool()
        self.connector = _FakeAsyncConnector()
        self.cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=self.connector,
            chunk_size=8,
        )

    @staticmethod
    def _key(count, namespace="adapter-a"):
        return RadixKey(array("q", range(count)), namespace)

    def _finish_prefix(self, count, *, first_slot=200):
        req = _FakeReq(f"finished-{count}", range(count))
        self.req_pool.req_to_token[0, :count] = torch.arange(
            first_slot,
            first_slot + count,
            dtype=torch.int64,
        )
        self.cache.cache_finished_req(req, kv_len_to_handle=count)
        return req

    def test_schedules_each_newly_complete_chunk_once(self):
        self._finish_prefix(16)

        self.assertEqual(len(self.connector.async_store_calls), 2)
        first, second = self.connector.async_store_calls
        self.assertEqual(first[0], list(range(8)))
        self.assertEqual(first[1].tolist(), list(range(200, 208)))
        self.assertEqual(first[2:], ("adapter-a", 0))
        self.assertEqual(second[0], list(range(16)))
        self.assertEqual(second[1].tolist(), list(range(208, 216)))
        self.assertEqual(second[2:], ("adapter-a", 8))
        self.assertEqual(set(self.cache._write_chunk_tasks), {"chunk-0", "chunk-1"})

        # Re-observing the same completed prefix while both jobs are in flight
        # must not enqueue duplicate compression work.
        self.cache._schedule_write_through_prefix(array("q", range(16)), "adapter-a")
        self.assertEqual(len(self.connector.async_store_calls), 2)

        for chunk_index, future in enumerate(self.connector.async_store_futures):
            future.set_result((f"chunk-{chunk_index}",))
        self.cache.check_hicache_events()
        self.assertEqual(self.cache.ongoing_write_through, {})
        self.assertEqual(
            self.cache._write_through_complete,
            {"chunk-0", "chunk-1"},
        )

    def test_batches_live_prefix_chunks_with_independent_locks_and_acks(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=8,
            encode_batch_size=8,
            max_deferred_writes=4,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        req = _FakeReq("batched", range(16))
        self.req_pool.req_to_token[0, :16] = torch.arange(300, 316)

        cache.cache_finished_req(req, kv_len_to_handle=16)

        self.assertEqual(connector.async_store_calls, [])
        self.assertEqual(len(connector.async_store_many_calls), 1)
        first, second = connector.async_store_many_calls[0]
        self.assertEqual(first[0], list(range(8)))
        self.assertEqual(first[1].tolist(), list(range(300, 308)))
        self.assertEqual(first[2:], ("adapter-a", 0))
        self.assertEqual(second[0], list(range(16)))
        self.assertEqual(second[1].tolist(), list(range(308, 316)))
        self.assertEqual(second[2:], ("adapter-a", 8))
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-0", "chunk-1"})
        tasks = tuple(cache.ongoing_write_through.values())
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(task.lock_node.lock_ref > 0 for task in tasks))

        # Re-observation cannot enqueue a duplicate while either child ACK is live.
        cache._schedule_write_through_prefix(array("q", range(16)), "adapter-a")
        self.assertEqual(len(connector.async_store_many_calls), 1)
        connector.async_store_futures[0].set_result(("chunk-0",))
        connector.async_store_futures[1].set_result(("chunk-1",))
        self.assertTrue(all(task.lock_node.lock_ref > 0 for task in tasks))

        cache.check_hicache_events()
        self.assertEqual(cache.ongoing_write_through, {})
        self.assertTrue(all(task.lock_node.lock_ref == 0 for task in tasks))
        self.assertEqual(cache._write_through_complete, {"chunk-0", "chunk-1"})

    def test_coalesces_new_prefix_callbacks_after_active_parent_clears(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=8,
            encode_batch_size=8,
            max_deferred_writes=8,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )

        def finish_prefix(count):
            req = _FakeReq(f"coalesced-{count}", range(count))
            self.req_pool.req_to_token[0, :count] = torch.arange(700, 700 + count)
            cache.cache_finished_req(req, kv_len_to_handle=count)

        # The first callback starts immediately.  Later callbacks expose one new
        # chunk apiece, but must not freeze those chunks into queued B=1 parents.
        finish_prefix(8)
        finish_prefix(16)
        finish_prefix(24)
        self.assertEqual(len(connector.async_store_calls), 1)
        self.assertEqual(connector.async_store_many_calls, [])
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-0"})
        self.assertEqual(set(cache._deferred_write_through), {"chunk-1", "chunk-2"})

        connector.async_store_futures[0].set_result(("chunk-0",))
        cache.check_hicache_events()

        # Reaping the first parent captures both deferred sources into one real
        # connector batch with independent locks and child ownership ACKs.
        self.assertEqual(len(connector.async_store_many_calls), 1)
        first, second = connector.async_store_many_calls[0]
        self.assertEqual(first[0], list(range(16)))
        self.assertEqual(first[1].tolist(), list(range(708, 716)))
        self.assertEqual(first[3], 8)
        self.assertEqual(second[0], list(range(24)))
        self.assertEqual(second[1].tolist(), list(range(716, 724)))
        self.assertEqual(second[3], 16)
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-1", "chunk-2"})
        self.assertEqual(cache._deferred_write_through, {})

        # A later singleton remains deferred until every child from that parent
        # has cleared, even if one child ACK is reaped earlier in this fake.
        finish_prefix(32)
        self.assertEqual(set(cache._deferred_write_through), {"chunk-3"})
        connector.async_store_futures[1].set_result(("chunk-1",))
        cache.check_hicache_events()
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-2"})
        self.assertEqual(set(cache._deferred_write_through), {"chunk-3"})
        self.assertEqual(len(connector.async_store_calls), 1)

        connector.async_store_futures[2].set_result(("chunk-2",))
        cache.check_hicache_events()
        self.assertEqual(len(connector.async_store_calls), 2)
        self.assertEqual(connector.async_store_calls[1][0], list(range(32)))
        self.assertEqual(
            connector.async_store_calls[1][1].tolist(), list(range(724, 732))
        )
        self.assertEqual(connector.async_store_calls[1][3], 24)
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-3"})

        connector.async_store_futures[3].set_result(("chunk-3",))
        cache.check_hicache_events()
        self.assertEqual(cache.ongoing_write_through, {})
        self.assertEqual(
            cache._write_through_complete,
            {"chunk-0", "chunk-1", "chunk-2", "chunk-3"},
        )

    def test_batched_scheduler_flushes_one_single_chunk_tail(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=3,
            encode_batch_size=2,
            max_deferred_writes=4,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        req = _FakeReq("batched-tail", range(24))
        self.req_pool.req_to_token[0, :24] = torch.arange(600, 624)

        cache.cache_finished_req(req, kv_len_to_handle=24)

        self.assertEqual(len(connector.async_store_many_calls), 1)
        self.assertEqual(len(connector.async_store_many_calls[0]), 2)
        self.assertEqual(connector.async_store_calls, [])
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-0", "chunk-1"})
        self.assertEqual(set(cache._deferred_write_through), {"chunk-2"})
        tasks = tuple(cache.ongoing_write_through.values())
        self.assertEqual(len(tasks), 2)
        self.assertTrue(all(task.lock_node.lock_ref > 0 for task in tasks))

        for chunk_index, future in enumerate(connector.async_store_futures):
            future.set_result((f"chunk-{chunk_index}",))
        cache.check_hicache_events()

        self.assertTrue(all(task.future.done() for task in tasks))
        self.assertEqual(len(connector.async_store_calls), 1)
        self.assertEqual(connector.async_store_calls[0][0], list(range(24)))
        self.assertEqual(
            connector.async_store_calls[0][1].tolist(), list(range(616, 624))
        )
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-2"})
        self.assertEqual(cache._deferred_write_through, {})
        tail_task = next(iter(cache.ongoing_write_through.values()))
        self.assertEqual(tail_task.lock_node.lock_ref, 1)

        connector.async_store_futures[2].set_result(("chunk-2",))
        cache.check_hicache_events()
        self.assertEqual(cache.ongoing_write_through, {})
        self.assertEqual(
            cache._write_through_complete,
            {"chunk-0", "chunk-1", "chunk-2"},
        )

    def test_batched_partial_admission_retries_only_unpublished_child(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=2,
            encode_batch_size=2,
            max_deferred_writes=4,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        req = _FakeReq("partial-batch", range(16))
        self.req_pool.req_to_token[0, :16] = torch.arange(400, 416)
        cache.cache_finished_req(req, kv_len_to_handle=16)
        connector.async_store_futures[0].set_result(("chunk-0",))
        connector.async_store_futures[1].set_result(())

        cache.check_hicache_events()

        self.assertIn("chunk-0", cache._write_through_complete)
        self.assertNotIn("chunk-1", cache._write_through_complete)
        self.assertEqual(len(connector.async_store_many_calls), 1)
        self.assertEqual(len(connector.async_store_calls), 1)
        retry = connector.async_store_calls[0]
        self.assertEqual(retry[0], list(range(16)))
        self.assertEqual(retry[1].tolist(), list(range(408, 416)))
        self.assertEqual(retry[3], 8)
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-1"})

    def test_batched_enqueue_failure_releases_all_locks_and_defers_each_chunk(self):
        connector = _FakeAsyncConnector()
        connector.raise_store_many_async = True
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=2,
            encode_batch_size=2,
            max_deferred_writes=4,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )
        req = _FakeReq("failed-batch", range(16))
        self.req_pool.req_to_token[0, :16] = torch.arange(500, 516)

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_radix_cache",
            level="ERROR",
        ):
            cache.cache_finished_req(req, kv_len_to_handle=16)

        self.assertEqual(cache.ongoing_write_through, {})
        self.assertEqual(cache._write_chunk_tasks, {})
        self.assertEqual(set(cache._deferred_write_through), {"chunk-0", "chunk-1"})
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=self._key(8))
            ).last_device_node.lock_ref,
            0,
        )
        self.assertEqual(
            cache.match_prefix(
                MatchPrefixParams(key=self._key(16))
            ).last_device_node.lock_ref,
            0,
        )

    def test_source_lock_is_owned_until_future_ack_is_reaped(self):
        self._finish_prefix(8)
        task = next(iter(self.cache.ongoing_write_through.values()))
        future = self.connector.async_store_futures[0]

        self.assertEqual(task.chunk_key, "chunk-0")
        self.assertEqual(task.lock_node.lock_ref, 1)
        self.assertEqual(self.cache.evictable_size(), 0)

        # Completing the worker future alone is not enough: ownership moves
        # only when the scheduler consumes the ACK.
        future.set_result(("chunk-0",))
        self.assertEqual(task.lock_node.lock_ref, 1)
        self.assertEqual(len(self.cache.ongoing_write_through), 1)

        self.cache.check_hicache_events()
        self.assertEqual(task.lock_node.lock_ref, 0)
        self.assertEqual(self.cache.ongoing_write_through, {})
        self.assertEqual(self.cache._write_chunk_tasks, {})
        self.assertIn("chunk-0", self.cache._write_through_complete)

        # Successful admission suppresses later work for the same chunk.
        self.cache._schedule_write_through_prefix(array("q", range(8)), "adapter-a")
        self.assertEqual(len(self.connector.async_store_calls), 1)

    def test_eviction_callback_does_not_store_or_globally_fence(self):
        self._finish_prefix(8)
        self.connector.async_store_futures[0].set_result(("chunk-0",))
        self.cache.check_hicache_events()
        self.connector.raise_store = True
        self.connector.raise_synchronize = True

        result = self.cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(result.num_tokens_evicted, 8)
        self.assertEqual(self.cache.total_size(), 0)
        self.assertEqual(self.connector.store_calls, [])
        self.assertEqual(self.connector.synchronize_count, 0)
        self.assertEqual(self.allocator.freed[-1].tolist(), list(range(200, 208)))

    def test_failed_future_releases_lock_and_allows_retry(self):
        self._finish_prefix(8)
        task = next(iter(self.cache.ongoing_write_through.values()))
        self.connector.async_store_futures[0].set_exception(
            RuntimeError("injected background failure")
        )

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_radix_cache",
            level="ERROR",
        ):
            self.cache.check_hicache_events()

        self.assertNotIn("chunk-0", self.cache._write_through_complete)
        # The same scheduler tick uses the newly freed bounded worker slot.
        self.assertEqual(len(self.connector.async_store_calls), 2)
        retry_task = next(iter(self.cache.ongoing_write_through.values()))
        self.assertIsNot(retry_task, task)
        # The retry targets the same radix node: the failed task's reference
        # was released before the replacement acquired its bounded reference.
        self.assertIs(retry_task.lock_node, task.lock_node)
        self.assertEqual(retry_task.lock_node.lock_ref, 1)
        self.connector.async_store_futures[1].set_result(("chunk-0",))
        self.cache.check_hicache_events()
        self.assertEqual(retry_task.lock_node.lock_ref, 0)
        self.assertIn("chunk-0", self.cache._write_through_complete)

    def test_enqueue_failure_releases_source_lock_without_task_leak(self):
        self.connector.raise_store_async = True

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_radix_cache",
            level="ERROR",
        ):
            self._finish_prefix(8)

        match = self.cache.match_prefix(MatchPrefixParams(key=self._key(8)))
        self.assertEqual(match.last_device_node.lock_ref, 0)
        self.assertEqual(self.cache.ongoing_write_through, {})
        self.assertEqual(self.cache._write_chunk_tasks, {})
        self.assertNotIn("chunk-0", self.cache._write_through_complete)

    def test_real_connector_limits_source_locks_and_retries_deferred_chunk(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=1,
            max_deferred_writes=4,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )

        req = _FakeReq("bounded", range(16))
        self.req_pool.req_to_token[0, :16] = torch.arange(300, 316)
        cache.cache_finished_req(req, kv_len_to_handle=16)

        self.assertEqual(len(connector.async_store_calls), 1)
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-0"})
        self.assertEqual(set(cache._deferred_write_through), {"chunk-1"})
        self.assertEqual(len(cache.ongoing_write_through), 1)

        connector.async_store_futures[0].set_result(("chunk-0",))
        cache.check_hicache_events()
        self.assertEqual(len(connector.async_store_calls), 2)
        self.assertEqual(set(cache._write_chunk_tasks), {"chunk-1"})
        self.assertEqual(cache._deferred_write_through, {})

    def test_deferred_write_metadata_is_bounded_and_never_pins_source(self):
        connector = _FakeAsyncConnector()
        connector.config = SimpleNamespace(
            chunk_tokens=8,
            max_pending_writes=1,
            max_deferred_writes=1,
            write_retry_limit=3,
        )
        cache = SVDChunkRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=4,
            ),
            server_args=None,
            connector=connector,
            chunk_size=8,
        )

        req = _FakeReq("bounded", range(24))
        self.req_pool.req_to_token[0, :24] = torch.arange(400, 424)
        cache.cache_finished_req(req, kv_len_to_handle=24)

        self.assertEqual(len(connector.async_store_calls), 1)
        self.assertEqual(set(cache._deferred_write_through), {"chunk-2"})
        self.assertEqual(cache.write_through_dropped_count, 1)
        self.assertEqual(
            sum(
                task.lock_node.lock_ref for task in cache.ongoing_write_through.values()
            ),
            1,
        )


class TestSVDServerArgsCompatibility(unittest.TestCase):
    @staticmethod
    def _args(**changes):
        values = dict(
            hicache_svd_config='{"chunk_tokens":4096}',
            enable_hierarchical_cache=True,
            disable_radix_cache=False,
            radix_cache_backend=None,
            enable_lmcache=False,
            enable_flexkv=False,
            disaggregation_decode_enable_radix_cache=False,
            hicache_svd_shadow_config=None,
            hicache_storage_backend=None,
            speculative_algorithm=None,
            tp_size=1,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_rejects_external_backend_and_decode_disaggregation_conflicts(self):
        from sglang.srt.server_args import ServerArgs

        cases = (
            ({"enable_lmcache": True}, "LMCache or FlexKV"),
            ({"enable_flexkv": True}, "LMCache or FlexKV"),
            (
                {"disaggregation_decode_enable_radix_cache": True},
                "decode disaggregation",
            ),
        )
        for changes, message in cases:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, message),
            ):
                ServerArgs._handle_cache_compatibility(self._args(**changes))


if __name__ == "__main__":
    unittest.main()
