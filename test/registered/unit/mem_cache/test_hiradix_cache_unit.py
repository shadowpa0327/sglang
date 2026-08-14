"""Unit tests for srt/mem_cache/hiradix_cache.py KV cache events."""

import json
import os
import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

PAGE_SIZE = 2


class TestHiRadixCacheRuntimeSnapshot(CustomTestCase):
    @staticmethod
    def _queue(size):
        queue = Queue()
        for item in range(size):
            queue.put(item)
        return queue

    def _cache(self):
        cache = object.__new__(HiRadixCache)
        cache.enable_storage = True
        cache.eviction_policy = "lru"
        cache._device_evicted_tokens = 0
        cache._host_evicted_tokens = 0
        cache._dropped_tokens = 0
        cache.ongoing_write_through = {1: object(), 2: object()}
        cache.ongoing_load_back = {3: object()}
        cache.ongoing_prefetch = {"request": object()}
        cache.ongoing_backup = {4: object()}
        cache.work_list = [object()]
        cache.cache_controller = SimpleNamespace(
            write_queue=[object()],
            load_queue=[],
            ack_write_queue=[object(), object()],
            ack_load_queue=[],
            prefetch_queue=self._queue(3),
            backup_queue=self._queue(1),
            prefetch_hit_queue=self._queue(2),
            ack_backup_queue=self._queue(1),
            host_mem_release_queue=self._queue(4),
            prefetch_buffer=self._queue(2),
            extra_host_mem_release_queues={
                "draft": self._queue(2),
                "indexer": self._queue(1),
            },
        )
        return cache

    def test_runtime_snapshot_reports_all_activity(self):
        snapshot = self._cache().hicache_runtime_snapshot()

        json.dumps(snapshot)
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["backend"], "hiradix")
        self.assertTrue(snapshot["enabled"])
        self.assertTrue(snapshot["storage_enabled"])
        self.assertEqual(snapshot["radix_eviction_policy"], "lru")
        self.assertEqual(
            snapshot["eviction_counters"],
            {
                "device_evicted_tokens": 0,
                "host_evicted_tokens": 0,
                "dropped_tokens": 0,
            },
        )
        self.assertEqual(
            snapshot["ongoing"],
            {
                "write": 2,
                "load": 1,
                "prefetch": 1,
                "backup": 1,
                "work": 1,
            },
        )
        self.assertEqual(snapshot["controller_queues"]["ack_write_queue"], 2)
        self.assertEqual(snapshot["controller_queues"]["prefetch_queue"], 3)
        self.assertEqual(snapshot["controller_queues"]["prefetch_buffer"], 2)
        self.assertEqual(snapshot["controller_queues"]["extra_host_mem_release"], 3)
        self.assertFalse(snapshot["quiescent"])

    def test_runtime_snapshot_marks_empty_controller_quiescent(self):
        cache = self._cache()
        cache.enable_storage = False
        cache.ongoing_write_through.clear()
        cache.ongoing_load_back.clear()
        cache.ongoing_prefetch.clear()
        cache.ongoing_backup.clear()
        cache.work_list.clear()
        cache.cache_controller = SimpleNamespace(
            write_queue=[],
            load_queue=[],
            ack_write_queue=[],
            ack_load_queue=[],
        )

        snapshot = cache.hicache_runtime_snapshot()

        self.assertFalse(snapshot["storage_enabled"])
        self.assertTrue(snapshot["quiescent"])
        self.assertTrue(
            all(count == 0 for count in snapshot["controller_queues"].values())
        )

    def test_lifetime_eviction_counters_are_monotonic(self):
        cache = self._cache()

        cache._record_evicted_tokens(device=8, dropped=4)
        cache._record_evicted_tokens(device=2, host=6, dropped=2)

        self.assertEqual(
            cache.hicache_runtime_snapshot()["eviction_counters"],
            {
                "device_evicted_tokens": 10,
                "host_evicted_tokens": 6,
                "dropped_tokens": 6,
            },
        )
        with self.assertRaises(ValueError):
            cache._record_evicted_tokens(host=-1)

    def test_regular_unbacked_eviction_records_device_release_and_drop(self):
        cache = self._cache()
        cache.cache_controller.mem_pool_device_allocator = SimpleNamespace(
            free=mock.Mock()
        )
        cache._record_remove_event = mock.Mock()
        cache._delete_leaf = mock.Mock()
        node = SimpleNamespace(children={}, value=torch.arange(4), id=17)

        self.assertEqual(cache._evict_regular(node), 4)

        cache.cache_controller.mem_pool_device_allocator.free.assert_called_once_with(
            node.value
        )
        self.assertEqual(
            cache.hicache_runtime_snapshot()["eviction_counters"],
            {
                "device_evicted_tokens": 4,
                "host_evicted_tokens": 0,
                "dropped_tokens": 4,
            },
        )


class TestHiRadixCacheKVEvents(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiRadixCache tests.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29601")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    def _build_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE_SIZE,
            hicache_io_backend="direct",
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_through",
        )
        set_global_server_args_for_scheduler(server_args)
        req_to_token_pool = ReqToTokenPool(
            size=10,
            max_context_len=512,
            device="cuda",
            enable_memory_saver=False,
        )
        kv_pool = MHATokenToKVPool(
            size=256,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=4,
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=256,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE_SIZE,
            disable=False,
            enable_kv_cache_events=True,
            tp_cache_group=torch.distributed.group.WORLD,
        )
        cache = HiRadixCache(params, server_args)
        # Disable hit-count-driven write-through; tests back up explicitly.
        cache.write_through_threshold = 1 << 30
        return cache, allocator

    def _insert(self, cache, allocator, tokens):
        key = RadixKey(array("q", tokens))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        return cache.insert(InsertParams(key=key, value=value[: len(tokens)]))

    def _leaf_for(self, cache, tokens):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertIsNot(match.last_device_node, cache.root_node)
        return match.last_device_node

    def _stored_cpu_events(self, cache):
        return [
            e
            for e in cache.take_events()
            if isinstance(e, BlockStored) and e.medium == StorageMedium.CPU
        ]

    def test_split_pending_write_through_publishes_fragments(self):
        cache, allocator = self._build_cache()
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        node = self._leaf_for(cache, [1, 2, 3, 4])
        backed_up = cache.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(cache, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_cpu_events(cache), [])

        cache.writing_check(write_back=True)

        # Both split fragments must be published, with intact parentage.
        stored_cpu = self._stored_cpu_events(cache)
        self.assertEqual(
            [list(e.token_ids) for e in stored_cpu],
            [[1, 2], [3, 4]],
        )
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(stored_cpu[1].parent_block_hash, stored_cpu[0].block_hashes[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
