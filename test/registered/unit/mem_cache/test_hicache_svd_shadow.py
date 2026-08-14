"""CPU tests for the non-owning HiCache SVD shadow hook."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_svd_shadow import (
    HICACHE_SVD_SHADOW_ENV,
    HiCacheSVDShadowConfig,
    HiCacheSVDShadowObserver,
    ShadowEncodeResult,
    maybe_create_hicache_svd_shadow,
    resolve_hicache_svd_shadow_config,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordingEncoder:
    def __init__(self):
        self.chunks = []

    def encode_chunk(self, device_pool, device_indices):
        self.chunks.append((device_pool, device_indices.clone()))
        return ShadowEncodeResult(
            raw_nbytes=64 * device_indices.numel(),
            encoded_nbytes=4 * device_indices.numel(),
            factor_matrices=2,
        )


class _FailingEncoder:
    def encode_chunk(self, device_pool, device_indices):
        raise RuntimeError("injected shadow failure")


class TestHiCacheSVDShadow(unittest.TestCase):
    def test_explicit_and_environment_config(self):
        explicit = resolve_hicache_svd_shadow_config(
            '{"chunk_tokens": 8192, "rank": 16, "bits": 2}'
        )
        self.assertEqual(explicit.chunk_tokens, 8192)
        self.assertEqual(explicit.rank, 16)
        self.assertEqual(explicit.bits_u, 2)
        self.assertEqual(explicit.bits_r, 2)

        with mock.patch.dict(
            os.environ,
            {HICACHE_SVD_SHADOW_ENV: '{"rank": 8}'},
            clear=False,
        ):
            from_env = resolve_hicache_svd_shadow_config()
        self.assertEqual(from_env.rank, 8)
        self.assertIsNone(resolve_hicache_svd_shadow_config(False))

    def test_observer_processes_only_complete_bounded_chunks(self):
        encoder = _RecordingEncoder()
        observer = HiCacheSVDShadowObserver(
            HiCacheSVDShadowConfig(
                chunk_tokens=4096,
                max_chunks_per_backup=1,
                log_every=100,
            ),
            encoder,
        )
        device_pool = object()
        indices = torch.arange(4096 * 2 + 17, dtype=torch.int64)

        observer.observe_backup(device_pool, indices, "kernel")

        self.assertEqual(len(encoder.chunks), 1)
        torch.testing.assert_close(
            encoder.chunks[0][1], torch.arange(4096, dtype=torch.int64)
        )
        metrics = observer.metrics_snapshot()
        self.assertEqual(metrics["backup_calls"], 1)
        self.assertEqual(metrics["encoded_chunks"], 1)
        self.assertEqual(metrics["encoded_tokens"], 4096)
        self.assertEqual(metrics["skipped_tail_tokens"], 17)
        self.assertEqual(metrics["skipped_capped_tokens"], 4096)
        self.assertEqual(metrics["capacity_ratio"], 16.0)

    def test_observer_failure_is_nonfatal_and_disables_itself(self):
        observer = HiCacheSVDShadowObserver(
            HiCacheSVDShadowConfig(log_every=100), _FailingEncoder()
        )

        observer.observe_backup(
            object(), torch.arange(4096, dtype=torch.int64), "kernel"
        )
        observer.observe_backup(
            object(), torch.arange(4096, dtype=torch.int64), "kernel"
        )

        self.assertFalse(observer.active)
        self.assertEqual(observer.metrics.failures, 1)
        self.assertEqual(observer.metrics.backup_calls, 1)

    def test_non_kernel_backend_is_skipped(self):
        encoder = _RecordingEncoder()
        observer = HiCacheSVDShadowObserver(
            HiCacheSVDShadowConfig(log_every=100), encoder
        )
        observer.observe_backup(
            object(), torch.arange(4096, dtype=torch.int64), "direct"
        )
        self.assertEqual(encoder.chunks, [])
        self.assertEqual(observer.metrics.skipped_backend_backups, 1)

    def test_incompatible_cpu_pool_is_disabled_before_optional_import(self):
        device_pool = SimpleNamespace(
            kv_cache_layout="nhd",
            is_quantized_kv_cache=False,
            head_dim=4,
            v_head_dim=4,
            k_buffer=[torch.zeros(4096, 1, 4)],
            v_buffer=[torch.zeros(4096, 1, 4)],
        )
        real_import = __import__

        def reject_optional_import(name, *args, **kwargs):
            if name == "paged_svd":
                raise AssertionError("optional package should not be imported")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_optional_import):
            observer = maybe_create_hicache_svd_shadow(
                device_pool=device_pool,
                page_size=1,
                pool_label="kv",
                config_value=True,
                use_env_fallback=False,
            )
        self.assertIsNone(observer)

    def test_mha_hook_runs_after_raw_backup_and_is_non_owning(self):
        order = []

        class Observer:
            def observe_backup(self, device_pool, device_indices, io_backend):
                order.append("shadow")

        host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host.mtp_draft_device_pools = ()
        host.layout = "layer_first"
        host.can_use_jit = True
        host._hicache_svd_shadow = Observer()
        host.k_data_ptrs = torch.tensor([1], dtype=torch.uint64)
        host.v_data_ptrs = torch.tensor([2], dtype=torch.uint64)
        host.token_stride_size = 8
        host.element_dim = 4
        host.dtype = torch.float16
        device_pool = SimpleNamespace(
            k_data_ptrs=torch.tensor([3], dtype=torch.uint64),
            v_data_ptrs=torch.tensor([4], dtype=torch.uint64),
            k_buffer=[torch.zeros(8, 1, 4)],
            v_buffer=[torch.zeros(8, 1, 4)],
        )

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.jit_transfer_hicache_all_layer",
            side_effect=lambda **_: order.append("raw"),
        ):
            host.backup_from_device_all_layer(
                device_pool,
                torch.arange(4, dtype=torch.int64),
                torch.arange(4, dtype=torch.int64),
                "kernel",
            )

        self.assertEqual(order, ["raw", "shadow"])

    def test_mha_boundary_swallows_unexpected_observer_failure(self):
        class Observer:
            def observe_backup(self, device_pool, device_indices, io_backend):
                raise RuntimeError("injected boundary failure")

        host = MHATokenToKVPoolHost.__new__(MHATokenToKVPoolHost)
        host._hicache_svd_shadow = Observer()
        host._observe_svd_shadow_backup(object(), torch.arange(1), "kernel")
        self.assertIsNone(host._hicache_svd_shadow)


if __name__ == "__main__":
    unittest.main()
