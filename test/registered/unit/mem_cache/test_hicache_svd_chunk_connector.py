"""CPU tests for the HiCache joint-head SVD chunk connector.

python3 test/registered/unit/mem_cache/test_hicache_svd_chunk_connector.py -v
"""

import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector import (
    DirectPagedSVDEncoder,
    LoadMarker,
    SVDChunkConnector,
    SVDChunkConnectorConfig,
    SVDChunkL3Adapter,
    build_svd_codec_metadata_from_pool,
    create_svd_chunk_connector,
    derive_svd_chunk_identities,
    resolve_svd_chunk_connector_config,
)
from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_pool import SVDChunkPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _ExactJointHeadEncoder:
    """Dense CPU reference injected in place of the optional CUDA package."""

    def __init__(self):
        self.calls = []
        self.batch_calls = []

    def factorize(self, **kwargs):
        pages = kwargs["pages"]
        page_ids = kwargs["page_ids"].to(device=pages.device, dtype=torch.int64)
        rank = kwargs["rank"]
        matrix = pages.index_select(0, page_ids).reshape(-1, pages.shape[-1])
        self.calls.append(
            {
                "pages_shape": tuple(pages.shape),
                "matrix_shape": tuple(matrix.shape),
                "rank": rank,
                "seed": kwargs["seed"],
            }
        )
        u, sigma, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        return u[:, :rank], sigma[:rank], vh[:rank].transpose(0, 1)

    def factorize_batch(self, **kwargs):
        page_ids = tuple(kwargs["page_ids"])
        seeds = tuple(kwargs["seeds"])
        self.batch_calls.append(
            {
                "batch_size": len(page_ids),
                "rank": kwargs["rank"],
                "seeds": seeds,
                "page_ids": tuple(ids.clone() for ids in page_ids),
            }
        )
        common = {
            key: value
            for key, value in kwargs.items()
            if key not in ("page_ids", "seeds")
        }
        return tuple(
            self.factorize(**common, page_ids=ids, seed=seed)
            for ids, seed in zip(page_ids, seeds)
        )


class _BlobBackend:
    def __init__(self):
        self.values = {}

    def exists(self, key):
        return key in self.values

    def get(self, key):
        value = self.values.get(key)
        return None if value is None else value.clone()

    def set(self, key, value):
        self.values.setdefault(key, value.clone())
        return True

    def delete(self, key):
        self.values.pop(key, None)
        return True


class _FailOnceBlobBackend(_BlobBackend):
    def __init__(self):
        super().__init__()
        self.remaining_failures = 1

    def set(self, key, value):
        if self.remaining_failures:
            self.remaining_failures -= 1
            return False
        return super().set(key, value)


class _BlockingBlobBackend(_BlobBackend):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def set(self, key, value):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("blocking L3 test backend was not released")
        return super().set(key, value)


class _BlockingReadBlobBackend(_BlobBackend):
    def __init__(self):
        super().__init__()
        self.blocked_key = None
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def get(self, key):
        if key == self.blocked_key:
            self.read_started.set()
            if not self.release_read.wait(timeout=5):
                raise TimeoutError("blocking L3 read test backend was not released")
        return super().get(key)


class _BadShapeEncoder:
    def factorize(self, **kwargs):
        pages = kwargs["pages"]
        rows = kwargs["page_ids"].numel() * pages.shape[1]
        rank = kwargs["rank"]
        return (
            torch.zeros(rows - 1, rank, device=pages.device),
            torch.ones(rank, device=pages.device),
            torch.zeros(pages.shape[2], rank, device=pages.device),
        )


def _device_pool(*, chunks=3, page_size=16):
    generator = torch.Generator().manual_seed(1234)
    slots = chunks * 4096
    key = torch.randn(slots, 2, 2, generator=generator) * 0.2
    value = torch.randn(slots, 2, 3, generator=generator) * 0.2
    return SimpleNamespace(
        page_size=page_size,
        kv_cache_layout="nhd",
        is_quantized_kv_cache=False,
        head_num=2,
        head_dim=2,
        v_head_dim=3,
        k_buffer=[key],
        v_buffer=[value],
        start_layer=7,
    )


def _config(**changes):
    values = dict(
        chunk_tokens=4096,
        key_rank=4,
        value_rank=6,
        bits_u=8,
        bits_r=8,
        scale_mode="matrix",
        niter=0,
    )
    values.update(changes)
    return SVDChunkConnectorConfig(**values)


def _metadata(config, device_pool):
    return build_svd_codec_metadata_from_pool(
        config,
        device_pool,
        model_name="tiny-joint-head-test",
        tp_rank=0,
        tp_size=1,
    )


def _connector(device_pool, *, config=None, pool=None, storage=None, encoder=None):
    config = config or _config()
    return SVDChunkConnector(
        config,
        device_pool=device_pool,
        page_size=device_pool.page_size,
        pool=pool if pool is not None else SVDChunkPool(32 * 1024**2),
        codec_metadata=_metadata(config, device_pool),
        storage=storage,
        encoder=encoder or _ExactJointHeadEncoder(),
    )


class TestSVDChunkConfigAndIdentity(unittest.TestCase):
    def test_config_aliases_and_runtime_capacity_fields(self):
        config = resolve_svd_chunk_connector_config(
            '{"chunk_tokens":8192,"rank":16,"bits":2,'
            '"scale_mode":"row","group_size":8,'
            '"restore_compute":"fp8_e4m3",'
            '"restore_io":"pointer",'
            '"restore_prewarm_chunks":12,'
            '"l2_capacity_gb":1.5,"l3_path":"/tmp/svd-test"}'
        )
        self.assertEqual(config.key_rank, 16)
        self.assertEqual(config.value_rank, 16)
        self.assertEqual(config.bits_u, 2)
        self.assertEqual(config.bits_r, 2)
        self.assertEqual(config.group_size_u, 8)
        self.assertEqual(config.group_size_r, 8)
        self.assertEqual(config.restore_compute, "fp8_e4m3")
        self.assertEqual(config.restore_io, "pointer")
        self.assertEqual(config.restore_prewarm_chunks, 12)
        self.assertEqual(config.l2_capacity_gb, 1.5)
        self.assertEqual(config.l3_path, "/tmp/svd-test")
        self.assertIsNone(resolve_svd_chunk_connector_config(False))

    def test_invalid_experiment_contracts_are_rejected(self):
        for kwargs in (
            {"chunk_tokens": 2048},
            {"key_rank": 129},
            {"bits_u": 3},
            {"scale_mode": "row"},
            {"l2_capacity_gb": -1},
            {"l3_path": ""},
            {"max_pending_writes": 0},
            {"encode_batch_size": 0},
            {"max_deferred_writes": 0},
            {"write_retry_limit": 0},
            {"l3_retry_limit": 0},
            {"restore_schedule": "invalid"},
            {"restore_compute": "fp32"},
            {"restore_io": "invalid"},
            {"restore_io": "tma"},
            {"restore_prewarm_chunks": 0},
            {"log_every": 0},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((TypeError, ValueError)),
            ):
                SVDChunkConnectorConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_svd_chunk_connector_config({"mystery": 1})

    def test_restore_execution_mode_does_not_change_codec_metadata(self):
        pointer = SVDChunkConnectorConfig(
            key_rank=128,
            value_rank=128,
            bits_u=4,
            bits_r=4,
            restore_compute="fp16",
            restore_io="pointer",
        )
        tma = SVDChunkConnectorConfig(
            key_rank=128,
            value_rank=128,
            bits_u=4,
            bits_r=4,
            restore_compute="fp8_e4m3",
            restore_io="tma",
        )

        self.assertEqual(
            _metadata(pointer, _device_pool()), _metadata(tma, _device_pool())
        )

    def test_keys_are_fixed_chunked_cumulative_and_namespaced(self):
        config = _config()
        pool = _device_pool()
        metadata = _metadata(config, pool)
        tokens = list(range(2 * config.chunk_tokens + 13))
        baseline = derive_svd_chunk_identities(
            tokens,
            namespace={"adapter": "base"},
            codec_metadata=metadata,
            chunk_tokens=config.chunk_tokens,
        )
        self.assertEqual(len(baseline), 2)
        self.assertEqual((baseline[0].token_start, baseline[0].token_end), (0, 4096))
        self.assertEqual((baseline[1].token_start, baseline[1].token_end), (4096, 8192))

        changed_first = tokens.copy()
        changed_first[0] += 100_000
        changed = derive_svd_chunk_identities(
            changed_first,
            namespace={"adapter": "base"},
            codec_metadata=metadata,
            chunk_tokens=config.chunk_tokens,
        )
        self.assertNotEqual(baseline[0].key, changed[0].key)
        self.assertNotEqual(baseline[1].key, changed[1].key)

        changed_second = tokens.copy()
        changed_second[5000] += 100_000
        changed = derive_svd_chunk_identities(
            changed_second,
            namespace={"adapter": "base"},
            codec_metadata=metadata,
            chunk_tokens=config.chunk_tokens,
        )
        self.assertEqual(baseline[0].key, changed[0].key)
        self.assertNotEqual(baseline[1].key, changed[1].key)

        namespaced = derive_svd_chunk_identities(
            tokens,
            namespace={"adapter": "lora"},
            codec_metadata=metadata,
            chunk_tokens=config.chunk_tokens,
        )
        self.assertNotEqual(baseline[0].key, namespaced[0].key)

    def test_codec_metadata_records_joint_head_shapes(self):
        pool = _device_pool()
        metadata = _metadata(_config(), pool)
        geometry = metadata["geometry"]
        self.assertEqual(geometry["local_layer_ids"], [7])
        self.assertEqual(geometry["key_features"], 4)
        self.assertEqual(geometry["value_features"], 6)
        self.assertEqual(geometry["layout"], "NHD-joint-head")

    def test_missing_optional_backend_has_an_actionable_error(self):
        missing = ModuleNotFoundError("No module named 'paged_svd'", name="paged_svd")
        with (
            mock.patch(
                "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector."
                "importlib.import_module",
                side_effect=missing,
            ),
            self.assertRaisesRegex(ImportError, "svd_on_paged"),
        ):
            DirectPagedSVDEncoder()._module()

    def test_factory_can_eagerly_preflight_the_encoder(self):
        device_pool = _device_pool(chunks=2)
        with mock.patch.object(DirectPagedSVDEncoder, "preflight") as preflight:
            connector = create_svd_chunk_connector(
                _config(),
                device_pool=device_pool,
                model_name="preflight-test",
                l2_capacity_bytes=1024**2,
                preflight_encoder=True,
            )
        self.assertIsNotNone(connector)
        preflight.assert_called_once_with()

    def test_direct_encoder_invalidates_cached_layout_after_index_mutation(self):
        rows, features, rank = 4, 3, 2
        module = SimpleNamespace(
            prepare_paged_layout=mock.Mock(side_effect=lambda *args: object()),
            prepare_direct_workspace=mock.Mock(side_effect=lambda *args: object()),
            paged_svd_lowrank_direct=mock.Mock(
                return_value=(
                    torch.zeros(rows, rank),
                    torch.ones(rank),
                    torch.zeros(features, rank),
                )
            ),
        )
        encoder = DirectPagedSVDEncoder(module)
        pages = torch.zeros(4, 2, features)
        page_ids = torch.tensor([0, 1], dtype=torch.int64)
        kwargs = dict(
            pages=pages,
            page_ids=page_ids,
            rank=rank,
            niter=0,
            backend="triton",
            precision="ieee",
            seed=0,
        )

        with encoder.chunk_session():
            encoder.factorize(**kwargs)
            encoder.factorize(**kwargs)
            self.assertEqual(module.prepare_paged_layout.call_count, 1)
            page_ids[0] = 1
            encoder.factorize(**kwargs)

        self.assertEqual(module.prepare_paged_layout.call_count, 2)


class TestSVDChunkConnector(unittest.TestCase):
    def test_encode_chunks_matches_independent_svd_blobs_and_batch_permutation(self):
        device_pool = _device_pool(chunks=2)
        config = _config(encode_batch_size=2)
        batch_encoder = _ExactJointHeadEncoder()
        batch = _connector(device_pool, config=config, encoder=batch_encoder)
        tokens = torch.arange(8192, dtype=torch.int64)
        identities = batch.identities(tokens, "base")
        source_indices = (
            torch.arange(4096, dtype=torch.int64),
            torch.arange(4096, 8192, dtype=torch.int64),
        )

        blobs = batch.encode_chunks(
            identities,
            source_indices,
            namespaces=("base", "base"),
        )

        single_encoder = _ExactJointHeadEncoder()
        single = _connector(device_pool, config=config, encoder=single_encoder)
        expected = tuple(
            single.encode_chunk(identity, indices, namespace="base")
            for identity, indices in zip(identities, source_indices)
        )
        self.assertTrue(torch.equal(blobs[0], expected[0]))
        self.assertTrue(torch.equal(blobs[1], expected[1]))
        self.assertEqual(
            sorted(call["seed"] for call in batch_encoder.calls),
            sorted(call["seed"] for call in single_encoder.calls),
        )
        self.assertEqual(
            [(call["batch_size"], call["rank"]) for call in batch_encoder.batch_calls],
            [(2, 4), (2, 6)],
        )
        for call in batch_encoder.batch_calls:
            self.assertEqual(call["page_ids"][0].dtype, torch.int64)
            self.assertEqual(call["page_ids"][0].tolist(), list(range(256)))
            self.assertEqual(call["page_ids"][1].tolist(), list(range(256, 512)))

        permuted_encoder = _ExactJointHeadEncoder()
        permuted = _connector(device_pool, config=config, encoder=permuted_encoder)
        reverse = permuted.encode_chunks(
            identities[::-1],
            source_indices[::-1],
            namespaces=("base", "base"),
        )
        self.assertTrue(torch.equal(reverse[0], blobs[1]))
        self.assertTrue(torch.equal(reverse[1], blobs[0]))

        metrics = batch.metrics_snapshot()
        self.assertEqual(metrics["encode_batches"], 1)
        self.assertEqual(metrics["batched_encoded_chunks"], 2)
        self.assertEqual(metrics["encoded_chunks"], 2)
        batch.shutdown()
        single.shutdown()
        permuted.shutdown()

    def test_store_many_async_batches_missing_chunks_and_fans_out_acks(self):
        device_pool = _device_pool(chunks=2)
        config = _config(encode_batch_size=2)
        encoder = _ExactJointHeadEncoder()
        connector = _connector(device_pool, config=config, encoder=encoder)
        tokens = torch.arange(8192, dtype=torch.int64)
        identities = connector.identities(tokens, "base")

        futures = connector.store_many_async(
            (
                {
                    "token_ids": tokens[:4096],
                    "device_indices": torch.arange(4096, dtype=torch.int64),
                    "namespace": "base",
                    "start_token": 0,
                },
                {
                    "token_ids": tokens,
                    "device_indices": torch.arange(4096, 8192, dtype=torch.int64),
                    "namespace": "base",
                    "start_token": 4096,
                },
            )
        )

        self.assertEqual(len(futures), 2)
        self.assertFalse(futures[0].cancel())
        self.assertEqual(futures[0].result(timeout=5), (identities[0].key,))
        self.assertEqual(futures[1].result(timeout=5), (identities[1].key,))
        self.assertEqual(
            [(call["batch_size"], call["rank"]) for call in encoder.batch_calls],
            [(2, 4), (2, 6)],
        )
        self.assertTrue(connector.pool.exists(identities[0].key))
        self.assertTrue(connector.pool.exists(identities[1].key))
        metrics = connector.metrics_snapshot()
        self.assertEqual(metrics["store_calls"], 2)
        self.assertEqual(metrics["encode_batches"], 1)
        self.assertEqual(metrics["encoded_chunks"], 2)
        connector.shutdown()

    def test_store_many_reports_final_residency_under_l2_pressure(self):
        device_pool = _device_pool(chunks=2)
        config = _config(encode_batch_size=2)
        probe_pool = SVDChunkPool(32 * 1024**2)
        probe = _connector(device_pool, config=config, pool=probe_pool)
        probe.store(
            token_ids=torch.arange(4096, dtype=torch.int64),
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="capacity-probe",
        )
        one_blob_capacity = probe_pool.used_bytes
        probe.shutdown()

        encoder = _ExactJointHeadEncoder()
        pool = SVDChunkPool(one_blob_capacity)
        connector = _connector(
            device_pool,
            config=config,
            pool=pool,
            encoder=encoder,
        )
        tokens = torch.arange(8192, dtype=torch.int64)
        identities = connector.identities(tokens, "base")
        futures = connector.store_many_async(
            (
                {
                    "token_ids": tokens[:4096],
                    "device_indices": torch.arange(4096, dtype=torch.int64),
                    "namespace": "base",
                    "start_token": 0,
                },
                {
                    "token_ids": tokens,
                    "device_indices": torch.arange(4096, 8192, dtype=torch.int64),
                    "namespace": "base",
                    "start_token": 4096,
                },
            )
        )

        self.assertEqual(futures[0].result(timeout=5), (identities[0].key,))
        self.assertEqual(futures[1].result(timeout=5), ())
        self.assertTrue(pool.exists(identities[0].key))
        self.assertFalse(pool.exists(identities[1].key))
        self.assertEqual(
            [(call["batch_size"], call["rank"]) for call in encoder.batch_calls],
            [(2, 4), (2, 6)],
        )
        connector.shutdown()

    def test_store_many_deduplicates_identical_chunk_encoding_and_acks_both(self):
        config = _config(encode_batch_size=2)
        encoder = _ExactJointHeadEncoder()
        connector = _connector(
            _device_pool(chunks=1),
            config=config,
            encoder=encoder,
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        identity = connector.identities(tokens, "base")[0]
        request = {
            "token_ids": tokens,
            "device_indices": torch.arange(4096, dtype=torch.int64),
            "namespace": "base",
        }

        futures = connector.store_many_async((request, request))

        self.assertEqual(futures[0].result(timeout=5), (identity.key,))
        self.assertEqual(futures[1].result(timeout=5), (identity.key,))
        self.assertEqual(encoder.batch_calls, [])
        self.assertEqual(len(encoder.calls), 2)
        connector.shutdown()

    def test_store_many_setup_failure_cannot_launch_hidden_source_read(self):
        class _EnqueueThenRaiseExecutor:
            def __init__(self):
                self.thread = None
                self.worker_errors = []

            def submit(self, function):
                def target():
                    try:
                        function()
                    except BaseException as exc:
                        self.worker_errors.append(exc)

                self.thread = threading.Thread(target=target)
                self.thread.start()
                raise RuntimeError("injected post-enqueue submit failure")

        connector = _connector(_device_pool(chunks=1))
        original_executor = connector._write_executor
        broken_executor = _EnqueueThenRaiseExecutor()
        connector._write_executor = broken_executor
        try:
            with (
                mock.patch.object(connector, "_store_many") as store_many,
                self.assertRaisesRegex(RuntimeError, "post-enqueue"),
            ):
                connector.store_many_async(
                    (
                        {
                            "token_ids": torch.arange(4096, dtype=torch.int64),
                            "device_indices": torch.arange(4096, dtype=torch.int64),
                            "namespace": "base",
                        },
                    )
                )
            broken_executor.thread.join(timeout=5)
            self.assertFalse(broken_executor.thread.is_alive())
            self.assertTrue(broken_executor.worker_errors)
            store_many.assert_not_called()
        finally:
            connector._write_executor = original_executor
            connector.shutdown()

    def test_store_many_rejects_malformed_indices_before_worker_submission(self):
        connector = _connector(_device_pool(chunks=1))
        base = {
            "token_ids": torch.arange(4096, dtype=torch.int64),
            "device_indices": torch.arange(4096, dtype=torch.int64),
            "namespace": "base",
        }
        invalid = (
            ({**base, "device_indices": list(range(4096))}, TypeError),
            ({**base, "device_indices": torch.arange(4096).reshape(64, 64)}, TypeError),
            ({**base, "device_indices": torch.arange(4096).float()}, TypeError),
            ({**base, "device_indices": torch.arange(4095)}, ValueError),
        )
        with mock.patch.object(connector._write_executor, "submit") as submit:
            for request, error in invalid:
                with self.subTest(error=error), self.assertRaises(error):
                    connector.store_many_async((request,))
            submit.assert_not_called()
        connector.shutdown()

    def test_encoder_must_honor_joint_matrix_shape_contract(self):
        device_pool = _device_pool(chunks=2)
        connector = _connector(device_pool, encoder=_BadShapeEncoder())
        with self.assertRaisesRegex(ValueError, "U must have shape"):
            connector.store(
                token_ids=torch.arange(4096, dtype=torch.int64),
                device_indices=torch.arange(4096, dtype=torch.int64),
                namespace="base",
            )

    def test_store_lookup_and_partial_first_chunk_restore(self):
        device_pool = _device_pool()
        encoder = _ExactJointHeadEncoder()
        pool = SVDChunkPool(32 * 1024**2)
        connector = _connector(device_pool, pool=pool, encoder=encoder)
        tokens = torch.arange(8192, dtype=torch.int64)
        source_slots = torch.arange(8192, dtype=torch.int64)

        keys = connector.store(
            token_ids=tokens,
            device_indices=source_slots,
            namespace=("base",),
        )

        self.assertEqual(len(keys), 2)
        self.assertEqual(
            [call["matrix_shape"] for call in encoder.calls],
            [(4096, 4), (4096, 6), (4096, 4), (4096, 6)],
        )
        self.assertEqual([call["rank"] for call in encoder.calls], [4, 6, 4, 6])
        expected_k = device_pool.k_buffer[0][4097:8192].clone()
        expected_v = device_pool.v_buffer[0][4097:8192].clone()
        destination = torch.arange(8192, 8192 + 4095, dtype=torch.int64)
        device_pool.k_buffer[0][destination] = 0
        device_pool.v_buffer[0][destination] = 0

        marker = connector.lookup(
            token_ids=tokens,
            namespace=("base",),
            device_len=4097,
            rid="request-1",
        )

        self.assertIsInstance(marker, LoadMarker)
        self.assertEqual(marker.chunk_start, 4096)
        self.assertEqual(marker.matched_end, 8192)
        self.assertEqual(marker.matched_tokens, 4095)
        self.assertEqual(pool.stats().pin_count, 1)
        restored = connector.retrieve(marker, destination)
        self.assertEqual(restored, 4095)
        self.assertEqual(pool.stats().pin_count, 0)
        self.assertTrue(marker.released)
        torch.testing.assert_close(
            device_pool.k_buffer[0][destination], expected_k, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            device_pool.v_buffer[0][destination], expected_v, atol=2e-2, rtol=2e-2
        )

    def test_store_can_publish_only_the_evicted_suffix_chunk(self):
        device_pool = _device_pool(chunks=2)
        encoder = _ExactJointHeadEncoder()
        connector = _connector(device_pool, encoder=encoder)
        tokens = torch.arange(8192, dtype=torch.int64)

        keys = connector.store(
            token_ids=tokens,
            device_indices=torch.arange(4096, 8192, dtype=torch.int64),
            namespace="base",
            start_token=4096,
        )

        self.assertEqual(len(keys), 1)
        self.assertEqual(len(encoder.calls), 2)  # K and V for one chunk.
        self.assertIsNone(
            connector.lookup(
                token_ids=tokens,
                namespace="base",
                device_len=0,
                rid=None,
            )
        )
        marker = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=4096,
            rid=None,
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker.matched_end, 8192)
        marker.release()

        for invalid_start in (1, 8192):
            with self.subTest(start_token=invalid_start), self.assertRaises(ValueError):
                connector.store(
                    token_ids=tokens,
                    device_indices=torch.arange(8192, dtype=torch.int64),
                    namespace="base",
                    start_token=invalid_start,
                )

    def test_l2_pressure_keeps_earliest_contiguous_chunk(self):
        device_pool = _device_pool(chunks=2)
        probe_pool = SVDChunkPool(32 * 1024**2)
        probe = _connector(device_pool, pool=probe_pool)
        probe.store(
            token_ids=torch.arange(4096, dtype=torch.int64),
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="capacity-probe",
        )
        one_blob_capacity = probe_pool.used_bytes

        pool = SVDChunkPool(one_blob_capacity)
        connector = _connector(device_pool, pool=pool)
        tokens = torch.arange(8192, dtype=torch.int64)
        identities = connector.identities(tokens, "base")
        keys = connector.store(
            token_ids=tokens,
            device_indices=torch.arange(8192, dtype=torch.int64),
            namespace="base",
        )

        self.assertEqual(keys, (identities[0].key,))
        self.assertTrue(pool.exists(identities[0].key))
        self.assertFalse(pool.exists(identities[1].key))
        marker = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid=None,
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker.matched_end, 4096)
        marker.release()

    def test_pending_marker_can_be_released_without_restore(self):
        device_pool = _device_pool(chunks=2)
        pool = SVDChunkPool(32 * 1024**2)
        connector = _connector(device_pool, pool=pool)
        tokens = torch.arange(4096, dtype=torch.int64)
        connector.store(
            token_ids=tokens,
            device_indices=torch.arange(4096),
            namespace=None,
        )
        marker = connector.lookup(
            token_ids=tokens,
            namespace=None,
            device_len=0,
            rid="cancelled",
        )
        self.assertEqual(pool.stats().pin_count, 1)
        connector.release_pending("cancelled")
        self.assertTrue(marker.released)
        self.assertEqual(pool.stats().pin_count, 0)

    def test_l3_blob_can_restore_when_l2_capacity_is_zero(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        expected_k = device_pool.k_buffer[0][:4096].clone()
        expected_v = device_pool.v_buffer[0][:4096].clone()
        keys = connector.store(
            token_ids=tokens,
            device_indices=torch.arange(4096),
            namespace="base",
        )
        self.assertEqual(len(keys), 1)
        self.assertIn(keys[0], backend.values)

        destination = torch.arange(4096, 8192, dtype=torch.int64)
        marker = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid=None,
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker._leases, ())
        self.assertEqual(connector.retrieve(marker, destination), 4096)
        torch.testing.assert_close(
            device_pool.k_buffer[0][destination], expected_k, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            device_pool.v_buffer[0][destination], expected_v, atol=2e-2, rtol=2e-2
        )
        metrics = connector.metrics_snapshot()
        self.assertEqual(metrics["encoded_chunks"], 1)
        self.assertEqual(metrics["encoded_tokens"], 4096)
        self.assertEqual(metrics["restored_chunks"], 1)
        self.assertEqual(metrics["restored_tokens"], 4096)
        self.assertEqual(metrics["l3_hits"], 1)
        self.assertGreater(metrics["compression_ratio"], 1.0)

    def test_file_l3_survives_a_fresh_connector_with_zero_l2(self):
        device_pool = _device_pool(chunks=2)
        tokens = torch.arange(4096, dtype=torch.int64)
        source_slots = torch.arange(4096, dtype=torch.int64)
        destination = torch.arange(4096, 8192, dtype=torch.int64)
        expected_k = device_pool.k_buffer[0][source_slots].clone()
        config = _config(l2_capacity_gb=0)

        with tempfile.TemporaryDirectory(prefix="hicache-svd-l3-test-") as path:
            config = SVDChunkConnectorConfig(**{**vars(config), "l3_path": path})
            writer = create_svd_chunk_connector(
                config,
                device_pool=device_pool,
                model_name="persistent-model@revision-1",
                encoder=_ExactJointHeadEncoder(),
            )
            keys = writer.store(
                token_ids=tokens,
                device_indices=source_slots,
                namespace="base",
            )
            writer.shutdown()

            device_pool.k_buffer[0][destination] = 0
            device_pool.v_buffer[0][destination] = 0
            reader = create_svd_chunk_connector(
                config,
                device_pool=device_pool,
                model_name="persistent-model@revision-1",
                encoder=_ExactJointHeadEncoder(),
            )
            marker = reader.lookup(
                token_ids=tokens,
                namespace="base",
                device_len=0,
                rid="fresh-reader",
            )
            self.assertEqual(len(keys), 1)
            self.assertIsNotNone(marker)
            self.assertEqual(reader.retrieve(marker, destination), 4096)
            self.assertEqual(reader.pool.used_bytes, 0)
            self.assertEqual(reader.metrics_snapshot()["l3_hits"], 1)
            torch.testing.assert_close(
                device_pool.k_buffer[0][destination],
                expected_k,
                atol=2e-2,
                rtol=2e-2,
            )

    def test_zero_l2_without_l3_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires l3_path"):
            create_svd_chunk_connector(
                _config(l2_capacity_gb=0),
                device_pool=_device_pool(),
                model_name="no-storage",
                encoder=_ExactJointHeadEncoder(),
            )

    def test_invalid_l2_blob_falls_through_to_valid_l3(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        writer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        expected = device_pool.k_buffer[0][:4096].clone()
        keys = writer.store(
            token_ids=tokens,
            device_indices=torch.arange(4096),
            namespace="base",
        )

        poisoned_pool = SVDChunkPool(1024**2)
        self.assertTrue(poisoned_pool.put(keys[0], torch.arange(17, dtype=torch.uint8)))
        reader = _connector(
            device_pool,
            pool=poisoned_pool,
            storage=SVDChunkL3Adapter(backend),
        )
        marker = reader.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="poison-fallback",
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker._leases, ())
        destination = torch.arange(4096, 8192)
        self.assertEqual(reader.retrieve(marker, destination), 4096)
        torch.testing.assert_close(
            device_pool.k_buffer[0][destination], expected, atol=2e-2, rtol=2e-2
        )
        metrics = reader.metrics_snapshot()
        self.assertEqual(metrics["invalid_l2_blobs"], 1)
        self.assertEqual(metrics["l3_hits"], 1)
        self.assertEqual(
            reader.store(
                token_ids=tokens,
                device_indices=torch.arange(4096),
                namespace="base",
            ),
            keys,
        )

    def test_corrupt_l3_is_deleted_and_reencoded(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        tokens = torch.arange(4096, dtype=torch.int64)
        slots = torch.arange(4096, dtype=torch.int64)
        writer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        keys = writer.store(token_ids=tokens, device_indices=slots, namespace="base")
        backend.values[keys[0]] = torch.arange(17, dtype=torch.uint8)

        healer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        healed = healer.store(
            token_ids=tokens,
            device_indices=slots,
            namespace="base",
        )

        self.assertEqual(healed, keys)
        self.assertGreater(backend.values[keys[0]].numel(), 17)
        self.assertEqual(healer.metrics_snapshot()["invalid_l3_blobs"], 1)
        marker = healer.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="healed",
        )
        self.assertIsNotNone(marker)
        healer.release_pending("healed")

    def test_l3_failure_is_retried_automatically_from_pinned_l2_blob(self):
        device_pool = _device_pool(chunks=2)
        backend = _FailOnceBlobBackend()
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        slots = torch.arange(4096, dtype=torch.int64)

        key = connector.identities(tokens, "base")[0].key
        first = connector.store(
            token_ids=tokens,
            device_indices=slots,
            namespace="base",
        )

        self.assertEqual(first, ())
        self.assertEqual(connector.metrics_snapshot()["pending_l3_writes"], 1)
        self.assertEqual(connector.pool.stats().pin_count, 1)

        connector.check_events()

        deadline = time.monotonic() + 2
        while key not in backend.values and time.monotonic() < deadline:
            connector.check_events()
            time.sleep(0.001)

        self.assertIn(key, backend.values)
        self.assertEqual(connector.metrics_snapshot()["pending_l3_writes"], 0)
        self.assertEqual(connector.pool.stats().pin_count, 0)
        metrics = connector.metrics_snapshot()
        self.assertEqual(metrics["encoded_chunks"], 1)
        self.assertEqual(metrics["l3_rejections"], 1)
        self.assertEqual(metrics["l3_puts"], 1)
        self.assertEqual(metrics["l3_retry_queued"], 1)
        self.assertEqual(metrics["l3_retry_succeeded"], 1)

    def test_persistent_l3_failure_releases_pinned_l2_after_retry_bound(self):
        device_pool = _device_pool(chunks=2)
        backend = _FailOnceBlobBackend()
        backend.remaining_failures = 100
        connector = _connector(
            device_pool,
            config=_config(l3_retry_limit=2),
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        connector.store(
            token_ids=torch.arange(4096, dtype=torch.int64),
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="base",
        )
        self.assertEqual(connector.pool.stats().pin_count, 1)

        deadline = time.monotonic() + 2
        while connector.pool.stats().pin_count and time.monotonic() < deadline:
            connector.check_events()
            time.sleep(0.001)

        self.assertEqual(connector.pool.stats().pin_count, 0)
        metrics = connector.metrics_snapshot()
        self.assertEqual(metrics["pending_l3_writes"], 0)
        self.assertEqual(metrics["l3_retry_exhausted"], 1)
        # One synchronous publication plus exactly two bounded async retries.
        self.assertEqual(backend.remaining_failures, 97)
        connector.shutdown()

    def test_l3_prefetch_populates_l2_before_live_lookup(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        tokens = torch.arange(4096, dtype=torch.int64)
        slots = torch.arange(4096, dtype=torch.int64)
        writer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        keys = writer.store(
            token_ids=tokens,
            device_indices=slots,
            namespace="base",
        )
        self.assertEqual(len(keys), 1)

        reader = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        self.assertIsNone(
            reader.lookup(
                token_ids=tokens,
                namespace="base",
                device_len=0,
                rid="prefetch",
                allow_l3=False,
            )
        )
        reader.prefetch(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="prefetch",
        )
        deadline = time.monotonic() + 2
        while not reader.check_prefetch_progress("prefetch"):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        self.assertEqual(reader.pop_prefetch_loaded_tokens("prefetch"), 4096)
        marker = reader.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="prefetch",
            allow_l3=False,
        )
        self.assertIsNotNone(marker)
        self.assertEqual(marker.matched_end, 4096)
        marker.release()

    def test_l3_prefetch_publishes_contiguous_prefix_before_slow_tail(self):
        device_pool = _device_pool(chunks=3)
        backend = _BlockingReadBlobBackend()
        tokens = torch.arange(3 * 4096, dtype=torch.int64)
        writer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        keys = writer.store(
            token_ids=tokens,
            device_indices=torch.arange(3 * 4096, dtype=torch.int64),
            namespace="base",
        )
        self.assertEqual(len(keys), 3)
        writer.shutdown()

        backend.blocked_key = keys[1]
        reader = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        try:
            reader.prefetch(
                token_ids=tokens,
                namespace="base",
                device_len=0,
                rid="slow-tail",
            )
            with reader._prefetch_lock:
                state = reader._prefetches["slow-tail"]

            self.assertTrue(backend.read_started.wait(timeout=2))
            deadline = time.monotonic() + 2
            while not (state.futures[0].done() and state.futures[2].done()):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.001)
            self.assertFalse(state.futures[1].done())

            started = time.monotonic()
            self.assertTrue(reader.check_prefetch_progress("slow-tail"))
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(state.futures[1].cancelled())
            self.assertEqual(reader.pop_prefetch_loaded_tokens("slow-tail"), 4096)
            self.assertTrue(reader.pool.exists(keys[0]))
            self.assertFalse(reader.pool.exists(keys[1]))
            self.assertFalse(reader.pool.exists(keys[2]))

            marker = reader.lookup(
                token_ids=tokens,
                namespace="base",
                device_len=0,
                rid="slow-tail",
                allow_l3=False,
            )
            self.assertIsNotNone(marker)
            self.assertEqual(marker.matched_end, 4096)
            marker.release()
            self.assertEqual(reader.metrics_snapshot()["l3_hits"], 1)
        finally:
            backend.release_read.set()
            reader.shutdown()

    def test_releasing_old_lookup_preserves_marker_owned_prefetch_for_rematch(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        tokens = torch.arange(8192, dtype=torch.int64)
        slots = torch.arange(8192, dtype=torch.int64)
        writer = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        keys = writer.store(
            token_ids=tokens,
            device_indices=slots,
            namespace="base",
        )
        self.assertEqual(len(keys), 2)

        first_blob = backend.values[keys[0]]
        reader_pool = SVDChunkPool(first_blob.numel())
        self.assertTrue(reader_pool.put(keys[0], first_blob))
        reader = _connector(
            device_pool,
            pool=reader_pool,
            storage=SVDChunkL3Adapter(backend),
        )
        old_marker = reader.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="mixed-prefetch",
            allow_l3=False,
        )
        self.assertIsNotNone(old_marker)
        self.assertEqual(old_marker.matched_end, 4096)

        reader.prefetch(
            token_ids=tokens,
            namespace="base",
            device_len=4096,
            rid="mixed-prefetch",
        )
        deadline = time.monotonic() + 2
        while not reader.check_prefetch_progress("mixed-prefetch"):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        self.assertEqual(reader.pop_prefetch_loaded_tokens("mixed-prefetch"), 4096)
        # The first marker pins the only L2 entry, so the second chunk must stay
        # request-owned until the scheduler's rematch consumes it.
        self.assertFalse(reader.pool.exists(keys[1]))

        reader.release_pending("mixed-prefetch")
        marker = reader.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="mixed-prefetch",
            allow_l3=False,
        )

        self.assertIsNotNone(marker)
        self.assertEqual(marker.matched_end, 8192)
        marker.release()
        reader.shutdown()
        writer.shutdown()

    def test_store_async_moves_l3_publication_off_caller_thread(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlobBackend()
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        key = connector.identities(tokens, "base")[0].key
        future = connector.store_async(
            token_ids=tokens,
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="base",
        )
        self.assertIn(key, future.result(timeout=5))
        deadline = time.monotonic() + 2
        while key not in backend.values:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        connector.shutdown()

    def test_async_l3_keeps_l2_pinned_until_durable_ack(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlockingBlobBackend()
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        key = connector.identities(tokens, "base")[0].key
        future = connector.store_async(
            token_ids=tokens,
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="base",
        )

        self.assertTrue(backend.started.wait(timeout=5))
        self.assertIn(key, future.result(timeout=5))
        self.assertEqual(connector.pool.stats().pin_count, 1)
        backend.release.set()
        deadline = time.monotonic() + 2
        while connector.pool.stats().pin_count and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(connector.pool.stats().pin_count, 0)
        connector.shutdown()

    def test_async_store_waits_for_l3_ack_when_l2_rejects_blob(self):
        device_pool = _device_pool(chunks=2)
        backend = _BlockingBlobBackend()
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(0),
            storage=SVDChunkL3Adapter(backend),
        )
        tokens = torch.arange(4096, dtype=torch.int64)
        key = connector.identities(tokens, "base")[0].key
        future = connector.store_async(
            token_ids=tokens,
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="base",
        )

        self.assertTrue(backend.started.wait(timeout=5))
        self.assertFalse(future.done())
        backend.release.set()
        self.assertIn(key, future.result(timeout=5))
        connector.shutdown()

    def test_reset_releases_pending_l3_retry_lease(self):
        device_pool = _device_pool(chunks=2)
        backend = _FailOnceBlobBackend()
        backend.remaining_failures = 10
        connector = _connector(
            device_pool,
            pool=SVDChunkPool(32 * 1024**2),
            storage=SVDChunkL3Adapter(backend),
        )
        connector.store(
            token_ids=torch.arange(4096, dtype=torch.int64),
            device_indices=torch.arange(4096, dtype=torch.int64),
            namespace="base",
        )
        self.assertEqual(connector.pool.stats().pin_count, 1)

        connector.reset()

        self.assertEqual(connector.pool.stats().pin_count, 0)
        self.assertEqual(connector.pool.stats().entry_count, 0)
        self.assertEqual(connector.metrics_snapshot()["pending_l3_writes"], 0)

    def test_prewarm_restore_mappings_covers_cumulative_prefix(self):
        connector = SVDChunkConnector.__new__(SVDChunkConnector)
        connector._restore_device = SimpleNamespace(type="cuda")
        connector._load_workspaces = [None, None, None]
        workspace = torch.empty(192, dtype=torch.uint8)
        connector._ensure_restore_workspace = mock.Mock(return_value=workspace)
        connector._map_restore_workspace_blob = mock.Mock()
        parsed = SimpleNamespace(blob=torch.empty(10, dtype=torch.uint8))

        with mock.patch(
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector."
            "torch.cuda.device"
        ):
            self.assertTrue(
                connector._prewarm_restore_mappings(parsed, contiguous_chunks=3)
            )

        self.assertEqual(
            connector._ensure_restore_workspace.call_args_list,
            [mock.call(0, 192), mock.call(1, 192), mock.call(2, 192)],
        )
        offsets = [
            call.args[2].storage_offset()
            for call in connector._map_restore_workspace_blob.call_args_list
        ]
        self.assertEqual(offsets, [0, 64, 128] * 3)

    def test_tma_restore_contract_accepts_rank128_int4(self):
        config = SVDChunkConnectorConfig(
            key_rank=128,
            value_rank=128,
            bits_u=4,
            bits_r=4,
            restore_compute="fp8_e4m3",
            restore_io="tma",
        )

        self.assertEqual(config.restore_io, "tma")

        with self.assertRaisesRegex(ValueError, "restore_compute='fp8_e4m3'"):
            SVDChunkConnectorConfig(
                key_rank=128,
                value_rank=128,
                bits_u=4,
                bits_r=4,
                restore_compute="fp16",
                restore_io="tma",
            )

    def test_prewarm_restore_mappings_treats_cuda_oom_as_cache_miss(self):
        connector = SVDChunkConnector.__new__(SVDChunkConnector)
        connector._restore_device = SimpleNamespace(type="cuda")
        connector._load_workspaces = [None, None, None]
        connector._ensure_restore_workspace = mock.Mock(
            side_effect=torch.OutOfMemoryError("test prewarm OOM")
        )
        connector._map_restore_workspace_blob = mock.Mock()
        parsed = SimpleNamespace(blob=torch.empty(10, dtype=torch.uint8))

        with mock.patch(
            "sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector."
            "torch.cuda.device"
        ):
            self.assertFalse(
                connector._prewarm_restore_mappings(parsed, contiguous_chunks=8)
            )

        connector._map_restore_workspace_blob.assert_not_called()

    def test_store_caps_cumulative_mapping_prewarm(self):
        device_pool = _device_pool(chunks=3)
        connector = _connector(
            device_pool,
            config=_config(restore_prewarm_chunks=2),
        )
        connector._prewarm_restore_mappings = mock.Mock(return_value=True)

        connector.store(
            token_ids=torch.arange(3 * 4096, dtype=torch.int64),
            device_indices=torch.arange(3 * 4096, dtype=torch.int64),
            namespace="base",
        )

        self.assertEqual(
            [
                call.kwargs["contiguous_chunks"]
                for call in connector._prewarm_restore_mappings.call_args_list
            ],
            [2, 2, 1],
        )

    def test_factory_is_disabled_or_builds_default_pool(self):
        device_pool = _device_pool(chunks=2)
        self.assertIsNone(
            create_svd_chunk_connector(
                False,
                device_pool=device_pool,
                model_name="tiny",
                l2_capacity_bytes=1,
            )
        )
        connector = create_svd_chunk_connector(
            {
                "key_rank": 4,
                "value_rank": 6,
                "bits": 8,
                "niter": 0,
            },
            device_pool=device_pool,
            model_name="tiny",
            l2_capacity_bytes=1024**2,
            encoder=_ExactJointHeadEncoder(),
        )
        self.assertIsInstance(connector, SVDChunkConnector)
        self.assertEqual(connector.pool.capacity_bytes, 1024**2)

        decimal_gb = create_svd_chunk_connector(
            {
                "key_rank": 4,
                "value_rank": 6,
                "bits": 8,
                "niter": 0,
                "l2_capacity_gb": 0.000001,
            },
            device_pool=device_pool,
            model_name="tiny",
            encoder=_ExactJointHeadEncoder(),
        )
        self.assertEqual(decimal_gb.pool.capacity_bytes, 1000)

    def test_store_requires_the_authoritative_aligned_prefix(self):
        connector = _connector(_device_pool(chunks=2))
        with self.assertRaisesRegex(ValueError, "chunk-aligned"):
            connector.store(
                token_ids=torch.arange(4097),
                device_indices=torch.arange(4097),
                namespace=None,
            )
        with self.assertRaisesRegex(ValueError, "cover token_ids"):
            connector.store(
                token_ids=torch.arange(4096),
                device_indices=torch.arange(4095),
                namespace=None,
            )


if __name__ == "__main__":
    unittest.main()
