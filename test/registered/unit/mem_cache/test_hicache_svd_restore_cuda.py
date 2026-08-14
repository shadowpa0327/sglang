"""CUDA tests for fused HiCache SVD reconstruction.

PYTHONPATH=python python test/registered/unit/mem_cache/test_hicache_svd_restore_cuda.py
"""

import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.mem_cache.hicache_svd_chunk import (
    QuantizedKVChunk,
    parse_svd_chunk_blob,
    serialize_svd_chunk,
)
from sglang.srt.mem_cache.hicache_svd_codec import (
    QuantizedFactor,
    quantize_svd_factors,
    unpack_signed_rank_values,
)
from sglang.srt.mem_cache.hicache_svd_restore import (
    _fused_quantized_svd_restore_batch_tma_kernel,
    _fused_quantized_svd_restore_int4_rank128_fp8_kernel,
    _fused_quantized_svd_restore_tma_kernel,
    fused_reconstruct_svd_batch_into,
    fused_reconstruct_svd_into,
    fused_svd_restore_available,
)
from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_connector import (
    SVDChunkConnector,
    SVDChunkConnectorConfig,
    build_svd_codec_metadata_from_pool,
    reconstruct_svd_rows_into,
)
from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_pool import SVDChunkPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _reference_dequantize(factor: QuantizedFactor) -> torch.Tensor:
    signed = unpack_signed_rank_values(
        factor.qdata, bits=factor.bits, rank=factor.rank
    ).float()
    if factor.scale_mode == "matrix":
        scale = factor.scale.float()
    else:
        scale = factor.scale.float().repeat_interleave(factor.group_size, dim=1)[
            :, : factor.rank
        ]
    return signed * scale


def _reference_reconstruct(factors) -> torch.Tensor:
    u = _reference_dequantize(factors.u)
    right = _reference_dequantize(factors.right)
    return (u * factors.sigma.float()) @ right.T


def _make_uniform_mapped_factor_batch(
    count,
    rows,
    features,
    *,
    scale_mode="matrix",
):
    """Build production-format factor views over one padded CUDA workspace."""

    originals = []
    parsed_blobs = []
    for chunk_index in range(count):
        torch.manual_seed(1000 + chunk_index)
        factors = quantize_svd_factors(
            torch.randn(rows, 128) * 0.1,
            torch.rand(128),
            torch.randn(features, 128) * 0.1,
            bits_u=4,
            bits_r=4,
            scale_mode=scale_mode,
            group_size_u=32 if scale_mode == "row" else None,
            group_size_r=64 if scale_mode == "row" else None,
        )
        originals.append(factors)
        blob = serialize_svd_chunk(
            QuantizedKVChunk.from_sequences([factors], [factors]),
            codec_metadata={"format": "batch-tma-test", "version": 1},
            namespace="batch",
            end_page_hash=f"{chunk_index:064x}",
        )
        parsed_blobs.append(parse_svd_chunk_blob(blob))

    blob_nbytes = parsed_blobs[0].blob.numel()
    assert all(parsed.blob.numel() == blob_nbytes for parsed in parsed_blobs)
    blob_stride = (blob_nbytes + 63) // 64 * 64
    workspace = torch.empty(
        count * blob_stride,
        dtype=torch.uint8,
        device="cuda",
    )
    mapped = []
    for chunk_index, parsed in enumerate(parsed_blobs):
        device_blob = workspace.narrow(
            0,
            chunk_index * blob_stride,
            blob_nbytes,
        )
        device_blob.copy_(parsed.blob)
        mapped.append(parsed.map_blob(device_blob).chunk.key_layers[0])
    return originals, mapped, workspace


@unittest.skipUnless(
    torch.cuda.is_available() and fused_svd_restore_available(),
    "CUDA and Triton are required",
)
class TestHiCacheSVDFusedRestore(unittest.TestCase):
    def test_cuda_chunk_serialization_matches_cpu_with_one_stream_fence(self):
        torch.manual_seed(3)
        rows, features, rank = 73, 66, 7
        key_factors = quantize_svd_factors(
            torch.randn(rows, rank),
            torch.rand(rank),
            torch.randn(features, rank),
            bits_u=4,
            bits_r=2,
        )
        value_factors = quantize_svd_factors(
            torch.randn(rows, rank),
            torch.rand(rank),
            torch.randn(features, rank),
            bits_u=4,
            bits_r=2,
        )
        cpu_chunk = QuantizedKVChunk.from_sequences([key_factors], [value_factors])
        gpu_chunk = QuantizedKVChunk.from_sequences(
            [key_factors.to("cuda")], [value_factors.to("cuda")]
        )
        metadata = {"format": "cuda-serialization-test", "version": 1}

        cpu_blob = serialize_svd_chunk(
            cpu_chunk,
            codec_metadata=metadata,
            namespace="same",
            end_page_hash="ab" * 32,
        )
        gpu_blob = serialize_svd_chunk(
            gpu_chunk,
            codec_metadata=metadata,
            namespace="same",
            end_page_hash="ab" * 32,
        )

        self.assertTrue(gpu_blob.is_pinned())
        self.assertTrue(torch.equal(gpu_blob, cpu_blob))

    def test_connector_layerwise_restore_uses_event_owned_marker_lifetime(self):
        torch.manual_seed(7)
        rows, rank, features, pool_rows = 4096, 4, 4, 8192
        key_factors = quantize_svd_factors(
            torch.randn(rows, rank) * 0.1,
            torch.rand(rank),
            torch.randn(features, rank) * 0.1,
            bits_u=4,
            bits_r=4,
        )
        value_factors = quantize_svd_factors(
            torch.randn(rows, rank) * 0.1,
            torch.rand(rank),
            torch.randn(features, rank) * 0.1,
            bits_u=4,
            bits_r=4,
        )
        expected_key = _reference_reconstruct(key_factors).half()
        expected_value = _reference_reconstruct(value_factors).half()
        device_pool = SimpleNamespace(
            page_size=16,
            kv_cache_layout="nhd",
            is_quantized_kv_cache=False,
            head_num=2,
            head_dim=2,
            v_head_dim=2,
            start_layer=0,
            k_buffer=[torch.zeros(pool_rows, 2, 2, dtype=torch.float16, device="cuda")],
            v_buffer=[torch.zeros(pool_rows, 2, 2, dtype=torch.float16, device="cuda")],
        )
        config = SVDChunkConnectorConfig(
            chunk_tokens=rows,
            key_rank=rank,
            value_rank=rank,
            bits_u=4,
            bits_r=4,
            niter=0,
        )
        metadata = build_svd_codec_metadata_from_pool(
            config,
            device_pool,
            model_name="layerwise-cuda-test",
            tp_rank=0,
            tp_size=1,
        )
        pool = SVDChunkPool(8 * 1024**2, pin_memory=True)
        connector = SVDChunkConnector(
            config,
            device_pool=device_pool,
            page_size=16,
            pool=pool,
            codec_metadata=metadata,
        )
        tokens = list(range(rows))
        identity = connector.identities(tokens, "base")[0]
        blob = serialize_svd_chunk(
            QuantizedKVChunk.from_sequences([key_factors], [value_factors]),
            codec_metadata=metadata,
            namespace="base",
            end_page_hash=identity.prefix_hash,
        )
        self.assertTrue(pool.put(identity.key, blob))
        marker = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=0,
            rid="layerwise",
            allow_l3=False,
        )
        self.assertIsNotNone(marker)
        slots = torch.arange(rows, pool_rows, dtype=torch.int64, device="cuda")
        self.assertEqual(connector.enqueue_retrieve(marker, slots), rows)
        producer = connector.start_loading()
        self.assertGreaterEqual(producer, 0)
        self.assertFalse(marker.released)
        connector.layer_done_counter.set_consumer(producer)
        connector.layer_done_counter.wait_until(0)
        torch.cuda.current_stream().synchronize()
        connector.check_events()
        self.assertTrue(marker.released)
        metrics = connector.metrics_snapshot()
        self.assertGreater(metrics["restored_blob_nbytes"], 0)
        self.assertGreater(metrics["restore_enqueue_cpu_ms"], 0)
        self.assertGreater(metrics["restore_h2d_ms"], 0)
        self.assertGreater(metrics["restore_gpu_ms"], 0)
        torch.testing.assert_close(
            device_pool.k_buffer[0][slots].reshape(rows, features).cpu(),
            expected_key,
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            device_pool.v_buffer[0][slots].reshape(rows, features).cpu(),
            expected_value,
            atol=5e-3,
            rtol=5e-3,
        )
        connector.shutdown()

    def test_connector_batches_two_tma_operations_with_partial_first_chunks(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("batched TMA FP8 restore requires compute capability 9.0+")

        torch.manual_seed(8)
        chunk_rows, chunk_count, rank, features = 4096, 8, 128, 128
        first_row_start = 13
        restored_rows = chunk_count * chunk_rows - first_row_start
        second_first_row_start = 1
        second_restore_start = chunk_rows + second_first_row_start
        second_restored_rows = (chunk_count - 1) * chunk_rows - second_first_row_start
        pool_rows = (restored_rows + second_restored_rows + 41 + 15) // 16 * 16
        device_pool = SimpleNamespace(
            page_size=16,
            kv_cache_layout="nhd",
            is_quantized_kv_cache=False,
            head_num=1,
            head_dim=features,
            v_head_dim=features,
            start_layer=0,
            k_buffer=[
                torch.zeros(
                    pool_rows,
                    1,
                    features,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            ],
            v_buffer=[
                torch.zeros(
                    pool_rows,
                    1,
                    features,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            ],
        )
        config = SVDChunkConnectorConfig(
            chunk_tokens=chunk_rows,
            key_rank=rank,
            value_rank=rank,
            bits_u=4,
            bits_r=4,
            niter=0,
            restore_compute="fp8_e4m3",
            restore_io="tma",
        )
        metadata = build_svd_codec_metadata_from_pool(
            config,
            device_pool,
            model_name="batch-tma-connector-test",
            tp_rank=0,
            tp_size=1,
        )
        pool = SVDChunkPool(16 * 1024**2, pin_memory=True)
        connector = SVDChunkConnector(
            config,
            device_pool=device_pool,
            page_size=16,
            pool=pool,
            codec_metadata=metadata,
        )
        batch_kernel_cache = (
            _fused_quantized_svd_restore_batch_tma_kernel.device_caches[
                torch.cuda.current_device()
            ][0]
        )
        prewarmed_variant_count = len(batch_kernel_cache)
        tokens = list(range(chunk_count * chunk_rows))
        identities = connector.identities(tokens, "base")
        key_chunks = []
        value_chunks = []
        for chunk_index, identity in enumerate(identities):
            torch.manual_seed(2000 + chunk_index)
            key_factors = quantize_svd_factors(
                torch.randn(chunk_rows, rank) * 0.1,
                torch.rand(rank),
                torch.randn(features, rank) * 0.1,
                bits_u=4,
                bits_r=4,
            )
            value_factors = quantize_svd_factors(
                torch.randn(chunk_rows, rank) * 0.1,
                torch.rand(rank),
                torch.randn(features, rank) * 0.1,
                bits_u=4,
                bits_r=4,
            )
            key_chunks.append(key_factors)
            value_chunks.append(value_factors)
            blob = serialize_svd_chunk(
                QuantizedKVChunk.from_sequences([key_factors], [value_factors]),
                codec_metadata=metadata,
                namespace="base",
                end_page_hash=identity.prefix_hash,
            )
            if chunk_index == 0:
                self.assertTrue(
                    connector._prewarm_restore_mappings(
                        parse_svd_chunk_blob(blob),
                        contiguous_chunks=chunk_count,
                    )
                )
                self.assertTrue(connector._batch_tma_warm_signatures)
            self.assertTrue(pool.put(identity.key, blob))

        marker_a = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=first_row_start,
            rid="batch-tma-a",
            allow_l3=False,
        )
        marker_b = connector.lookup(
            token_ids=tokens,
            namespace="base",
            device_len=second_restore_start,
            rid="batch-tma-b",
            allow_l3=False,
        )
        self.assertIsNotNone(marker_a)
        self.assertIsNotNone(marker_b)
        slots_a = torch.randperm(restored_rows, device="cuda")
        slots_b = restored_rows + torch.randperm(
            second_restored_rows,
            device="cuda",
        )
        self.assertEqual(
            connector.enqueue_retrieve(marker_a, slots_a),
            restored_rows,
        )
        self.assertEqual(
            connector.enqueue_retrieve(marker_b, slots_b),
            second_restored_rows,
        )
        producer = connector.start_loading()
        self.assertGreaterEqual(producer, 0)
        connector.layer_done_counter.events[producer].finish_event.synchronize()
        connector.check_events()
        self.assertEqual(len(batch_kernel_cache), prewarmed_variant_count)

        metrics = connector.metrics_snapshot()
        self.assertEqual(metrics["restored_chunks"], 2 * chunk_count - 1)
        self.assertEqual(metrics["restore_fp8_kernel_launches"], 4)
        self.assertEqual(metrics["restore_tma_kernel_launches"], 4)
        self.assertTrue(marker_a.released)
        self.assertTrue(marker_b.released)

        def assert_operation_samples(
            slots,
            *,
            first_chunk,
            first_row,
            factors_by_chunk,
            destination,
        ):
            first_rows = chunk_rows - first_row
            logical_rows = slots.numel()
            positions = sorted(
                {
                    0,
                    first_rows - 1,
                    first_rows,
                    logical_rows // 2,
                    logical_rows - 1,
                }
            )
            for position in positions:
                if position < first_rows:
                    chunk_index = first_chunk
                    row = first_row + position
                else:
                    relative = position - first_rows
                    chunk_index = first_chunk + 1 + relative // chunk_rows
                    row = relative % chunk_rows
                factors = factors_by_chunk[chunk_index]
                u = (
                    _reference_dequantize(factors.u)[row : row + 1].half()
                    * factors.sigma.half()[None, :]
                ).to(torch.float8_e4m3fn)
                right = (
                    _reference_dequantize(factors.right).half().to(torch.float8_e4m3fn)
                )
                reference = (u.float() @ right.float().T).to(torch.bfloat16)
                actual = destination[slots[position]].reshape(1, features).cpu()
                torch.testing.assert_close(
                    actual,
                    reference,
                    atol=3e-2,
                    rtol=5e-2,
                )

        for factors_by_chunk, destination in (
            (key_chunks, device_pool.k_buffer[0]),
            (value_chunks, device_pool.v_buffer[0]),
        ):
            assert_operation_samples(
                slots_a,
                first_chunk=0,
                first_row=first_row_start,
                factors_by_chunk=factors_by_chunk,
                destination=destination,
            )
            assert_operation_samples(
                slots_b,
                first_chunk=1,
                first_row=second_first_row_start,
                factors_by_chunk=factors_by_chunk,
                destination=destination,
            )
            self.assertTrue(
                bool(
                    torch.all(
                        destination[restored_rows + second_restored_rows :] == 0
                    ).item()
                )
            )
        connector.shutdown()

    def test_mixed_bits_ranks_dtypes_and_scattered_slots(self):
        cases = (
            (2, 8, 15, torch.float16, torch.int32),
            (8, 2, 32, torch.bfloat16, torch.int64),
            (4, 4, 128, torch.float16, torch.int64),
        )
        for bits_u, bits_r, rank, output_dtype, slot_dtype in cases:
            with self.subTest(
                bits_u=bits_u,
                bits_r=bits_r,
                rank=rank,
                output_dtype=output_dtype,
            ):
                torch.manual_seed(rank)
                rows, features, pool_rows = 37, 70, 53
                factors = quantize_svd_factors(
                    torch.randn(rows, rank) * 0.1,
                    torch.rand(rank),
                    torch.randn(features, rank) * 0.1,
                    bits_u=bits_u,
                    bits_r=bits_r,
                    scale_mode="matrix",
                )
                reference = _reference_reconstruct(factors).to(output_dtype)
                device_factors = factors.to("cuda")
                slots = torch.randperm(pool_rows, device="cuda")[:rows].to(slot_dtype)
                destination = torch.full(
                    (pool_rows, 2, features // 2),
                    13.0,
                    dtype=output_dtype,
                    device="cuda",
                )

                fused_reconstruct_svd_into(device_factors, slots, destination)
                torch.cuda.synchronize()

                actual = destination[slots.long()].reshape(rows, features).cpu()
                tolerance = 5e-3 if output_dtype == torch.float16 else 2e-2
                torch.testing.assert_close(
                    actual,
                    reference,
                    atol=tolerance,
                    rtol=tolerance,
                )
                untouched = torch.ones(pool_rows, dtype=torch.bool, device="cuda")
                untouched[slots.long()] = False
                self.assertTrue(bool(torch.all(destination[untouched] == 13.0).item()))

    def test_hopper_fp8_e4m3_restore_matches_fp8_operand_reference(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("FP8-E4M3 restore requires compute capability 9.0+")

        rows, features, rank, pool_rows = 37, 128, 128, 53
        for scale_mode, group_size_u, group_size_r in (
            ("matrix", None, None),
            ("row", 32, 64),
        ):
            with self.subTest(scale_mode=scale_mode):
                torch.manual_seed(128)
                factors = quantize_svd_factors(
                    torch.randn(rows, rank) * 0.1,
                    torch.rand(rank),
                    torch.randn(features, rank) * 0.1,
                    bits_u=4,
                    bits_r=4,
                    scale_mode=scale_mode,
                    group_size_u=group_size_u,
                    group_size_r=group_size_r,
                )
                u_fp8 = (
                    _reference_dequantize(factors.u).half()
                    * factors.sigma.half()[None, :]
                ).to(torch.float8_e4m3fn)
                right_fp8 = (
                    _reference_dequantize(factors.right).half().to(torch.float8_e4m3fn)
                )
                reference = u_fp8.float() @ right_fp8.float().T
                slots = torch.randperm(pool_rows, device="cuda")[:rows].to(torch.int64)
                destination = torch.full(
                    (pool_rows, features),
                    13.0,
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                fused_reconstruct_svd_into(
                    factors.to("cuda"),
                    slots,
                    destination,
                    compute_mode="fp8_e4m3",
                )
                torch.cuda.synchronize()

                actual = destination[slots].float().cpu()
                torch.testing.assert_close(
                    actual,
                    reference,
                    atol=3e-2,
                    rtol=5e-2,
                )
                untouched = torch.ones(
                    pool_rows,
                    dtype=torch.bool,
                    device="cuda",
                )
                untouched[slots] = False
                self.assertTrue(bool(torch.all(destination[untouched] == 13.0).item()))

        kernel_cache = (
            _fused_quantized_svd_restore_int4_rank128_fp8_kernel.device_caches[
                torch.cuda.current_device()
            ][0]
        )
        generated_ptx = "\n".join(
            compiled.asm["ptx"] for compiled in kernel_cache.values()
        )
        self.assertNotIn("cp.async.bulk.tensor", generated_ptx)
        self.assertIn("wgmma.mma_async", generated_ptx)
        self.assertIn(".f16.e4m3.e4m3", generated_ptx)

    def test_hopper_tma_int4_rank128_restore_and_generated_code(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("TMA FP8 restore requires compute capability 9.0+")

        torch.manual_seed(129)
        rows, features, rank, pool_rows = 37, 128, 128, 53
        factors = quantize_svd_factors(
            torch.randn(rows, rank) * 0.1,
            torch.rand(rank),
            torch.randn(features, rank) * 0.1,
            bits_u=4,
            bits_r=4,
            scale_mode="matrix",
        )
        u_fp8 = (
            _reference_dequantize(factors.u).half() * factors.sigma.half()[None, :]
        ).to(torch.float8_e4m3fn)
        right_fp8 = _reference_dequantize(factors.right).half().to(torch.float8_e4m3fn)
        reference = u_fp8.float() @ right_fp8.float().T
        slots = torch.randperm(pool_rows, device="cuda")[:rows].to(torch.int64)
        destination = torch.full(
            (pool_rows, features),
            13.0,
            dtype=torch.bfloat16,
            device="cuda",
        )

        fused_reconstruct_svd_into(
            factors.to("cuda"),
            slots,
            destination,
            compute_mode="fp8_e4m3",
            use_tma=True,
        )
        torch.cuda.synchronize()

        actual = destination[slots].float().cpu()
        torch.testing.assert_close(actual, reference, atol=3e-2, rtol=5e-2)
        untouched = torch.ones(pool_rows, dtype=torch.bool, device="cuda")
        untouched[slots] = False
        self.assertTrue(bool(torch.all(destination[untouched] == 13.0).item()))

        kernel_cache = _fused_quantized_svd_restore_tma_kernel.device_caches[
            torch.cuda.current_device()
        ][0]
        generated_ptx = "\n".join(
            compiled.asm["ptx"] for compiled in kernel_cache.values()
        )
        self.assertGreaterEqual(generated_ptx.count("cp.async.bulk.tensor"), 2)
        self.assertIn("wgmma.mma_async", generated_ptx)
        self.assertIn(".f16.e4m3.e4m3", generated_ptx)

    def test_hopper_batch_tma_restores_uniform_chunks_with_one_launch(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("batched TMA FP8 restore requires compute capability 9.0+")

        rows, features, first_row_start = 128, 128, 13
        for chunk_count, scale_mode in ((2, "matrix"), (8, "matrix"), (2, "row")):
            with self.subTest(chunk_count=chunk_count, scale_mode=scale_mode):
                originals, mapped, workspace = _make_uniform_mapped_factor_batch(
                    chunk_count,
                    rows,
                    features,
                    scale_mode=scale_mode,
                )
                restored_rows = chunk_count * rows - first_row_start
                pool_rows = restored_rows + 37
                slots = torch.randperm(pool_rows, device="cuda")[:restored_rows]
                destination = torch.full(
                    (pool_rows, features),
                    13.0,
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                launched = fused_reconstruct_svd_batch_into(
                    mapped,
                    slots,
                    destination,
                    first_row_start=first_row_start,
                )
                self.assertTrue(launched)
                torch.cuda.synchronize()

                reference_parts = []
                for chunk_index, factors in enumerate(originals):
                    reference = _reference_reconstruct(factors)
                    if chunk_index == 0:
                        reference = reference[first_row_start:]
                    reference_parts.append(reference)
                reference = torch.cat(reference_parts).float()
                actual = destination[slots].float().cpu()
                torch.testing.assert_close(
                    actual,
                    reference,
                    atol=3e-2,
                    rtol=5e-2,
                )
                untouched = torch.ones(
                    pool_rows,
                    dtype=torch.bool,
                    device="cuda",
                )
                untouched[slots] = False
                self.assertTrue(bool(torch.all(destination[untouched] == 13.0).item()))
                self.assertGreater(workspace.numel(), 0)

                if chunk_count == 2 and scale_mode == "matrix":
                    kernel_cache = (
                        _fused_quantized_svd_restore_batch_tma_kernel.device_caches[
                            torch.cuda.current_device()
                        ][0]
                    )
                    compiled_variants = len(kernel_cache)
                    full_slots = torch.arange(
                        chunk_count * rows,
                        dtype=torch.int64,
                        device="cuda",
                    )
                    full_destination = torch.empty(
                        (chunk_count * rows, features),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    self.assertTrue(
                        fused_reconstruct_svd_batch_into(
                            mapped,
                            full_slots,
                            full_destination,
                            first_row_start=0,
                        )
                    )
                    torch.cuda.synchronize()
                    self.assertEqual(len(kernel_cache), compiled_variants)

        kernel_cache = _fused_quantized_svd_restore_batch_tma_kernel.device_caches[
            torch.cuda.current_device()
        ][0]
        generated_ptx = "\n".join(
            compiled.asm["ptx"] for compiled in kernel_cache.values()
        )
        self.assertIn("cp.async.bulk.tensor.3d", generated_ptx)
        self.assertIn("wgmma.mma_async", generated_ptx)
        self.assertIn(".f16.e4m3.e4m3", generated_ptx)

    def test_batch_tma_layout_mismatch_returns_false_for_fallback(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("batched TMA contract checks require capability 9.0+")

        originals, mapped, workspace = _make_uniform_mapped_factor_batch(2, 128, 128)
        mismatched = (mapped[0], originals[1].to("cuda"))
        slots = torch.arange(256, dtype=torch.int64, device="cuda")
        destination = torch.full(
            (256, 128),
            13.0,
            dtype=torch.float16,
            device="cuda",
        )

        self.assertFalse(
            fused_reconstruct_svd_batch_into(
                mismatched,
                slots,
                destination,
            )
        )
        self.assertTrue(bool(torch.all(destination == 13.0).item()))
        self.assertGreater(workspace.numel(), 0)

    def test_tma_restore_fails_closed_for_unsupported_contracts(self):
        if torch.cuda.get_device_capability() < (9, 0):
            self.skipTest("TMA contract checks require compute capability 9.0+")

        torch.manual_seed(130)
        rows, features, rank = 64, 128, 128
        factors = quantize_svd_factors(
            torch.randn(rows, rank) * 0.1,
            torch.rand(rank),
            torch.randn(features, rank) * 0.1,
            bits_u=4,
            bits_r=4,
        ).to("cuda")
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        destination = torch.empty(
            rows,
            features,
            dtype=torch.float16,
            device="cuda",
        )

        with self.assertRaisesRegex(ValueError, "compute_mode='fp8_e4m3'"):
            fused_reconstruct_svd_into(
                factors,
                slots,
                destination,
                compute_mode="fp16",
                use_tma=True,
            )

        rank64 = quantize_svd_factors(
            torch.randn(rows, 64),
            torch.rand(64),
            torch.randn(features, 64),
            bits_u=4,
            bits_r=4,
        ).to("cuda")
        with self.assertRaisesRegex(ValueError, "rank exactly 128"):
            fused_reconstruct_svd_into(
                rank64,
                slots,
                destination,
                compute_mode="fp8_e4m3",
                use_tma=True,
            )

        noncontiguous_storage = torch.empty(
            (rows, 128),
            dtype=torch.uint8,
            device="cuda",
        )
        noncontiguous_qdata = noncontiguous_storage[:, ::2]
        noncontiguous_qdata.copy_(factors.u.qdata)
        noncontiguous_factors = replace(
            factors,
            u=replace(factors.u, qdata=noncontiguous_qdata),
        )
        with self.assertRaisesRegex(ValueError, "contiguous with row stride"):
            fused_reconstruct_svd_into(
                noncontiguous_factors,
                slots,
                destination,
                compute_mode="fp8_e4m3",
                use_tma=True,
            )

        unaligned_storage = torch.empty(
            factors.u.qdata.numel() + 1,
            dtype=torch.uint8,
            device="cuda",
        )
        unaligned_qdata = unaligned_storage[1:].view_as(factors.u.qdata)
        unaligned_qdata.copy_(factors.u.qdata)
        unaligned_factors = replace(
            factors,
            u=replace(factors.u, qdata=unaligned_qdata),
        )
        self.assertEqual(unaligned_qdata.data_ptr() % 16, 1)
        with self.assertRaisesRegex(ValueError, "base must be 16-byte aligned"):
            fused_reconstruct_svd_into(
                unaligned_factors,
                slots,
                destination,
                compute_mode="fp8_e4m3",
                use_tma=True,
            )

    def test_fp8_restore_fails_closed_before_hopper(self):
        factors = quantize_svd_factors(
            torch.randn(16, 16),
            torch.rand(16),
            torch.randn(16, 16),
            bits_u=4,
            bits_r=4,
        ).to("cuda")
        slots = torch.arange(16, dtype=torch.int64, device="cuda")
        destination = torch.empty(16, 16, dtype=torch.float16, device="cuda")

        with (
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(8, 9),
            ),
            self.assertRaisesRegex(RuntimeError, "compute capability 9.0"),
        ):
            fused_reconstruct_svd_into(
                factors,
                slots,
                destination,
                compute_mode="fp8_e4m3",
                use_tma=True,
            )

    def test_partial_row_scaled_restore_bypasses_dense_dequantization(self):
        torch.manual_seed(123)
        chunk_rows, features, rank = 73, 66, 31
        factors = quantize_svd_factors(
            torch.randn(chunk_rows, rank) * 0.1,
            torch.rand(rank),
            torch.randn(features, rank) * 0.1,
            bits_u=4,
            bits_r=2,
            scale_mode="row",
            group_size_u=7,
            group_size_r=9,
        )
        row_start, row_end = 11, 68
        reference = _reference_reconstruct(factors)[row_start:row_end].to(
            torch.bfloat16
        )
        slots = torch.randperm(91)[: row_end - row_start].to(torch.int32)
        destination = torch.full(
            (91, 3, features // 3),
            -9.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with (
            mock.patch.object(
                QuantizedFactor,
                "dequantize",
                side_effect=AssertionError("dense dequantization was called"),
            ),
            mock.patch.object(
                QuantizedFactor,
                "dequantize_rows",
                side_effect=AssertionError("dense row dequantization was called"),
            ),
            torch.cuda.stream(stream),
        ):
            reconstruct_svd_rows_into(
                factors,
                row_start,
                row_end,
                destination_slots=slots,
                destination=destination,
            )
        torch.cuda.current_stream().wait_stream(stream)

        actual = destination[slots.long().cuda()].reshape(row_end - row_start, features)
        torch.testing.assert_close(actual.cpu(), reference, atol=2e-2, rtol=2e-2)

    def test_failed_async_encode_fences_source_stream_before_child_acks(self):
        if not hasattr(torch.cuda, "_sleep"):
            self.skipTest("torch.cuda._sleep is unavailable")

        chunk_tokens = 4096
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                launched = threading.Event()
                tail = torch.cuda.Event()

                class LaunchThenFailEncoder:
                    @staticmethod
                    def _launch():
                        torch.cuda._sleep(1_000_000_000)
                        tail.record(torch.cuda.current_stream())
                        launched.set()
                        raise RuntimeError("injected failure after CUDA source read")

                    def factorize(self, **kwargs):
                        del kwargs
                        self._launch()

                    def factorize_batch(self, **kwargs):
                        del kwargs
                        self._launch()

                slots = batch_size * chunk_tokens
                key = torch.randn(slots, 1, 1, dtype=torch.bfloat16, device="cuda")
                value = torch.randn_like(key)
                device_pool = SimpleNamespace(
                    page_size=64,
                    kv_cache_layout="nhd",
                    is_quantized_kv_cache=False,
                    head_num=1,
                    head_dim=1,
                    v_head_dim=1,
                    k_buffer=[key],
                    v_buffer=[value],
                    start_layer=0,
                )
                config = SVDChunkConnectorConfig(
                    chunk_tokens=chunk_tokens,
                    key_rank=1,
                    value_rank=1,
                    bits_u=4,
                    bits_r=4,
                    encode_batch_size=2,
                    max_pending_writes=2,
                )
                connector = SVDChunkConnector(
                    config,
                    device_pool=device_pool,
                    page_size=64,
                    pool=SVDChunkPool(8 * 1024**2),
                    codec_metadata=build_svd_codec_metadata_from_pool(
                        config,
                        device_pool,
                        model_name="exception-fence-test",
                        tp_rank=0,
                        tp_size=1,
                    ),
                    encoder=LaunchThenFailEncoder(),
                )
                tokens = torch.arange(slots, dtype=torch.int64)
                requests = tuple(
                    {
                        "token_ids": tokens[: (chunk_index + 1) * chunk_tokens],
                        "device_indices": torch.arange(
                            chunk_index * chunk_tokens,
                            (chunk_index + 1) * chunk_tokens,
                            dtype=torch.int64,
                            device="cuda",
                        ),
                        "namespace": "base",
                        "start_token": chunk_index * chunk_tokens,
                    }
                    for chunk_index in range(batch_size)
                )
                futures = connector.store_many_async(requests)
                try:
                    self.assertTrue(launched.wait(timeout=5))
                    self.assertFalse(tail.query())
                    self.assertTrue(all(not future.done() for future in futures))
                    for future in futures:
                        with self.assertRaisesRegex(
                            RuntimeError, "injected failure after CUDA source read"
                        ):
                            future.result(timeout=5)
                    self.assertTrue(tail.query())
                finally:
                    connector.shutdown()


if __name__ == "__main__":
    unittest.main()
