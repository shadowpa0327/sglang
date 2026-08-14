"""CPU contract tests for the experimental HiCache SVD factor codec.

python test/registered/unit/mem_cache/test_hicache_svd_codec.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.hicache_svd_codec import (
    estimate_quantized_kv_chunk_nbytes,
    estimate_quantized_svd_nbytes,
    fold_kv_heads,
    pack_signed_rank_values,
    page_ids_from_device_indices,
    quantize_factor,
    quantize_svd_factors,
    raw_kv_chunk_nbytes,
    unfold_kv_heads,
    unpack_signed_rank_values,
    view_nhd_as_joint_pages,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCacheSVDCodec(unittest.TestCase):
    def test_joint_head_fold_and_asymmetric_value_width(self):
        key = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
        value = torch.arange(4 * 2 * 5).reshape(4, 2, 5)

        flat_key = fold_kv_heads(key)
        flat_value = fold_kv_heads(value)

        self.assertEqual(flat_key.shape, (4, 6))
        self.assertEqual(flat_value.shape, (4, 10))
        torch.testing.assert_close(
            unfold_kv_heads(flat_key, kv_heads=2, head_dim=3), key
        )
        torch.testing.assert_close(
            unfold_kv_heads(flat_value, kv_heads=2, head_dim=5), value
        )

    def test_nhd_paged_view_is_zero_copy(self):
        nhd = torch.arange(8 * 2 * 3).reshape(8, 2, 3)
        pages = view_nhd_as_joint_pages(nhd, page_size=4)
        self.assertEqual(pages.shape, (2, 4, 6))
        self.assertEqual(
            pages.untyped_storage().data_ptr(), nhd.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(pages.reshape(8, 2, 3), nhd)

    def test_page_ids_preserve_logical_page_order(self):
        device_indices = torch.tensor(
            list(range(12, 16)) + list(range(4, 8)), dtype=torch.int64
        )
        page_ids = page_ids_from_device_indices(device_indices, page_size=4)
        torch.testing.assert_close(page_ids, torch.tensor([3, 1]))

    def test_page_ids_reject_misaligned_or_scrambled_slots(self):
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            page_ids_from_device_indices(torch.arange(2, 10), page_size=4)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            page_ids_from_device_indices(
                torch.tensor([0, 1, 3, 2], dtype=torch.int32), page_size=4
            )

    def test_int8_pack_contract(self):
        signed = torch.tensor([[-127, 0, 127]], dtype=torch.int8)
        packed = pack_signed_rank_values(signed, bits=8)
        torch.testing.assert_close(
            packed, torch.tensor([[1, 128, 255]], dtype=torch.uint8)
        )
        torch.testing.assert_close(
            unpack_signed_rank_values(packed, bits=8, rank=3), signed
        )

    def test_int4_pack_contract_and_odd_rank_padding(self):
        signed = torch.tensor([[-7, -1, 0, 1, 7]], dtype=torch.int8)
        packed = pack_signed_rank_values(signed, bits=4)
        # Low nibble is the earlier rank coordinate.  The final high nibble is
        # logical-zero padding (code 8).
        torch.testing.assert_close(
            packed, torch.tensor([[0x71, 0x98, 0x8F]], dtype=torch.uint8)
        )
        torch.testing.assert_close(
            unpack_signed_rank_values(packed, bits=4, rank=5), signed
        )

    def test_int2_pack_contract_and_odd_rank_padding(self):
        signed = torch.tensor([[-1, 0, 1, -1, 1]], dtype=torch.int8)
        packed = pack_signed_rank_values(signed, bits=2)
        torch.testing.assert_close(
            packed, torch.tensor([[0x79, 0xAB]], dtype=torch.uint8)
        )
        torch.testing.assert_close(
            unpack_signed_rank_values(packed, bits=2, rank=5), signed
        )

    def test_matrix_quantization_zero_contract(self):
        factor = quantize_factor(torch.zeros(3, 5), bits=4, scale_mode="matrix")
        self.assertEqual(factor.qdata.shape, (3, 3))
        self.assertEqual(factor.scale.shape, (1, 1))
        self.assertEqual(factor.scale.dtype, torch.float16)
        self.assertEqual(factor.storage_nbytes, 11)
        torch.testing.assert_close(factor.dequantize(), torch.zeros(3, 5))

    def test_row_group_scale_shape(self):
        factor = quantize_factor(
            torch.arange(10, dtype=torch.float32).reshape(2, 5),
            bits=4,
            scale_mode="row",
            group_size=2,
        )
        self.assertEqual(factor.qdata.shape, (2, 3))
        self.assertEqual(factor.scale.shape, (2, 3))
        self.assertEqual(factor.dequantize().shape, (2, 5))
        self.assertTrue(torch.isfinite(factor.dequantize()).all())

    def test_dequantization_honors_requested_compute_dtype(self):
        factor = quantize_factor(
            torch.arange(12, dtype=torch.float32).reshape(3, 4), bits=4
        )
        self.assertEqual(factor.dequantize(dtype=torch.float16).dtype, torch.float16)
        self.assertEqual(factor.dequantize(dtype=torch.bfloat16).dtype, torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "floating point"):
            factor.dequantize(dtype=torch.int32)

    def test_paged_svd_batch_shapes_are_normalized(self):
        torch.manual_seed(0)
        u = torch.randn(7, 3)
        sigma = torch.rand(1, 3)
        right = torch.randn(1, 5, 3)

        factors = quantize_svd_factors(
            u,
            sigma,
            right,
            bits_u=4,
            bits_r=2,
            scale_mode="matrix",
        )

        self.assertEqual(factors.u.qdata.shape, (7, 2))
        self.assertEqual(factors.sigma.shape, (3,))
        self.assertEqual(factors.right.qdata.shape, (5, 1))
        self.assertEqual(factors.reconstruct().shape, (7, 5))
        expected = (
            factors.u.dequantize() * factors.sigma.float()
        ) @ factors.right.dequantize().T
        torch.testing.assert_close(factors.reconstruct(), expected)

    def test_actual_storage_matches_estimator(self):
        torch.manual_seed(1)
        u = torch.randn(17, 5)
        sigma = torch.rand(5)
        right = torch.randn(9, 5)
        factors = quantize_svd_factors(
            u,
            sigma,
            right,
            bits_u=4,
            scale_mode="matrix",
        )
        estimated = estimate_quantized_svd_nbytes(
            chunk_tokens=17,
            feature_dim=9,
            rank=5,
            bits_u=4,
            scale_mode="matrix",
        )
        self.assertEqual(factors.storage_nbytes, estimated)

    def test_row_slice_reconstruction_and_device_copy(self):
        torch.manual_seed(3)
        factors = quantize_svd_factors(
            torch.randn(9, 4),
            torch.rand(4),
            torch.randn(7, 4),
            bits_u=4,
            bits_r=2,
            scale_mode="row",
            group_size_u=3,
            group_size_r=2,
        )

        sliced = factors.reconstruct_rows(2, 8)
        torch.testing.assert_close(sliced, factors.reconstruct()[2:8])
        copied = factors.to("cpu")
        self.assertIsNot(copied, factors)
        torch.testing.assert_close(copied.reconstruct_rows(2, 8), sliced)

        with self.assertRaisesRegex(ValueError, "row range"):
            factors.reconstruct_rows(-1, 2)

    def test_worked_capacity_numbers(self):
        mib = 1024**2
        common = dict(
            layers=32,
            chunk_tokens=4096,
            key_features=1024,
            value_features=1024,
            key_rank=32,
            value_rank=32,
            bits_u=4,
        )
        raw = raw_kv_chunk_nbytes(
            layers=32,
            chunk_tokens=4096,
            key_features=1024,
            value_features=1024,
            element_size=2,
        )
        matrix_scaled = estimate_quantized_kv_chunk_nbytes(
            **common, scale_mode="matrix"
        )
        row_scaled = estimate_quantized_kv_chunk_nbytes(
            **common,
            scale_mode="row",
            group_size_u=32,
            group_size_r=32,
        )

        self.assertEqual(raw, 512 * mib)
        self.assertEqual(matrix_scaled, 5 * mib + 4_352)
        self.assertEqual(row_scaled, 5 * mib + 655_360 + 4_096)

    def test_invalid_contracts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            fold_kv_heads(torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "group_size"):
            quantize_factor(
                torch.ones(2, 3),
                bits=4,
                scale_mode="matrix",
                group_size=3,
            )
        with self.assertRaisesRegex(ValueError, "INT4"):
            pack_signed_rank_values(torch.tensor([[-8, 0]], dtype=torch.int8), bits=4)


if __name__ == "__main__":
    unittest.main()
