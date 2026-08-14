"""CPU tests for canonical HiCache SVD chunk objects.

python test/registered/unit/mem_cache/test_hicache_svd_chunk.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.hicache_svd_chunk import (
    QuantizedKVChunk,
    deserialize_svd_chunk,
    make_svd_chunk_key,
    parse_svd_chunk_blob,
    serialize_svd_chunk,
)
from sglang.srt.mem_cache.hicache_svd_codec import quantize_svd_factors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_factors(*, tokens: int, features: int, rank: int, offset: int):
    generator = torch.Generator().manual_seed(100 + offset)
    return quantize_svd_factors(
        torch.randn(tokens, rank, generator=generator),
        torch.rand(rank, generator=generator),
        torch.randn(features, rank, generator=generator),
        bits_u=4,
        bits_r=2,
        scale_mode="row",
        group_size_u=2,
        group_size_r=3,
    )


def _make_chunk() -> QuantizedKVChunk:
    return QuantizedKVChunk.from_sequences(
        [
            _make_factors(tokens=8, features=6, rank=3, offset=layer)
            for layer in range(2)
        ],
        [
            _make_factors(tokens=8, features=10, rank=2, offset=10 + layer)
            for layer in range(2)
        ],
    )


def _codec_metadata():
    return {
        "model": "tiny-test-model",
        "local_layers": [0, 1],
        "page_size": 4,
        "chunk_tokens": 8,
        "tp_rank": 0,
        "tp_size": 1,
        "raw_dtype": "float16",
    }


def _factor_tensors(factors):
    return (
        factors.u.qdata,
        factors.u.scale,
        factors.sigma,
        factors.right.qdata,
        factors.right.scale,
    )


def _chunk_tensors(chunk):
    for key, value in zip(chunk.key_layers, chunk.value_layers):
        for factors in (key, value):
            yield from _factor_tensors(factors)


class TestHiCacheSVDChunk(unittest.TestCase):
    def test_parsed_blob_owns_one_buffer_and_exposes_zero_copy_views(self):
        kwargs = dict(
            codec_metadata=_codec_metadata(),
            namespace={"adapter": "base"},
            end_page_hash="page-view",
        )
        source = serialize_svd_chunk(_make_chunk(), **kwargs)
        source_snapshot = source.clone()
        parsed = parse_svd_chunk_blob(
            source,
            expected_key=make_svd_chunk_key(**kwargs),
            expected_codec_metadata=_codec_metadata(),
        )

        self.assertNotEqual(parsed.blob.data_ptr(), source.data_ptr())
        self.assertEqual(len(parsed.tensor_views), 2 * 2 * 5)
        storage_ptr = parsed.blob.untyped_storage().data_ptr()
        for tensor, view in zip(_chunk_tensors(parsed.chunk), parsed.tensor_views):
            self.assertEqual(tensor.untyped_storage().data_ptr(), storage_ptr)
            self.assertEqual(tuple(tensor.shape), view.shape)
            self.assertEqual(tensor.dtype, view.dtype)
            self.assertEqual(tensor.data_ptr(), view.bind(parsed.blob).data_ptr())
            self.assertEqual(view.byte_offset % 64, 0)

        source.zero_()
        torch.testing.assert_close(parsed.blob, source_snapshot)

    def test_parsed_layout_remaps_all_factors_to_one_new_blob(self):
        kwargs = dict(
            codec_metadata=_codec_metadata(),
            namespace=None,
            end_page_hash="page-remap",
        )
        source = serialize_svd_chunk(_make_chunk(), **kwargs)
        parsed = parse_svd_chunk_blob(source, copy_blob=False)
        self.assertEqual(parsed.blob.data_ptr(), source.data_ptr())

        target = parsed.blob.clone()
        mapped = parsed.map_blob(target)
        target_storage = target.untyped_storage().data_ptr()
        for expected, actual in zip(
            _chunk_tensors(parsed.chunk), _chunk_tensors(mapped.chunk)
        ):
            self.assertEqual(actual.untyped_storage().data_ptr(), target_storage)
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "exact size"):
            parsed.map_blob(target[:-1])

    def test_legacy_deserialize_keeps_independent_tensor_ownership(self):
        blob = serialize_svd_chunk(
            _make_chunk(),
            codec_metadata=_codec_metadata(),
            namespace=None,
            end_page_hash="legacy-owned",
        )
        restored, _ = deserialize_svd_chunk(blob)
        restored_snapshot = tuple(tensor.clone() for tensor in _chunk_tensors(restored))
        blob.zero_()
        for expected, actual in zip(restored_snapshot, _chunk_tensors(restored)):
            torch.testing.assert_close(actual, expected)

    def test_deterministic_round_trip(self):
        chunk = _make_chunk()
        kwargs = dict(
            codec_metadata=_codec_metadata(),
            namespace={"adapter": "base"},
            end_page_hash="0123456789abcdef",
        )

        first = serialize_svd_chunk(chunk, **kwargs)
        second = serialize_svd_chunk(chunk, **kwargs)
        torch.testing.assert_close(first, second)

        expected_key = make_svd_chunk_key(**kwargs)
        restored, metadata = deserialize_svd_chunk(
            first,
            expected_key=expected_key,
            expected_codec_metadata=_codec_metadata(),
        )
        self.assertEqual(metadata["chunk_key"], expected_key)
        self.assertEqual(restored.layer_count, 2)
        self.assertEqual(restored.chunk_tokens, 8)
        self.assertGreater(first.numel(), chunk.storage_nbytes)

        for expected_layers, actual_layers in (
            (chunk.key_layers, restored.key_layers),
            (chunk.value_layers, restored.value_layers),
        ):
            for expected, actual in zip(expected_layers, actual_layers):
                for expected_tensor, actual_tensor in zip(
                    _factor_tensors(expected), _factor_tensors(actual)
                ):
                    torch.testing.assert_close(expected_tensor, actual_tensor)
                torch.testing.assert_close(expected.reconstruct(), actual.reconstruct())

    def test_key_namespaces_codec_and_prefix_end(self):
        common = dict(
            codec_metadata=_codec_metadata(),
            namespace={"adapter": "base"},
            end_page_hash="page-63",
        )
        baseline = make_svd_chunk_key(**common)
        self.assertTrue(baseline.startswith("svdq1-"))
        self.assertEqual(len(baseline), len("svdq1-") + 64)
        self.assertEqual(baseline, make_svd_chunk_key(**common))
        self.assertNotEqual(
            baseline,
            make_svd_chunk_key(**{**common, "end_page_hash": "page-127"}),
        )
        self.assertNotEqual(
            baseline,
            make_svd_chunk_key(**{**common, "namespace": {"adapter": "lora"}}),
        )
        changed_codec = {**_codec_metadata(), "tp_size": 2}
        self.assertNotEqual(
            baseline,
            make_svd_chunk_key(**{**common, "codec_metadata": changed_codec}),
        )

    def test_corruption_truncation_and_mismatch_are_rejected(self):
        kwargs = dict(
            codec_metadata=_codec_metadata(),
            namespace=None,
            end_page_hash="page-1",
        )
        blob = serialize_svd_chunk(_make_chunk(), **kwargs)

        corrupted = blob.clone()
        corrupted[len(corrupted) // 2] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            deserialize_svd_chunk(corrupted)
        with self.assertRaisesRegex(ValueError, "total length"):
            deserialize_svd_chunk(blob[:-1].clone())
        with self.assertRaisesRegex(ValueError, "requested key"):
            deserialize_svd_chunk(blob, expected_key="svdq1-wrong")
        with self.assertRaisesRegex(ValueError, "codec metadata"):
            deserialize_svd_chunk(
                blob,
                expected_codec_metadata={**_codec_metadata(), "tp_size": 2},
            )

    def test_chunk_requires_consistent_layer_contracts(self):
        with self.assertRaisesRegex(ValueError, "share one factor contract"):
            QuantizedKVChunk.from_sequences(
                [
                    _make_factors(tokens=8, features=6, rank=3, offset=0),
                    _make_factors(tokens=8, features=7, rank=3, offset=1),
                ],
                [
                    _make_factors(tokens=8, features=10, rank=2, offset=10),
                    _make_factors(tokens=8, features=10, rank=2, offset=11),
                ],
            )


if __name__ == "__main__":
    unittest.main()
