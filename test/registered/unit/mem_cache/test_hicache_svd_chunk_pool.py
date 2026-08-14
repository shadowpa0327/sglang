"""CPU tests for the byte-budgeted HiCache SVD chunk pool."""

import threading
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_svd_chunk import (
    QuantizedKVChunk,
    make_svd_chunk_key,
    parse_svd_chunk_blob,
    serialize_svd_chunk,
)
from sglang.srt.mem_cache.hicache_svd_codec import quantize_svd_factors
from sglang.srt.mem_cache.storage.svd_chunk.svd_chunk_pool import SVDChunkPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _blob(size: int, value: int) -> torch.Tensor:
    return torch.full((size,), value, dtype=torch.uint8)


def _canonical_blob():
    metadata = {
        "model": "pool-test",
        "local_layers": [0],
        "page_size": 2,
        "chunk_tokens": 4,
    }

    def factors(seed: int):
        generator = torch.Generator().manual_seed(seed)
        return quantize_svd_factors(
            torch.randn(4, 2, generator=generator),
            torch.rand(2, generator=generator),
            torch.randn(4, 2, generator=generator),
            bits_u=4,
            bits_r=4,
        )

    chunk = QuantizedKVChunk.from_sequences([factors(1)], [factors(2)])
    kwargs = dict(
        codec_metadata=metadata,
        namespace=None,
        end_page_hash="pool-page",
    )
    return make_svd_chunk_key(**kwargs), metadata, serialize_svd_chunk(chunk, **kwargs)


class TestSVDChunkPool(unittest.TestCase):
    def test_put_parsed_can_carry_a_completed_contract_validation(self):
        key, metadata, blob = _canonical_blob()
        parsed = parse_svd_chunk_blob(
            blob,
            expected_key=key,
            expected_codec_metadata=metadata,
        )
        pool = SVDChunkPool(capacity_bytes=blob.numel())
        self.assertTrue(pool.put_parsed(key, parsed, contract_validated=True))
        validator = mock.MagicMock()

        with pool.acquire_parsed(
            [key], expected_codec_metadata=metadata, validator=validator
        ) as lease:
            self.assertEqual(len(lease), 1)
        validator.assert_not_called()

    def test_parsed_acquire_validates_once_and_reuses_zero_copy_views(self):
        key, metadata, blob = _canonical_blob()
        pool = SVDChunkPool(capacity_bytes=blob.numel())
        self.assertTrue(pool.put(key, blob))
        validated = []

        def validator(chunk):
            validated.append(chunk)

        with pool.acquire_parsed(
            [key], expected_codec_metadata=metadata, validator=validator
        ) as first_lease:
            self.assertEqual(len(first_lease.parsed_chunks), 1)
            first = first_lease.parsed_chunks[0]
            self.assertEqual(first.blob.data_ptr(), first_lease.blobs[0].data_ptr())
            storage_ptr = first.blob.untyped_storage().data_ptr()
            self.assertTrue(
                all(
                    view.bind(first.blob).untyped_storage().data_ptr() == storage_ptr
                    for view in first.tensor_views
                )
            )

        with pool.acquire_parsed(
            [key], expected_codec_metadata=metadata, validator=validator
        ) as second_lease:
            self.assertIs(second_lease.parsed_chunks[0], first)
        self.assertEqual(validated, [first.chunk])

        with self.assertRaisesRegex(ValueError, "codec metadata"):
            pool.acquire_parsed([key], expected_codec_metadata={"wrong": True})
        self.assertEqual(pool.stats().pin_count, 0)

    def test_failed_parsed_acquire_releases_lease(self):
        pool = SVDChunkPool(capacity_bytes=4)
        pool.put("not-canonical", _blob(4, 1))
        with self.assertRaises(ValueError):
            pool.acquire_parsed(["not-canonical"])
        self.assertEqual(pool.stats().pin_count, 0)
        self.assertEqual(pool.clear(), 1)

    def test_optional_pinned_blob_ownership(self):
        with self.assertRaisesRegex(TypeError, "pin_memory"):
            SVDChunkPool(capacity_bytes=1, pin_memory=1)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                SVDChunkPool(capacity_bytes=1, pin_memory=True)
            return

        pool = SVDChunkPool(capacity_bytes=4, pin_memory=True)
        self.assertTrue(pool.put("pinned", _blob(4, 1)))
        self.assertTrue(pool.get("pinned").is_pinned())

    def test_owns_input_and_accounts_exact_serialized_bytes(self):
        pool = SVDChunkPool(capacity_bytes=7)
        source = _blob(7, 3)
        self.assertTrue(pool.put("a", source))
        source.fill_(9)

        stored = pool.get("a")
        self.assertIsNotNone(stored)
        torch.testing.assert_close(stored, _blob(7, 3))
        self.assertTrue(stored.is_contiguous())
        self.assertEqual(pool.used_bytes, 7)
        self.assertEqual(pool.stats().available_bytes, 0)

    def test_lru_get_promotion_and_capacity_eviction(self):
        pool = SVDChunkPool(capacity_bytes=6)
        for key, value in (("a", 1), ("b", 2), ("c", 3)):
            self.assertTrue(pool.put(key, _blob(2, value)))

        self.assertIsNotNone(pool.get("a"))  # b, c, a from LRU to MRU
        self.assertTrue(pool.put("d", _blob(2, 4)))

        self.assertIsNone(pool.get("b"))
        for key in ("a", "c", "d"):
            self.assertIsNotNone(pool.get(key))
        stats = pool.stats()
        self.assertEqual(stats.used_bytes, 6)
        self.assertEqual(stats.entry_count, 3)
        self.assertEqual(stats.evictions, 1)

    def test_exists_does_not_promote_or_change_lookup_counters(self):
        pool = SVDChunkPool(capacity_bytes=4)
        pool.put("a", _blob(2, 1))
        pool.put("b", _blob(2, 2))
        self.assertTrue(pool.exists("a"))
        self.assertFalse(pool.exists("missing"))
        self.assertEqual(pool.stats().hits, 0)
        self.assertEqual(pool.stats().misses, 0)

        # If exists had promoted a, b would be evicted instead.
        pool.put("c", _blob(2, 3))
        self.assertFalse(pool.exists("a"))
        self.assertTrue(pool.exists("b"))
        self.assertTrue(pool.exists("c"))

    def test_rejected_put_does_not_partially_evict_around_pins(self):
        pool = SVDChunkPool(capacity_bytes=6)
        pool.put("a", _blob(4, 1))
        pool.put("b", _blob(2, 2))

        lease = pool.acquire(["a"])
        self.assertEqual(lease.keys, ("a",))
        # Only b's two bytes are evictable, which cannot admit four bytes.
        self.assertFalse(pool.put("c", _blob(4, 3)))
        self.assertIsNotNone(pool.get("b"), "failed admission must be transactional")
        stats = pool.stats()
        self.assertEqual(stats.entry_count, 2)
        self.assertEqual(stats.used_bytes, 6)
        self.assertEqual(stats.evictions, 0)
        self.assertEqual(stats.rejected_puts, 1)

        lease.release()
        lease.release()  # idempotent
        self.assertTrue(pool.put("c", _blob(4, 3)))

    def test_acquire_returns_and_pins_longest_contiguous_prefix(self):
        pool = SVDChunkPool(capacity_bytes=16)
        for key, value in (("a", 1), ("b", 2), ("d", 4)):
            pool.put(key, _blob(4, value))

        with pool.acquire(["a", "b", "c", "d"]) as lease:
            self.assertEqual(lease.keys, ("a", "b"))
            self.assertEqual(len(lease), 2)
            self.assertEqual([int(blob[0]) for blob in lease], [1, 2])
            stats = pool.stats()
            self.assertEqual(stats.pinned_entry_count, 2)
            self.assertEqual(stats.pin_count, 2)

            # clear cannot remove the acquired prefix, but removes d.
            self.assertEqual(pool.clear(), 1)
            self.assertEqual(pool.stats().entry_count, 2)

        self.assertEqual(pool.stats().pin_count, 0)
        self.assertEqual(pool.clear(), 2)
        self.assertEqual(len(pool), 0)

    def test_duplicate_keys_have_balanced_pin_counts(self):
        pool = SVDChunkPool(capacity_bytes=4)
        pool.put("a", _blob(4, 1))
        lease = pool.acquire(["a", "a"])
        self.assertEqual(lease.keys, ("a", "a"))
        self.assertEqual(pool.stats().pinned_entry_count, 1)
        self.assertEqual(pool.stats().pin_count, 2)
        lease.release()
        self.assertEqual(pool.stats().pin_count, 0)

    def test_miss_at_first_key_returns_empty_lease(self):
        pool = SVDChunkPool(capacity_bytes=4)
        pool.put("later", _blob(4, 1))
        with pool.acquire(["missing", "later"]) as lease:
            self.assertFalse(lease)
            self.assertEqual(lease.items, ())
        stats = pool.stats()
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.pin_count, 0)

    def test_immutable_key_binding_and_identical_put_promotion(self):
        pool = SVDChunkPool(capacity_bytes=4)
        self.assertTrue(pool.put("a", _blob(4, 1)))
        self.assertTrue(pool.put("a", _blob(4, 1)))
        self.assertEqual(pool.used_bytes, 4)
        self.assertEqual(len(pool), 1)
        with self.assertRaisesRegex(ValueError, "different immutable bytes"):
            pool.put("a", _blob(4, 2))

    def test_oversized_and_invalid_blobs_are_rejected(self):
        pool = SVDChunkPool(capacity_bytes=3)
        self.assertFalse(pool.put("large", _blob(4, 1)))
        self.assertEqual(pool.stats().rejected_puts, 1)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            pool.put("2d", torch.zeros((1, 2), dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "dtype"):
            pool.put("float", torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            pool.put("", _blob(1, 1))

    def test_cross_thread_lease_blocks_then_allows_eviction(self):
        pool = SVDChunkPool(capacity_bytes=4)
        pool.put("a", _blob(4, 1))
        acquired = threading.Event()
        release = threading.Event()

        def hold_lease():
            with pool.acquire(["a"]):
                acquired.set()
                self.assertTrue(release.wait(timeout=2))

        worker = threading.Thread(target=hold_lease)
        worker.start()
        self.assertTrue(acquired.wait(timeout=2))
        self.assertFalse(pool.put("b", _blob(4, 2)))
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

        self.assertTrue(pool.put("b", _blob(4, 2)))
        self.assertIsNone(pool.get("a"))
        self.assertIsNotNone(pool.get("b"))


if __name__ == "__main__":
    unittest.main()
