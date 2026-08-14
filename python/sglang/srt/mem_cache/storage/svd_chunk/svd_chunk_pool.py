# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""A byte-budgeted in-memory pool for serialized HiCache SVD chunks.

The pool stores content-addressed, one-dimensional CPU ``uint8`` tensors.  It
owns blobs on admission, optionally in pinned memory, never mutates admitted
storage, and refuses to bind different bytes to an existing key.  Returned
tensors are read-only by contract (PyTorch has no read-only tensor view).

Entries are ordered from least to most recently used.  ``get`` promotes one
entry without pinning it.  ``acquire`` atomically promotes and pins the longest
contiguous prefix of a requested key sequence; its lease must be released when
the caller has finished asynchronous I/O or reconstruction.  Pinned entries
are never capacity-evicted or removed by ``clear``.
"""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_svd_chunk import (
    ParsedSVDChunkBlob,
    parse_svd_chunk_blob,
)


@dataclass(frozen=True)
class SVDChunkPoolStats:
    """An atomic snapshot of pool occupancy and lifetime counters."""

    capacity_bytes: int
    used_bytes: int
    available_bytes: int
    entry_count: int
    pinned_entry_count: int
    pin_count: int
    hits: int
    misses: int
    evictions: int
    rejected_puts: int


@dataclass
class _Entry:
    blob: torch.Tensor
    nbytes: int
    pin_count: int = 0
    parsed: Optional[ParsedSVDChunkBlob] = None
    parsed_contract_validated: bool = False
    parse_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class SVDChunkLease:
    """Pins a contiguous sequence of chunks until released.

    Leases are context managers and ``release`` is idempotent.  ``items``,
    ``keys``, and ``blobs`` remain available after release, but callers must not
    retain or use those read-only tensor views beyond the lease lifetime if the
    pool's byte budget is intended to describe all live chunk storage.
    """

    def __init__(
        self,
        pool: SVDChunkPool,
        items: Tuple[Tuple[str, torch.Tensor], ...],
        parsed_chunks: Tuple[ParsedSVDChunkBlob, ...] = (),
    ) -> None:
        if parsed_chunks and len(parsed_chunks) != len(items):
            raise ValueError("parsed_chunks must correspond one-to-one with items")
        self._pool: Optional[SVDChunkPool] = pool
        self._items = items
        self._parsed_chunks = parsed_chunks
        self._release_lock = threading.Lock()

    @property
    def items(self) -> Tuple[Tuple[str, torch.Tensor], ...]:
        return self._items

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(key for key, _ in self._items)

    @property
    def blobs(self) -> Tuple[torch.Tensor, ...]:
        return tuple(blob for _, blob in self._items)

    @property
    def parsed_chunks(self) -> Tuple[ParsedSVDChunkBlob, ...]:
        """Validated zero-copy chunk views, populated by ``acquire_parsed``."""

        return self._parsed_chunks

    @property
    def released(self) -> bool:
        return self._pool is None

    def release(self) -> None:
        """Unpin every acquired occurrence exactly once."""

        with self._release_lock:
            pool = self._pool
            if pool is None:
                return
            self._pool = None
        pool._release(Counter(self.keys))

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.blobs)

    def __enter__(self) -> SVDChunkLease:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            # Destructors may run while Python modules and locks are shutting
            # down. Explicit release / context-manager use remains preferred.
            pass


class SVDChunkPool:
    """Thread-safe LRU storage bounded by serialized bytes.

    Args:
        capacity_bytes: Maximum sum of admitted blob sizes.  A zero capacity
            accepts only zero-length blobs.
        pin_memory: Own admitted blobs in CUDA-pinned CPU memory.  This enables
            a later nonblocking whole-blob H2D copy and requires available CUDA.

    ``put`` is transactional with respect to eviction: if unpinned victims
    cannot free enough space, it rejects the new blob without removing any
    existing entry.  Re-putting identical bytes promotes the existing entry;
    re-putting different bytes for the same key raises ``ValueError``.
    """

    def __init__(self, capacity_bytes: int, *, pin_memory: bool = False) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            raise TypeError("capacity_bytes must be an integer")
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a bool")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True requires an available CUDA runtime")

        self._capacity_bytes = capacity_bytes
        self._pin_memory = pin_memory
        self._used_bytes = 0
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejected_puts = 0

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def pin_memory(self) -> bool:
        return self._pin_memory

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def put(self, key: str, blob: torch.Tensor) -> bool:
        """Admit an owned copy of ``blob``, evicting unpinned LRU entries.

        Returns ``False`` when the blob exceeds the capacity or pinned entries
        leave insufficient evictable space.  Such a rejection never causes
        partial eviction.  Shape, device, and dtype violations raise.
        """

        self._validate_key(key)
        self._validate_blob(blob)
        owned = self._own_blob(blob)
        nbytes = owned.numel() * owned.element_size()

        with self._lock:
            current = self._entries.get(key)
            if current is not None:
                if current.nbytes != nbytes or not torch.equal(current.blob, owned):
                    raise ValueError(
                        f"key {key!r} is already bound to different immutable bytes"
                    )
                self._entries.move_to_end(key)
                return True

            if nbytes > self._capacity_bytes:
                self._rejected_puts += 1
                return False

            bytes_needed = self._used_bytes + nbytes - self._capacity_bytes
            victims = []
            freed_bytes = 0
            if bytes_needed > 0:
                for victim_key, entry in self._entries.items():
                    if entry.pin_count != 0:
                        continue
                    victims.append(victim_key)
                    freed_bytes += entry.nbytes
                    if freed_bytes >= bytes_needed:
                        break
                if freed_bytes < bytes_needed:
                    self._rejected_puts += 1
                    return False

            for victim_key in victims:
                victim = self._entries.pop(victim_key)
                self._used_bytes -= victim.nbytes
                self._evictions += 1

            self._entries[key] = _Entry(blob=owned, nbytes=nbytes)
            self._used_bytes += nbytes
            return True

    def put_parsed(
        self,
        key: str,
        parsed: ParsedSVDChunkBlob,
        *,
        contract_validated: bool = False,
    ) -> bool:
        """Admit a validated chunk and preserve its parse/layout cache.

        The pool still takes one owned (optionally pinned) whole-blob copy, but
        all factor tensors are remapped over that allocation without another
        checksum, JSON parse, or tensor payload copy.
        """

        if not isinstance(parsed, ParsedSVDChunkBlob):
            raise TypeError("parsed must be a ParsedSVDChunkBlob")
        if not isinstance(contract_validated, bool):
            raise TypeError("contract_validated must be a bool")
        parsed.validate_expectations(expected_key=key)
        admitted = self.put(key, parsed.blob)
        if not admitted:
            return False
        with self._lock:
            entry = self._entries[key]
        with entry.parse_lock:
            if entry.parsed is None:
                mapped = parsed.map_blob(entry.blob)
                entry.parsed = ParsedSVDChunkBlob(
                    blob=entry.blob,
                    chunk=mapped.chunk,
                    metadata=parsed.metadata,
                    tensor_views=parsed.tensor_views,
                )
            if contract_validated:
                entry.parsed_contract_validated = True
        return True

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return the read-only blob view for ``key`` and promote it to MRU."""

        self._validate_key(key)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return entry.blob

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is resident without changing recency/stats."""

        self._validate_key(key)
        with self._lock:
            return key in self._entries

    def acquire(self, keys: Iterable[str]) -> SVDChunkLease:
        """Pin and return the longest present prefix of ``keys`` atomically.

        Lookup stops at the first absent key.  The found prefix is promoted to
        MRU in request order and remains capacity-resident until the returned
        lease is released.  An empty request, or a miss at its first key,
        returns a valid empty lease.
        """

        requested = tuple(keys)
        for key in requested:
            self._validate_key(key)

        with self._lock:
            items = []
            for key in requested:
                entry = self._entries.get(key)
                if entry is None:
                    self._misses += 1
                    break
                self._hits += 1
                entry.pin_count += 1
                self._entries.move_to_end(key)
                items.append((key, entry.blob))
            return SVDChunkLease(self, tuple(items))

    def acquire_parsed(
        self,
        keys: Iterable[str],
        *,
        expected_codec_metadata: Optional[Mapping[str, Any]] = None,
        validator: Optional[Callable[[Any], None]] = None,
    ) -> SVDChunkLease:
        """Pin and return the longest prefix with one cached validation per entry.

        The first acquisition parses and checksums an entry, retaining factor
        tensors as views over the pool-owned blob.  Later acquisitions reuse the
        same :class:`ParsedSVDChunkBlob` and only recheck the caller's key/codec
        expectations.  A parse failure releases every pin before propagating.
        """

        requested = tuple(keys)
        for key in requested:
            self._validate_key(key)
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable or None")

        with self._lock:
            acquired = []
            for key in requested:
                entry = self._entries.get(key)
                if entry is None:
                    self._misses += 1
                    break
                self._hits += 1
                entry.pin_count += 1
                self._entries.move_to_end(key)
                acquired.append((key, entry))

        item_keys = tuple(key for key, _ in acquired)
        try:
            parsed_chunks = []
            for key, entry in acquired:
                with entry.parse_lock:
                    if entry.parsed is None:
                        entry.parsed = parse_svd_chunk_blob(
                            entry.blob,
                            expected_key=key,
                            expected_codec_metadata=expected_codec_metadata,
                            copy_blob=False,
                        )
                    else:
                        entry.parsed.validate_expectations(
                            expected_key=key,
                            expected_codec_metadata=expected_codec_metadata,
                        )
                    if validator is not None and not entry.parsed_contract_validated:
                        validator(entry.parsed.chunk)
                        entry.parsed_contract_validated = True
                    parsed_chunks.append(entry.parsed)
        except Exception:
            self._release(Counter(item_keys))
            raise

        items = tuple((key, entry.blob) for key, entry in acquired)
        return SVDChunkLease(self, items, tuple(parsed_chunks))

    def clear(self) -> int:
        """Remove every unpinned entry and return the number removed.

        Pinned entries remain in the pool until their leases are released; a
        subsequent ``clear`` can then remove them.  Explicit clears are not
        counted as capacity evictions in ``stats().evictions``.
        """

        with self._lock:
            removable = [
                key for key, entry in self._entries.items() if entry.pin_count == 0
            ]
            for key in removable:
                entry = self._entries.pop(key)
                self._used_bytes -= entry.nbytes
            return len(removable)

    def stats(self) -> SVDChunkPoolStats:
        """Return a consistent occupancy/counter snapshot."""

        with self._lock:
            pinned_entries = 0
            pins = 0
            for entry in self._entries.values():
                if entry.pin_count:
                    pinned_entries += 1
                    pins += entry.pin_count
            return SVDChunkPoolStats(
                capacity_bytes=self._capacity_bytes,
                used_bytes=self._used_bytes,
                available_bytes=self._capacity_bytes - self._used_bytes,
                entry_count=len(self._entries),
                pinned_entry_count=pinned_entries,
                pin_count=pins,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                rejected_puts=self._rejected_puts,
            )

    def _release(self, key_counts: Counter[str]) -> None:
        with self._lock:
            for key, count in key_counts.items():
                entry = self._entries.get(key)
                if entry is None or entry.pin_count < count:
                    raise RuntimeError(f"invalid lease release for key {key!r}")
                entry.pin_count -= count

    def _own_blob(self, blob: torch.Tensor) -> torch.Tensor:
        source = blob.detach().contiguous()
        if not self._pin_memory:
            return source.clone(memory_format=torch.contiguous_format)
        owned = torch.empty(
            source.shape,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        owned.copy_(source)
        return owned

    @staticmethod
    def _validate_key(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError("chunk key must be a string")
        if not key:
            raise ValueError("chunk key must not be empty")

    @staticmethod
    def _validate_blob(blob: torch.Tensor) -> None:
        if not isinstance(blob, torch.Tensor):
            raise TypeError("chunk blob must be a torch.Tensor")
        if blob.device.type != "cpu":
            raise ValueError("chunk blob must reside on CPU")
        if blob.dtype != torch.uint8:
            raise ValueError("chunk blob must have dtype torch.uint8")
        if blob.ndim != 1:
            raise ValueError("chunk blob must be one-dimensional")
