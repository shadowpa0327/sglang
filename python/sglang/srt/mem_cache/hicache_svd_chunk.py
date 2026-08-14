"""Canonical chunk objects for factor-quantized HiCache SVD storage."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.mem_cache.hicache_svd_codec import (
    SUPPORTED_FACTOR_BITS,
    QuantizedFactor,
    QuantizedSVDFactors,
)

SVD_CHUNK_MAGIC = b"HCSVDQ1\0"
SVD_CHUNK_VERSION = 1
SVD_CHUNK_ALIGNMENT = 64
SVD_CHUNK_CHECKSUM_BYTES = 32
_PREAMBLE = struct.Struct("<8sIIIIQ")
_LITTLE_ENDIAN_FLAG = 1
_KEY_DOMAIN = b"sglang-hicache-svdq1\0"


def _align_up(value: int, alignment: int = SVD_CHUNK_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("codec metadata must be canonical JSON data") from exc
    return encoded.encode("ascii")


def codec_digest(codec_metadata: Mapping[str, Any]) -> str:
    """Return the stable digest namespacing one codec/model geometry."""

    return hashlib.sha256(_canonical_json(codec_metadata)).hexdigest()


def namespace_digest(namespace: Any) -> str:
    """Digest a JSON-compatible cache namespace such as ``RadixKey.extra_key``."""

    return hashlib.sha256(_canonical_json({"namespace": namespace})).hexdigest()


def make_svd_chunk_key(
    *,
    codec_metadata: Mapping[str, Any],
    namespace: Any,
    end_page_hash: str,
) -> str:
    """Build a short filename-safe key for a complete compressed chunk."""

    if not isinstance(end_page_hash, str) or not end_page_hash:
        raise ValueError("end_page_hash must be a nonempty string")
    return _make_svd_chunk_key_from_digests(
        codec_hash=codec_digest(codec_metadata),
        namespace_hash=namespace_digest(namespace),
        end_page_hash=end_page_hash,
    )


def _make_svd_chunk_key_from_digests(
    *, codec_hash: str, namespace_hash: str, end_page_hash: str
) -> str:
    try:
        codec_bytes = bytes.fromhex(codec_hash)
        namespace_bytes = bytes.fromhex(namespace_hash)
    except ValueError as exc:
        raise ValueError("codec and namespace digests must be hexadecimal") from exc
    if len(codec_bytes) != 32 or len(namespace_bytes) != 32:
        raise ValueError("codec and namespace digests must be SHA-256 values")
    digest = hashlib.sha256(
        _KEY_DOMAIN + codec_bytes + namespace_bytes + end_page_hash.encode("utf-8")
    ).hexdigest()
    return f"svdq1-{digest}"


def _factor_signature(factors: QuantizedSVDFactors) -> tuple[Any, ...]:
    return (
        factors.chunk_tokens,
        factors.feature_dim,
        factors.rank,
        factors.u.bits,
        factors.right.bits,
        factors.u.scale_mode,
        factors.right.scale_mode,
        factors.u.group_size,
        factors.right.group_size,
        factors.u.scale.dtype,
        factors.right.scale.dtype,
        factors.sigma.dtype,
    )


@dataclass(frozen=True)
class QuantizedKVChunk:
    """Every local layer's K and V factors for one immutable token chunk."""

    key_layers: tuple[QuantizedSVDFactors, ...]
    value_layers: tuple[QuantizedSVDFactors, ...]

    def __post_init__(self) -> None:
        if not self.key_layers or len(self.key_layers) != len(self.value_layers):
            raise ValueError(
                "key_layers and value_layers must have equal nonzero length"
            )
        for name, layers in (
            ("key", self.key_layers),
            ("value", self.value_layers),
        ):
            signature = _factor_signature(layers[0])
            for layer in layers[1:]:
                if _factor_signature(layer) != signature:
                    raise ValueError(
                        f"all {name} layers must share one factor contract"
                    )
        if self.key_layers[0].chunk_tokens != self.value_layers[0].chunk_tokens:
            raise ValueError("K and V factors must cover the same chunk tokens")

    @classmethod
    def from_sequences(
        cls,
        key_layers: Sequence[QuantizedSVDFactors],
        value_layers: Sequence[QuantizedSVDFactors],
    ) -> QuantizedKVChunk:
        return cls(tuple(key_layers), tuple(value_layers))

    @property
    def layer_count(self) -> int:
        return len(self.key_layers)

    @property
    def chunk_tokens(self) -> int:
        return self.key_layers[0].chunk_tokens

    @property
    def storage_nbytes(self) -> int:
        return sum(
            key.storage_nbytes + value.storage_nbytes
            for key, value in zip(self.key_layers, self.value_layers)
        )


@dataclass(frozen=True)
class SVDChunkTensorView:
    """Location and tensor contract for one payload tensor in a chunk blob.

    ``bind`` only creates a tensor view.  It never copies payload bytes, so a
    caller can copy the complete serialized blob to another device once and
    rebuild every packed factor over that single allocation.
    """

    name: str
    byte_offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math_prod(self.shape) * torch.empty((), dtype=self.dtype).element_size()

    def bind(self, blob: torch.Tensor) -> torch.Tensor:
        """Return this tensor as a zero-copy view of ``blob``."""

        _validate_blob_storage(blob)
        end = self.byte_offset + self.nbytes
        if self.byte_offset < 0 or end > blob.numel():
            raise ValueError(f"tensor view {self.name!r} is outside the chunk blob")
        byte_view = blob.narrow(0, self.byte_offset, self.nbytes)
        if self.dtype != torch.uint8:
            byte_view = byte_view.view(self.dtype)
        return byte_view.reshape(self.shape)


@dataclass(frozen=True)
class MappedSVDChunkBlob:
    """A chunk whose factor tensors view one serialized blob allocation."""

    blob: torch.Tensor
    chunk: QuantizedKVChunk
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ParsedSVDChunkBlob:
    """One validated chunk and its reusable zero-copy tensor layout.

    The blob is immutable by contract.  ``tensor_views`` records every factor's
    byte offset, shape, and dtype in serialization order.  ``map_blob`` rebuilds
    all factor tensors over another one-dimensional ``uint8`` allocation without
    parsing, checksumming, or copying it.  ``to`` performs exactly one blob-level
    ``Tensor.to`` and then applies that layout.
    """

    blob: torch.Tensor
    chunk: QuantizedKVChunk
    metadata: Mapping[str, Any]
    tensor_views: tuple[SVDChunkTensorView, ...]

    @property
    def key(self) -> str:
        return str(self.metadata["chunk_key"])

    def validate_expectations(
        self,
        *,
        expected_key: str | None = None,
        expected_codec_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Recheck caller expectations without revalidating the blob payload."""

        if expected_key is not None and expected_key != self.key:
            raise ValueError("SVD chunk does not match the requested key")
        if (
            expected_codec_metadata is not None
            and dict(expected_codec_metadata) != self.metadata["codec"]
        ):
            raise ValueError("SVD chunk codec metadata is incompatible")

    def map_blob(self, blob: torch.Tensor) -> MappedSVDChunkBlob:
        """Remap all factors onto a same-sized CPU or device blob, without copies."""

        _validate_blob_storage(blob)
        if blob.numel() != self.blob.numel():
            raise ValueError("mapped blob must have the serialized chunk's exact size")
        tensors = tuple(view.bind(blob) for view in self.tensor_views)
        return MappedSVDChunkBlob(
            blob=blob,
            chunk=_remap_chunk(self.chunk, tensors),
            metadata=self.metadata,
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MappedSVDChunkBlob:
        """Copy the complete blob once, then bind device factor views to it."""

        mapped_blob = self.blob.to(device=device, non_blocking=non_blocking)
        return self.map_blob(mapped_blob)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    raise TypeError(f"SVD chunk v1 requires FP16 scale/Sigma tensors, got {dtype}")


def _factor_spec(factors: QuantizedSVDFactors) -> dict[str, Any]:
    if factors.u.scale_mode != factors.right.scale_mode:
        raise ValueError("U and R must use the same scale mode in chunk v1")
    return {
        "feature_dim": factors.feature_dim,
        "rank": factors.rank,
        "bits_u": factors.u.bits,
        "bits_r": factors.right.bits,
        "scale_mode": factors.u.scale_mode,
        "group_size_u": factors.u.group_size,
        "group_size_r": factors.right.group_size,
        "scale_dtype": _dtype_name(factors.u.scale.dtype),
        "right_scale_dtype": _dtype_name(factors.right.scale.dtype),
        "sigma_dtype": _dtype_name(factors.sigma.dtype),
        "packing": "rank-low-bits-first-offset-binary",
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype not in (torch.uint8, torch.float16):
        raise TypeError(f"unsupported chunk tensor dtype: {tensor.dtype}")
    return tensor.numpy().tobytes(order="C")


def _append_aligned_tensor(buffer: bytearray, tensor: torch.Tensor) -> None:
    aligned = _align_up(len(buffer))
    if aligned > len(buffer):
        buffer.extend(b"\0" * (aligned - len(buffer)))
    buffer.extend(_tensor_bytes(tensor))


def _factor_tensors(factors: QuantizedSVDFactors) -> tuple[torch.Tensor, ...]:
    return (
        factors.u.qdata,
        factors.u.scale,
        factors.sigma,
        factors.right.qdata,
        factors.right.scale,
    )


def serialize_svd_chunk(
    chunk: QuantizedKVChunk,
    *,
    codec_metadata: Mapping[str, Any],
    namespace: Any,
    end_page_hash: str,
) -> torch.Tensor:
    """Serialize one chunk to a deterministic checksummed CPU ``uint8`` tensor."""

    codec_metadata = dict(codec_metadata)
    codec_hash = codec_digest(codec_metadata)
    namespace_hash = namespace_digest(namespace)
    chunk_key = _make_svd_chunk_key_from_digests(
        codec_hash=codec_hash,
        namespace_hash=namespace_hash,
        end_page_hash=end_page_hash,
    )
    metadata = {
        "version": SVD_CHUNK_VERSION,
        "chunk_tokens": chunk.chunk_tokens,
        "layers": chunk.layer_count,
        "key": _factor_spec(chunk.key_layers[0]),
        "value": _factor_spec(chunk.value_layers[0]),
        "codec": codec_metadata,
        "codec_digest": codec_hash,
        "namespace_digest": namespace_hash,
        "end_page_hash": end_page_hash,
        "chunk_key": chunk_key,
    }
    metadata_bytes = _canonical_json(metadata)
    payload_start = _align_up(_PREAMBLE.size + len(metadata_bytes))

    ordered_tensors = tuple(
        tensor
        for key_factors, value_factors in zip(chunk.key_layers, chunk.value_layers)
        for factors in (key_factors, value_factors)
        for tensor in _factor_tensors(factors)
    )
    if any(tensor.device.type == "cuda" for tensor in ordered_tensors):
        devices = {tensor.device for tensor in ordered_tensors}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("all serialized chunk tensors must share one CUDA device")
        device = next(iter(devices))
        entries = []
        cursor = payload_start
        for tensor in ordered_tensors:
            contiguous = tensor.detach().contiguous()
            if contiguous.dtype not in (torch.uint8, torch.float16):
                raise TypeError(f"unsupported chunk tensor dtype: {contiguous.dtype}")
            cursor = _align_up(cursor)
            nbytes = contiguous.numel() * contiguous.element_size()
            entries.append((cursor, nbytes, contiguous))
            cursor += nbytes

        payload_bytes = cursor - payload_start
        total_bytes = cursor + SVD_CHUNK_CHECKSUM_BYTES
        # Every tensor copy is enqueued onto the caller's CUDA stream and a
        # single stream fence makes the complete pinned blob CPU-readable.
        # The old tensor-by-tensor ``.cpu()`` path imposed hundreds of scalar
        # synchronization points for a multi-layer model.
        blob = torch.zeros(
            (total_bytes,),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        preamble = _PREAMBLE.pack(
            SVD_CHUNK_MAGIC,
            SVD_CHUNK_VERSION,
            _LITTLE_ENDIAN_FLAG,
            len(metadata_bytes),
            payload_bytes,
            total_bytes,
        )
        blob[: _PREAMBLE.size].copy_(
            torch.frombuffer(bytearray(preamble), dtype=torch.uint8)
        )
        blob[_PREAMBLE.size : _PREAMBLE.size + len(metadata_bytes)].copy_(
            torch.frombuffer(bytearray(metadata_bytes), dtype=torch.uint8)
        )

        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            for offset, nbytes, tensor in entries:
                source = tensor.view(torch.uint8).reshape(-1)
                blob[offset : offset + nbytes].copy_(source, non_blocking=True)
            stream.synchronize()

        checksum = hashlib.sha256(memoryview(blob[:cursor].numpy())).digest()
        blob[cursor:].copy_(torch.frombuffer(bytearray(checksum), dtype=torch.uint8))
        return blob

    buffer = bytearray(payload_start)
    buffer[_PREAMBLE.size : _PREAMBLE.size + len(metadata_bytes)] = metadata_bytes

    for tensor in ordered_tensors:
        _append_aligned_tensor(buffer, tensor)

    payload_bytes = len(buffer) - payload_start
    total_bytes = len(buffer) + SVD_CHUNK_CHECKSUM_BYTES
    buffer[: _PREAMBLE.size] = _PREAMBLE.pack(
        SVD_CHUNK_MAGIC,
        SVD_CHUNK_VERSION,
        _LITTLE_ENDIAN_FLAG,
        len(metadata_bytes),
        payload_bytes,
        total_bytes,
    )
    buffer.extend(hashlib.sha256(buffer).digest())
    return torch.frombuffer(buffer, dtype=torch.uint8).clone()


def _require_zero_padding(raw: bytes | memoryview, start: int, end: int) -> None:
    if any(raw[start:end]):
        raise ValueError("SVD chunk contains nonzero alignment padding")


def _validate_blob_storage(blob: torch.Tensor, *, cpu_only: bool = False) -> None:
    if not isinstance(blob, torch.Tensor):
        raise TypeError("blob must be a torch.Tensor")
    if cpu_only and blob.device.type != "cpu":
        raise TypeError("blob must be a one-dimensional CPU uint8 tensor")
    if blob.dtype != torch.uint8 or blob.ndim != 1:
        qualifier = "CPU " if cpu_only else ""
        raise TypeError(f"blob must be a one-dimensional {qualifier}uint8 tensor")
    if not blob.is_contiguous():
        raise ValueError("blob storage must be contiguous")


def _read_tensor_view(
    blob: torch.Tensor,
    raw: bytes | memoryview,
    *,
    name: str,
    cursor: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    payload_end: int,
) -> tuple[torch.Tensor, int, SVDChunkTensorView]:
    aligned = _align_up(cursor)
    _require_zero_padding(raw, cursor, aligned)
    element_size = torch.empty((), dtype=dtype).element_size()
    nbytes = math_prod(shape) * element_size
    end = aligned + nbytes
    if end > payload_end:
        raise ValueError("SVD chunk payload is truncated")
    view = SVDChunkTensorView(
        name=name,
        byte_offset=aligned,
        shape=shape,
        dtype=dtype,
    )
    return view.bind(blob), end, view


def math_prod(values: tuple[int, ...]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


def _scale_shape(
    *, rows: int, rank: int, scale_mode: str, group_size: int | None
) -> tuple[int, int]:
    if scale_mode == "matrix":
        return (1, 1)
    if scale_mode != "row" or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("invalid row scale contract in SVD chunk metadata")
    return (rows, (rank + group_size - 1) // group_size)


def _decode_factor(
    blob: torch.Tensor,
    raw: bytes | memoryview,
    *,
    name: str,
    cursor: int,
    chunk_tokens: int,
    spec: Mapping[str, Any],
    payload_end: int,
) -> tuple[QuantizedSVDFactors, int, tuple[SVDChunkTensorView, ...]]:
    try:
        feature_dim = int(spec["feature_dim"])
        rank = int(spec["rank"])
        bits_u = int(spec["bits_u"])
        bits_r = int(spec["bits_r"])
        scale_mode = str(spec["scale_mode"])
        group_size_u = spec["group_size_u"]
        group_size_r = spec["group_size_r"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid factor specification in SVD chunk metadata") from exc
    if bits_u not in SUPPORTED_FACTOR_BITS or bits_r not in SUPPORTED_FACTOR_BITS:
        raise ValueError("unsupported factor bit width in SVD chunk metadata")
    if feature_dim < 1 or rank < 1:
        raise ValueError("factor feature_dim and rank must be positive")
    if spec.get("packing") != "rank-low-bits-first-offset-binary":
        raise ValueError("unsupported factor packing order")
    if any(
        spec.get(name) != "float16"
        for name in ("scale_dtype", "right_scale_dtype", "sigma_dtype")
    ):
        raise ValueError("SVD chunk v1 requires FP16 scales and Sigma")

    packed_u = (rank * bits_u + 7) // 8
    packed_r = (rank * bits_r + 7) // 8
    scale_u_shape = _scale_shape(
        rows=chunk_tokens,
        rank=rank,
        scale_mode=scale_mode,
        group_size=group_size_u,
    )
    scale_r_shape = _scale_shape(
        rows=feature_dim,
        rank=rank,
        scale_mode=scale_mode,
        group_size=group_size_r,
    )
    tensor_views = []
    q_u, cursor, tensor_view = _read_tensor_view(
        blob,
        raw,
        name=f"{name}.u.qdata",
        cursor=cursor,
        shape=(chunk_tokens, packed_u),
        dtype=torch.uint8,
        payload_end=payload_end,
    )
    tensor_views.append(tensor_view)
    scale_u, cursor, tensor_view = _read_tensor_view(
        blob,
        raw,
        name=f"{name}.u.scale",
        cursor=cursor,
        shape=scale_u_shape,
        dtype=torch.float16,
        payload_end=payload_end,
    )
    tensor_views.append(tensor_view)
    sigma, cursor, tensor_view = _read_tensor_view(
        blob,
        raw,
        name=f"{name}.sigma",
        cursor=cursor,
        shape=(rank,),
        dtype=torch.float16,
        payload_end=payload_end,
    )
    tensor_views.append(tensor_view)
    q_r, cursor, tensor_view = _read_tensor_view(
        blob,
        raw,
        name=f"{name}.right.qdata",
        cursor=cursor,
        shape=(feature_dim, packed_r),
        dtype=torch.uint8,
        payload_end=payload_end,
    )
    tensor_views.append(tensor_view)
    scale_r, cursor, tensor_view = _read_tensor_view(
        blob,
        raw,
        name=f"{name}.right.scale",
        cursor=cursor,
        shape=scale_r_shape,
        dtype=torch.float16,
        payload_end=payload_end,
    )
    tensor_views.append(tensor_view)
    return (
        QuantizedSVDFactors(
            u=QuantizedFactor(
                qdata=q_u,
                scale=scale_u,
                bits=bits_u,
                rows=chunk_tokens,
                rank=rank,
                scale_mode=scale_mode,
                group_size=group_size_u,
            ),
            sigma=sigma,
            right=QuantizedFactor(
                qdata=q_r,
                scale=scale_r,
                bits=bits_r,
                rows=feature_dim,
                rank=rank,
                scale_mode=scale_mode,
                group_size=group_size_r,
            ),
        ),
        cursor,
        tuple(tensor_views),
    )


def _remap_factors(
    template: QuantizedSVDFactors,
    tensors: Sequence[torch.Tensor],
    offset: int,
) -> tuple[QuantizedSVDFactors, int]:
    q_u, scale_u, sigma, q_r, scale_r = tensors[offset : offset + 5]
    return (
        QuantizedSVDFactors(
            u=QuantizedFactor(
                qdata=q_u,
                scale=scale_u,
                bits=template.u.bits,
                rows=template.u.rows,
                rank=template.u.rank,
                scale_mode=template.u.scale_mode,
                group_size=template.u.group_size,
            ),
            sigma=sigma,
            right=QuantizedFactor(
                qdata=q_r,
                scale=scale_r,
                bits=template.right.bits,
                rows=template.right.rows,
                rank=template.right.rank,
                scale_mode=template.right.scale_mode,
                group_size=template.right.group_size,
            ),
        ),
        offset + 5,
    )


def _remap_chunk(
    template: QuantizedKVChunk,
    tensors: Sequence[torch.Tensor],
) -> QuantizedKVChunk:
    expected = template.layer_count * 10
    if len(tensors) != expected:
        raise ValueError(
            f"chunk layout requires {expected} tensors, got {len(tensors)}"
        )
    key_layers = []
    value_layers = []
    offset = 0
    for key_template, value_template in zip(template.key_layers, template.value_layers):
        key, offset = _remap_factors(key_template, tensors, offset)
        value, offset = _remap_factors(value_template, tensors, offset)
        key_layers.append(key)
        value_layers.append(value)
    return QuantizedKVChunk.from_sequences(key_layers, value_layers)


def _clone_chunk(chunk: QuantizedKVChunk) -> QuantizedKVChunk:
    tensors = []
    for key, value in zip(chunk.key_layers, chunk.value_layers):
        for factors in (key, value):
            tensors.extend(tensor.clone() for tensor in _factor_tensors(factors))
    return _remap_chunk(chunk, tensors)


def _prepare_parse_blob(
    blob: torch.Tensor,
    *,
    copy_blob: bool,
    pin_memory: bool,
) -> torch.Tensor:
    if not isinstance(blob, torch.Tensor):
        raise TypeError("blob must be a one-dimensional CPU uint8 tensor")
    if blob.dtype != torch.uint8 or blob.ndim != 1 or blob.device.type != "cpu":
        raise TypeError("blob must be a one-dimensional CPU uint8 tensor")
    if not isinstance(copy_blob, bool):
        raise TypeError("copy_blob must be a bool")
    if not isinstance(pin_memory, bool):
        raise TypeError("pin_memory must be a bool")

    contiguous = blob.detach().contiguous()
    if pin_memory:
        if not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True requires an available CUDA runtime")
        owner = torch.empty(
            contiguous.shape,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        owner.copy_(contiguous)
        return owner
    if copy_blob:
        return contiguous.clone(memory_format=torch.contiguous_format)
    return contiguous


def parse_svd_chunk_blob(
    blob: torch.Tensor,
    *,
    expected_key: str | None = None,
    expected_codec_metadata: Mapping[str, Any] | None = None,
    copy_blob: bool = True,
    pin_memory: bool = False,
) -> ParsedSVDChunkBlob:
    """Validate once and expose zero-copy factor views over one owned blob.

    Corruption, truncation, incompatible metadata, and key mismatches raise
    ``ValueError`` so callers can convert them into clean cache misses.  The
    default makes one whole-blob ownership copy.  ``copy_blob=False`` borrows a
    contiguous input (or makes one contiguous copy if needed); the caller must
    then preserve its immutable-by-contract bytes.  ``pin_memory=True`` makes a
    single owned pinned copy and requires CUDA.
    """

    owner = _prepare_parse_blob(
        blob,
        copy_blob=copy_blob,
        pin_memory=pin_memory,
    )
    _validate_blob_storage(owner, cpu_only=True)
    raw = memoryview(owner.numpy())
    if len(raw) < _PREAMBLE.size + SVD_CHUNK_CHECKSUM_BYTES:
        raise ValueError("SVD chunk is truncated before its preamble/checksum")
    magic, version, flags, metadata_len, payload_len, total_len = _PREAMBLE.unpack_from(
        raw
    )
    if magic != SVD_CHUNK_MAGIC or version != SVD_CHUNK_VERSION:
        raise ValueError("unsupported SVD chunk magic or version")
    if flags != _LITTLE_ENDIAN_FLAG:
        raise ValueError("unsupported SVD chunk byte-order flags")
    if total_len != len(raw):
        raise ValueError("SVD chunk total length does not match its preamble")
    checksum_start = len(raw) - SVD_CHUNK_CHECKSUM_BYTES
    if hashlib.sha256(raw[:checksum_start]).digest() != bytes(raw[checksum_start:]):
        raise ValueError("SVD chunk checksum mismatch")

    metadata_end = _PREAMBLE.size + metadata_len
    payload_start = _align_up(metadata_end)
    if payload_start > checksum_start:
        raise ValueError("SVD chunk metadata is truncated")
    _require_zero_padding(raw, metadata_end, payload_start)
    if payload_len != checksum_start - payload_start:
        raise ValueError("SVD chunk payload length does not match its preamble")
    metadata_bytes = bytes(raw[_PREAMBLE.size : metadata_end])
    try:
        metadata = json.loads(metadata_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SVD chunk metadata is not canonical JSON") from exc
    if _canonical_json(metadata) != metadata_bytes:
        raise ValueError("SVD chunk metadata is not canonically encoded")
    if not isinstance(metadata, dict):
        raise ValueError("SVD chunk metadata must be a JSON object")
    if metadata.get("version") != SVD_CHUNK_VERSION:
        raise ValueError("SVD chunk metadata version mismatch")
    codec = metadata.get("codec")
    if not isinstance(codec, dict):
        raise ValueError("SVD chunk codec metadata must be a JSON object")
    if metadata.get("codec_digest") != codec_digest(codec):
        raise ValueError("SVD chunk codec digest mismatch")
    try:
        recomputed_key = _make_svd_chunk_key_from_digests(
            codec_hash=metadata["codec_digest"],
            namespace_hash=metadata["namespace_digest"],
            end_page_hash=metadata["end_page_hash"],
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid SVD chunk key metadata") from exc
    if metadata.get("chunk_key") != recomputed_key:
        raise ValueError("SVD chunk key metadata mismatch")
    if expected_key is not None and expected_key != recomputed_key:
        raise ValueError("SVD chunk does not match the requested key")
    if expected_codec_metadata is not None and dict(expected_codec_metadata) != codec:
        raise ValueError("SVD chunk codec metadata is incompatible")

    try:
        chunk_tokens = int(metadata["chunk_tokens"])
        layer_count = int(metadata["layers"])
        key_spec = metadata["key"]
        value_spec = metadata["value"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid SVD chunk geometry metadata") from exc
    if chunk_tokens < 1 or layer_count < 1:
        raise ValueError("SVD chunk geometry must be positive")

    cursor = payload_start
    key_layers = []
    value_layers = []
    tensor_views = []
    for layer in range(layer_count):
        key, cursor, views = _decode_factor(
            owner,
            raw,
            name=f"key_layers.{layer}",
            cursor=cursor,
            chunk_tokens=chunk_tokens,
            spec=key_spec,
            payload_end=checksum_start,
        )
        tensor_views.extend(views)
        value, cursor, views = _decode_factor(
            owner,
            raw,
            name=f"value_layers.{layer}",
            cursor=cursor,
            chunk_tokens=chunk_tokens,
            spec=value_spec,
            payload_end=checksum_start,
        )
        tensor_views.extend(views)
        key_layers.append(key)
        value_layers.append(value)
    if cursor != checksum_start:
        raise ValueError("SVD chunk has trailing or missing payload bytes")
    return ParsedSVDChunkBlob(
        blob=owner,
        chunk=QuantizedKVChunk.from_sequences(key_layers, value_layers),
        metadata=metadata,
        tensor_views=tuple(tensor_views),
    )


def deserialize_svd_chunk(
    blob: torch.Tensor,
    *,
    expected_key: str | None = None,
    expected_codec_metadata: Mapping[str, Any] | None = None,
) -> tuple[QuantizedKVChunk, dict[str, Any]]:
    """Validate and decode a canonical chunk into independently-owned tensors.

    This compatibility API preserves the original ownership behavior.  New
    restore paths should use :func:`parse_svd_chunk_blob` to keep zero-copy
    payload views and remap them after one whole-blob device transfer.
    """

    parsed = parse_svd_chunk_blob(
        blob,
        expected_key=expected_key,
        expected_codec_metadata=expected_codec_metadata,
        copy_blob=False,
    )
    return _clone_chunk(parsed.chunk), dict(parsed.metadata)
