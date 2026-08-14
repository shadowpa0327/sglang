"""Prototype codec for joint-head SVD HiCache chunks.

This module intentionally contains only tensor contracts, factor quantization,
reconstruction, and exact byte accounting.  It does not alter HiCache lookup or
transfer semantics yet.  Both factors are packed along the retained-rank axis;
singular values remain unquantized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

ScaleMode = Literal["matrix", "row"]
SUPPORTED_FACTOR_BITS = (2, 4, 8)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_FACTOR_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_FACTOR_BITS}, got {bits}")


def _normalized_group_size(
    rank: int, scale_mode: ScaleMode, group_size: int | None
) -> int:
    if scale_mode == "matrix":
        if group_size is not None:
            raise ValueError("group_size is only valid for row scale mode")
        return rank
    if scale_mode != "row":
        raise ValueError("scale_mode must be 'matrix' or 'row'")
    group_size = rank if group_size is None else group_size
    if not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    return group_size


def fold_kv_heads(values: torch.Tensor) -> torch.Tensor:
    """Fold ``[tokens, kv_heads, head_dim]`` into ``[tokens, kv_heads*head_dim]``."""

    if values.ndim != 3:
        raise ValueError(
            "values must have shape [tokens, kv_heads, head_dim], "
            f"got {tuple(values.shape)}"
        )
    return values.reshape(values.shape[0], values.shape[1] * values.shape[2])


def view_nhd_as_joint_pages(values: torch.Tensor, *, page_size: int) -> torch.Tensor:
    """Zero-copy ``[slots,H,D] -> [pool_pages,P,H*D]`` view for paged SVD."""

    if values.ndim != 3:
        raise ValueError(
            "values must have shape [slots, kv_heads, head_dim], "
            f"got {tuple(values.shape)}"
        )
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if values.shape[0] % page_size != 0:
        raise ValueError("the slot dimension must be divisible by page_size")
    if not values.is_contiguous():
        raise ValueError("NHD values must be contiguous for a zero-copy paged view")
    pool_pages = values.shape[0] // page_size
    feature_dim = values.shape[1] * values.shape[2]
    return values.view(pool_pages, page_size, feature_dim)


def page_ids_from_device_indices(
    device_indices: torch.Tensor, *, page_size: int
) -> torch.Tensor:
    """Validate whole pages and return their IDs in logical prefix order."""

    if device_indices.ndim != 1 or device_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("device_indices must be a one-dimensional INT32/INT64 tensor")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if device_indices.numel() < page_size or device_indices.numel() % page_size:
        raise ValueError("device_indices must contain one or more complete pages")

    page_rows = device_indices.reshape(-1, page_size)
    page_starts = page_rows[:, 0]
    if not bool(torch.all(torch.remainder(page_starts, page_size) == 0)):
        raise ValueError("every selected page must begin at a page-aligned slot")
    expected = page_starts[:, None] + torch.arange(
        page_size, dtype=device_indices.dtype, device=device_indices.device
    )
    if not bool(torch.equal(page_rows, expected)):
        raise ValueError("each selected page must contain contiguous slots in order")
    return torch.div(page_starts, page_size, rounding_mode="floor")


def unfold_kv_heads(
    values: torch.Tensor, *, kv_heads: int, head_dim: int
) -> torch.Tensor:
    """Restore a folded ``[tokens, features]`` matrix to NHD layout."""

    if values.ndim != 2:
        raise ValueError(
            f"values must have shape [tokens, features], got {tuple(values.shape)}"
        )
    if kv_heads < 1 or head_dim < 1:
        raise ValueError("kv_heads and head_dim must be positive")
    expected_features = kv_heads * head_dim
    if values.shape[1] != expected_features:
        raise ValueError(
            f"folded feature width must be {expected_features}, got {values.shape[1]}"
        )
    return values.reshape(values.shape[0], kv_heads, head_dim)


def pack_signed_rank_values(
    values: torch.Tensor,
    *,
    bits: int,
    validate_range: bool = True,
) -> torch.Tensor:
    """Pack signed quantized values into bytes, low rank coordinate first.

    Logical signed values use the symmetric ranges ``[-1, 1]``, ``[-7, 7]``,
    or ``[-127, 127]``.  Codes are biased by ``2**(bits-1)``; code zero is
    intentionally unused.  Padding coordinates encode logical zero.
    """

    _validate_bits(bits)
    if values.ndim != 2:
        raise ValueError(
            f"values must have shape [rows, rank], got {tuple(values.shape)}"
        )
    if values.shape[0] < 1:
        raise ValueError("rows must be positive")
    if values.shape[1] < 1:
        raise ValueError("rank must be positive")
    if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("values must have a signed integer dtype")

    qmax = (1 << (bits - 1)) - 1
    if validate_range:
        value_min, value_max = torch.aminmax(values)
        if int(value_min) < -qmax or int(value_max) > qmax:
            raise ValueError(f"values must be in [-{qmax}, {qmax}] for INT{bits}")

    rows, rank = map(int, values.shape)
    values_per_byte = 8 // bits
    packed_rank = math.ceil(rank / values_per_byte)
    zero_code = 1 << (bits - 1)
    padded_codes = torch.full(
        (rows, packed_rank * values_per_byte),
        zero_code,
        dtype=torch.int16,
        device=values.device,
    )
    padded_codes[:, :rank] = values.to(torch.int16) + zero_code

    packed = torch.zeros(rows, packed_rank, dtype=torch.int16, device=values.device)
    for position in range(values_per_byte):
        packed.bitwise_or_(
            torch.bitwise_left_shift(
                padded_codes[:, position::values_per_byte], position * bits
            )
        )
    return packed.to(torch.uint8)


def unpack_signed_rank_values(
    packed: torch.Tensor, *, bits: int, rank: int
) -> torch.Tensor:
    """Unpack bytes produced by :func:`pack_signed_rank_values` to INT8."""

    _validate_bits(bits)
    if packed.ndim != 2 or packed.dtype != torch.uint8:
        raise TypeError("packed must be a uint8 tensor with shape [rows, packed_rank]")
    if rank < 1:
        raise ValueError("rank must be positive")

    values_per_byte = 8 // bits
    expected_packed_rank = math.ceil(rank / values_per_byte)
    if packed.shape[1] != expected_packed_rank:
        raise ValueError(
            f"packed rank must be {expected_packed_rank}, got {packed.shape[1]}"
        )

    mask = (1 << bits) - 1
    raw = packed.to(torch.int16)
    codes = [
        torch.bitwise_and(torch.bitwise_right_shift(raw, position * bits), mask)
        for position in range(values_per_byte)
    ]
    interleaved = torch.stack(codes, dim=-1).reshape(packed.shape[0], -1)
    zero_code = 1 << (bits - 1)
    return (interleaved[:, :rank] - zero_code).to(torch.int8)


@dataclass(frozen=True)
class QuantizedFactor:
    """One bit-packed two-dimensional SVD factor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    bits: int
    rows: int
    rank: int
    scale_mode: ScaleMode
    group_size: int | None

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        if self.rows < 1 or self.rank < 1:
            raise ValueError("rows and rank must be positive")
        if self.qdata.dtype != torch.uint8:
            raise TypeError("qdata must use torch.uint8")
        expected_qdata = (self.rows, math.ceil(self.rank * self.bits / 8))
        if tuple(self.qdata.shape) != expected_qdata:
            raise ValueError(f"qdata must have shape {expected_qdata}")
        if not self.scale.is_floating_point():
            raise TypeError("scale must have a floating dtype")
        if self.qdata.device != self.scale.device:
            raise ValueError("qdata and scale must be on the same device")

        normalized_group = _normalized_group_size(
            self.rank, self.scale_mode, self.group_size
        )
        if self.scale_mode == "matrix":
            expected_scale = (1, 1)
        else:
            expected_scale = (
                self.rows,
                math.ceil(self.rank / normalized_group),
            )
        if tuple(self.scale.shape) != expected_scale:
            raise ValueError(f"scale must have shape {expected_scale}")

    @property
    def storage_nbytes(self) -> int:
        return _tensor_nbytes(self.qdata) + _tensor_nbytes(self.scale)

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.dequantize_rows(0, self.rows, dtype=dtype)

    def dequantize_rows(
        self,
        row_start: int,
        row_end: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize a half-open row slice without expanding other rows."""

        if not 0 <= row_start <= row_end <= self.rows:
            raise ValueError(f"row range must satisfy 0 <= start <= end <= {self.rows}")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dequantization dtype must be floating point")
        signed = unpack_signed_rank_values(
            self.qdata[row_start:row_end], bits=self.bits, rank=self.rank
        ).to(dtype)
        if self.scale_mode == "matrix":
            expanded_scale = self.scale
        else:
            group_size = _normalized_group_size(
                self.rank, self.scale_mode, self.group_size
            )
            expanded_scale = self.scale[row_start:row_end].repeat_interleave(
                group_size, dim=1
            )[:, : self.rank]
        return signed * expanded_scale.to(dtype)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> QuantizedFactor:
        """Copy packed data and scales while preserving the factor contract."""

        return QuantizedFactor(
            qdata=self.qdata.to(device=device, non_blocking=non_blocking),
            scale=self.scale.to(device=device, non_blocking=non_blocking),
            bits=self.bits,
            rows=self.rows,
            rank=self.rank,
            scale_mode=self.scale_mode,
            group_size=self.group_size,
        )


def quantize_factor(
    values: torch.Tensor,
    *,
    bits: int,
    scale_mode: ScaleMode = "matrix",
    group_size: int | None = None,
    scale_dtype: torch.dtype = torch.float16,
) -> QuantizedFactor:
    """Symmetrically quantize and bit-pack one ``[rows, rank]`` factor."""

    _validate_bits(bits)
    if values.ndim != 2 or not values.is_floating_point():
        raise TypeError("values must be a floating tensor with shape [rows, rank]")
    rows, rank = map(int, values.shape)
    if rows < 1 or rank < 1:
        raise ValueError("rows and rank must be positive")
    if not torch.empty((), dtype=scale_dtype).is_floating_point():
        raise TypeError("scale_dtype must be floating point")

    normalized_group = _normalized_group_size(rank, scale_mode, group_size)
    work = values.to(torch.float32)
    qmax = (1 << (bits - 1)) - 1

    if scale_mode == "matrix":
        maxima = work.abs().amax().reshape(1, 1)
    else:
        group_count = math.ceil(rank / normalized_group)
        padded_rank = group_count * normalized_group
        if padded_rank == rank:
            padded = work
        else:
            padded = torch.cat(
                (
                    work,
                    torch.zeros(
                        rows,
                        padded_rank - rank,
                        dtype=work.dtype,
                        device=work.device,
                    ),
                ),
                dim=1,
            )
        maxima = padded.reshape(rows, group_count, normalized_group).abs().amax(dim=2)

    scale = torch.where(maxima == 0, torch.ones_like(maxima), maxima / qmax).to(
        scale_dtype
    )
    # FP16 can underflow an extremely small nonzero scale.  Such a factor is
    # represented as all zeros instead of producing infinities during encode.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    if scale_mode == "matrix":
        expanded_scale = scale
    else:
        expanded_scale = scale.repeat_interleave(normalized_group, dim=1)[:, :rank]

    signed = (
        torch.round(work / expanded_scale.to(torch.float32))
        .clamp(-qmax, qmax)
        .to(torch.int8)
    )
    # ``signed`` was just clamped to the representable range.  Re-validating
    # it would force a GPU->CPU scalar synchronization for every U/R factor.
    qdata = pack_signed_rank_values(signed, bits=bits, validate_range=False)
    return QuantizedFactor(
        qdata=qdata,
        scale=scale,
        bits=bits,
        rows=rows,
        rank=rank,
        scale_mode=scale_mode,
        group_size=None if scale_mode == "matrix" else normalized_group,
    )


@dataclass(frozen=True)
class QuantizedSVDFactors:
    """Packed ``U`` and right factor ``R`` with unquantized singular values."""

    u: QuantizedFactor
    sigma: torch.Tensor
    right: QuantizedFactor

    def __post_init__(self) -> None:
        if self.u.rank != self.right.rank:
            raise ValueError("U and R must have the same retained rank")
        if tuple(self.sigma.shape) != (self.u.rank,):
            raise ValueError(f"sigma must have shape ({self.u.rank},)")
        if not self.sigma.is_floating_point():
            raise TypeError("sigma must have a floating dtype")
        if (
            self.sigma.device != self.u.qdata.device
            or self.sigma.device != self.right.qdata.device
        ):
            raise ValueError("U, sigma, and R must be on the same device")

    @property
    def chunk_tokens(self) -> int:
        return self.u.rows

    @property
    def feature_dim(self) -> int:
        return self.right.rows

    @property
    def rank(self) -> int:
        return self.u.rank

    @property
    def storage_nbytes(self) -> int:
        return (
            self.u.storage_nbytes
            + _tensor_nbytes(self.sigma)
            + self.right.storage_nbytes
        )

    def reconstruct(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.reconstruct_rows(0, self.chunk_tokens, dtype=dtype)

    def reconstruct_rows(
        self,
        row_start: int,
        row_end: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Reconstruct only selected token rows as ``[rows, feature_dim]``."""

        u = self.u.dequantize_rows(row_start, row_end, dtype=dtype)
        right = self.right.dequantize(dtype=dtype)
        sigma = self.sigma.to(dtype)
        return (u * sigma) @ right.transpose(0, 1)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> QuantizedSVDFactors:
        """Copy all packed factors to a reconstruction device."""

        return QuantizedSVDFactors(
            u=self.u.to(device, non_blocking=non_blocking),
            sigma=self.sigma.to(device=device, non_blocking=non_blocking),
            right=self.right.to(device, non_blocking=non_blocking),
        )


def quantize_svd_factors(
    u: torch.Tensor,
    sigma: torch.Tensor,
    right: torch.Tensor,
    *,
    bits_u: int,
    bits_r: int | None = None,
    scale_mode: ScaleMode = "matrix",
    group_size_u: int | None = None,
    group_size_r: int | None = None,
    scale_dtype: torch.dtype = torch.float16,
    sigma_dtype: torch.dtype = torch.float16,
) -> QuantizedSVDFactors:
    """Pack a single-matrix paged-SVD result.

    ``sigma`` accepts ``[rank]`` or ``[1, rank]`` and ``right`` accepts
    ``[features, rank]`` or the paged-SVD batch shape ``[1, features, rank]``.
    """

    if u.ndim != 2:
        raise ValueError(f"u must have shape [tokens, rank], got {tuple(u.shape)}")
    if sigma.ndim == 2 and sigma.shape[0] == 1:
        sigma = sigma[0]
    if right.ndim == 3 and right.shape[0] == 1:
        right = right[0]
    if sigma.ndim != 1:
        raise ValueError(
            "sigma must have shape [rank] or [1, rank], " f"got {tuple(sigma.shape)}"
        )
    if right.ndim != 2:
        raise ValueError(
            "right must have shape [features, rank] or [1, features, rank], "
            f"got {tuple(right.shape)}"
        )
    if u.shape[1] != sigma.shape[0] or right.shape[1] != sigma.shape[0]:
        raise ValueError("U, sigma, and R retained-rank dimensions must match")

    bits_r = bits_u if bits_r is None else bits_r
    return QuantizedSVDFactors(
        u=quantize_factor(
            u,
            bits=bits_u,
            scale_mode=scale_mode,
            group_size=group_size_u,
            scale_dtype=scale_dtype,
        ),
        sigma=sigma.to(sigma_dtype),
        right=quantize_factor(
            right,
            bits=bits_r,
            scale_mode=scale_mode,
            group_size=group_size_r,
            scale_dtype=scale_dtype,
        ),
    )


def estimate_quantized_factor_nbytes(
    *,
    rows: int,
    rank: int,
    bits: int,
    scale_mode: ScaleMode,
    group_size: int | None = None,
    scale_bytes: int = 2,
) -> int:
    """Exact payload plus scale bytes for one factor matrix."""

    _validate_bits(bits)
    if rows < 1 or rank < 1 or scale_bytes < 1:
        raise ValueError("rows, rank, and scale_bytes must be positive")
    normalized_group = _normalized_group_size(rank, scale_mode, group_size)
    payload = rows * math.ceil(rank * bits / 8)
    if scale_mode == "matrix":
        scale_count = 1
    else:
        scale_count = rows * math.ceil(rank / normalized_group)
    return payload + scale_count * scale_bytes


def estimate_quantized_svd_nbytes(
    *,
    chunk_tokens: int,
    feature_dim: int,
    rank: int,
    bits_u: int,
    bits_r: int | None = None,
    scale_mode: ScaleMode = "matrix",
    group_size_u: int | None = None,
    group_size_r: int | None = None,
    scale_bytes: int = 2,
    sigma_bytes: int = 2,
) -> int:
    """Exact bytes for one layer and one of K/V, excluding object metadata."""

    if sigma_bytes < 1:
        raise ValueError("sigma_bytes must be positive")
    bits_r = bits_u if bits_r is None else bits_r
    return (
        estimate_quantized_factor_nbytes(
            rows=chunk_tokens,
            rank=rank,
            bits=bits_u,
            scale_mode=scale_mode,
            group_size=group_size_u,
            scale_bytes=scale_bytes,
        )
        + rank * sigma_bytes
        + estimate_quantized_factor_nbytes(
            rows=feature_dim,
            rank=rank,
            bits=bits_r,
            scale_mode=scale_mode,
            group_size=group_size_r,
            scale_bytes=scale_bytes,
        )
    )


def estimate_quantized_kv_chunk_nbytes(
    *,
    layers: int,
    chunk_tokens: int,
    key_features: int,
    value_features: int,
    key_rank: int,
    value_rank: int,
    bits_u: int,
    bits_r: int | None = None,
    scale_mode: ScaleMode = "matrix",
    group_size_u: int | None = None,
    group_size_r: int | None = None,
    scale_bytes: int = 2,
    sigma_bytes: int = 2,
) -> int:
    """Exact K+V packed bytes for a complete all-layer chunk."""

    if layers < 1:
        raise ValueError("layers must be positive")
    per_key = estimate_quantized_svd_nbytes(
        chunk_tokens=chunk_tokens,
        feature_dim=key_features,
        rank=key_rank,
        bits_u=bits_u,
        bits_r=bits_r,
        scale_mode=scale_mode,
        group_size_u=group_size_u,
        group_size_r=group_size_r,
        scale_bytes=scale_bytes,
        sigma_bytes=sigma_bytes,
    )
    per_value = estimate_quantized_svd_nbytes(
        chunk_tokens=chunk_tokens,
        feature_dim=value_features,
        rank=value_rank,
        bits_u=bits_u,
        bits_r=bits_r,
        scale_mode=scale_mode,
        group_size_u=group_size_u,
        group_size_r=group_size_r,
        scale_bytes=scale_bytes,
        sigma_bytes=sigma_bytes,
    )
    return layers * (per_key + per_value)


def raw_kv_chunk_nbytes(
    *,
    layers: int,
    chunk_tokens: int,
    key_features: int,
    value_features: int,
    element_size: int,
) -> int:
    """Raw all-layer K+V storage for the same chunk geometry."""

    values = (layers, chunk_tokens, key_features, value_features, element_size)
    if any(value < 1 for value in values):
        raise ValueError("all dimensions and element_size must be positive")
    return layers * chunk_tokens * (key_features + value_features) * element_size
