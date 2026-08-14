# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""Fused GPU reconstruction for quantized HiCache SVD factors.

The kernel consumes rank-packed INT2/INT4/INT8 ``U`` and ``R`` factors,
dequantizes them to FP16 registers, applies ``Sigma``, performs one tensor-core
GEMM with either FP16 or Hopper FP8-E4M3 operands, and writes directly to
arbitrary paged-KV slots.  It deliberately avoids materializing dequantized
factors or a dense reconstruction tensor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.srt.mem_cache.hicache_svd_codec import QuantizedSVDFactors

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep CPU-only codec and connector tests importable.
    triton = None
    tl = None

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # Preserve pointer kernels with older Triton installations.
    TensorDescriptor = None


SUPPORTED_SVD_RESTORE_COMPUTE = ("fp16", "fp8_e4m3")
_fused_quantized_svd_restore_tma_kernel = None
_fused_quantized_svd_restore_batch_tma_kernel = None


if triton is not None:

    @triton.jit
    def _fused_quantized_svd_restore_kernel(
        u_qdata_ptr,
        u_scale_ptr,
        sigma_ptr,
        right_qdata_ptr,
        right_scale_ptr,
        slots_ptr,
        output_ptr,
        rows,
        features,
        rank,
        stride_uq_row,
        stride_uq_packed,
        stride_us_row,
        stride_us_group,
        stride_rq_row,
        stride_rq_packed,
        stride_rs_row,
        stride_rs_group,
        stride_output_row,
        stride_output_feature,
        BITS_U: tl.constexpr,
        BITS_R: tl.constexpr,
        U_ROW_SCALE: tl.constexpr,
        R_ROW_SCALE: tl.constexpr,
        U_GROUP_SIZE: tl.constexpr,
        R_GROUP_SIZE: tl.constexpr,
        FP8_COMPUTE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        mask_m = offsets_m < rows
        mask_n = offsets_n < features
        mask_k = offsets_k < rank

        values_per_u_byte: tl.constexpr = 8 // BITS_U
        u_byte_offsets = offsets_k // values_per_u_byte
        u_shifts = (offsets_k % values_per_u_byte) * BITS_U
        u_packed = tl.load(
            u_qdata_ptr
            + offsets_m[:, None] * stride_uq_row
            + u_byte_offsets[None, :] * stride_uq_packed,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int32)
        u_codes = (u_packed >> u_shifts[None, :]) & ((1 << BITS_U) - 1)
        u_signed = u_codes - (1 << (BITS_U - 1))
        if U_ROW_SCALE:
            u_groups = offsets_k // U_GROUP_SIZE
            u_scale = tl.load(
                u_scale_ptr
                + offsets_m[:, None] * stride_us_row
                + u_groups[None, :] * stride_us_group,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
        else:
            u_scale = tl.load(u_scale_ptr)
        sigma = tl.load(sigma_ptr + offsets_k, mask=mask_k, other=0.0)
        u = (
            u_signed.to(tl.float16)
            * u_scale.to(tl.float16)
            * sigma[None, :].to(tl.float16)
        )

        values_per_r_byte: tl.constexpr = 8 // BITS_R
        r_byte_offsets = offsets_k // values_per_r_byte
        r_shifts = (offsets_k % values_per_r_byte) * BITS_R
        r_packed = tl.load(
            right_qdata_ptr
            + offsets_n[None, :] * stride_rq_row
            + r_byte_offsets[:, None] * stride_rq_packed,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.int32)
        r_codes = (r_packed >> r_shifts[:, None]) & ((1 << BITS_R) - 1)
        r_signed = r_codes - (1 << (BITS_R - 1))
        if R_ROW_SCALE:
            r_groups = offsets_k // R_GROUP_SIZE
            r_scale = tl.load(
                right_scale_ptr
                + offsets_n[None, :] * stride_rs_row
                + r_groups[:, None] * stride_rs_group,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
        else:
            r_scale = tl.load(right_scale_ptr)
        right = r_signed.to(tl.float16) * r_scale.to(tl.float16)

        # Hopper only emits FP8 WGMMA for this path when the launch uses an
        # M=64 tile.  Keeping the accumulator/output FP16 avoids introducing
        # an FP32 reconstruction tile while still exercising E4M3 operands.
        if FP8_COMPUTE:
            reconstructed = tl.dot(
                u.to(tl.float8e4nv),
                right.to(tl.float8e4nv),
                out_dtype=tl.float16,
            )
        else:
            reconstructed = tl.dot(u, right, out_dtype=tl.float16)
        destination_rows = tl.load(slots_ptr + offsets_m, mask=mask_m, other=0).to(
            tl.int64
        )
        tl.store(
            output_ptr
            + destination_rows[:, None] * stride_output_row
            + offsets_n[None, :] * stride_output_feature,
            reconstructed,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fused_quantized_svd_restore_int4_rank128_fp8_kernel(
        u_qdata_ptr,
        u_scale_ptr,
        sigma_ptr,
        right_qdata_ptr,
        right_scale_ptr,
        slots_ptr,
        output_ptr,
        rows,
        features,
        stride_uq_row,
        stride_uq_packed,
        stride_us_row,
        stride_us_group,
        stride_rq_row,
        stride_rq_packed,
        stride_rs_row,
        stride_rs_group,
        stride_output_row,
        stride_output_feature,
        U_ROW_SCALE: tl.constexpr,
        R_ROW_SCALE: tl.constexpr,
        U_GROUP_SIZE: tl.constexpr,
        R_GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Pointer-load specialization for INT4/rank-128 Hopper restores.

        The generic kernel addresses packed data with one logical load per
        retained-rank value, so every INT4 byte is loaded twice.  This kernel
        instead loads each [row, 64-byte] packed tile once and expands its low
        and high nibbles into the rank-128 order in registers.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_packed_k = tl.arange(0, 64)
        offsets_k = tl.arange(0, 128)
        mask_m = offsets_m < rows
        mask_n = offsets_n < features

        u_packed = tl.load(
            u_qdata_ptr
            + offsets_m[:, None] * stride_uq_row
            + offsets_packed_k[None, :] * stride_uq_packed,
            mask=mask_m[:, None],
            other=0,
        ).to(tl.int32)
        u_codes = tl.interleave(u_packed & 0xF, (u_packed >> 4) & 0xF)
        u_signed = u_codes - 8
        if U_ROW_SCALE:
            u_groups = offsets_k // U_GROUP_SIZE
            u_scale = tl.load(
                u_scale_ptr
                + offsets_m[:, None] * stride_us_row
                + u_groups[None, :] * stride_us_group,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            u_scale = tl.load(u_scale_ptr)
        sigma = tl.load(sigma_ptr + offsets_k)
        u = (
            u_signed.to(tl.float16)
            * u_scale.to(tl.float16)
            * sigma[None, :].to(tl.float16)
        )

        right_packed = tl.load(
            right_qdata_ptr
            + offsets_n[:, None] * stride_rq_row
            + offsets_packed_k[None, :] * stride_rq_packed,
            mask=mask_n[:, None],
            other=0,
        ).to(tl.int32)
        right_codes = tl.interleave(
            right_packed & 0xF,
            (right_packed >> 4) & 0xF,
        )
        right_signed = right_codes - 8
        if R_ROW_SCALE:
            r_groups = offsets_k // R_GROUP_SIZE
            right_scale = tl.load(
                right_scale_ptr
                + offsets_n[:, None] * stride_rs_row
                + r_groups[None, :] * stride_rs_group,
                mask=mask_n[:, None],
                other=0.0,
            )
        else:
            right_scale = tl.load(right_scale_ptr)
        right = tl.trans(right_signed.to(tl.float16) * right_scale.to(tl.float16))

        reconstructed = tl.dot(
            u.to(tl.float8e4nv),
            right.to(tl.float8e4nv),
            out_dtype=tl.float16,
        )
        destination_rows = tl.load(
            slots_ptr + offsets_m,
            mask=mask_m,
            other=0,
        ).to(tl.int64)
        tl.store(
            output_ptr
            + destination_rows[:, None] * stride_output_row
            + offsets_n[None, :] * stride_output_feature,
            reconstructed,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fused_quantized_svd_restore_tma_kernel(
        u_qdata_desc,
        u_scale_ptr,
        sigma_ptr,
        right_qdata_desc,
        right_scale_ptr,
        slots_ptr,
        output_ptr,
        rows,
        features,
        stride_us_row,
        stride_us_group,
        stride_rs_row,
        stride_rs_group,
        stride_output_row,
        stride_output_feature,
        U_ROW_SCALE: tl.constexpr,
        R_ROW_SCALE: tl.constexpr,
        U_GROUP_SIZE: tl.constexpr,
        R_GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Hopper-only INT4/rank-128 restore with two explicit TMA reads."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, 128)
        mask_m = offsets_m < rows
        mask_n = offsets_n < features

        # The descriptors read one packed U [64,64] tile and one packed R
        # [128,64] tile. INT4 values are ordered low nibble then high nibble,
        # so interleave expands packed-K=64 to retained-rank 128.
        u_packed = u_qdata_desc.load([pid_m * BLOCK_M, 0]).to(tl.int32)
        u_codes = tl.interleave(u_packed & 0xF, (u_packed >> 4) & 0xF)
        u_signed = u_codes - 8
        if U_ROW_SCALE:
            u_groups = offsets_k // U_GROUP_SIZE
            u_scale = tl.load(
                u_scale_ptr
                + offsets_m[:, None] * stride_us_row
                + u_groups[None, :] * stride_us_group,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            u_scale = tl.load(u_scale_ptr)
        sigma = tl.load(sigma_ptr + offsets_k)
        u = (
            u_signed.to(tl.float16)
            * u_scale.to(tl.float16)
            * sigma[None, :].to(tl.float16)
        )

        right_packed = right_qdata_desc.load([pid_n * BLOCK_N, 0]).to(tl.int32)
        right_codes = tl.interleave(
            right_packed & 0xF,
            (right_packed >> 4) & 0xF,
        )
        right_signed = right_codes - 8
        if R_ROW_SCALE:
            r_groups = offsets_k // R_GROUP_SIZE
            right_scale = tl.load(
                right_scale_ptr
                + offsets_n[:, None] * stride_rs_row
                + r_groups[None, :] * stride_rs_group,
                mask=mask_n[:, None],
                other=0.0,
            )
        else:
            right_scale = tl.load(right_scale_ptr)
        right = tl.trans(right_signed.to(tl.float16) * right_scale.to(tl.float16))

        reconstructed = tl.dot(
            u.to(tl.float8e4nv),
            right.to(tl.float8e4nv),
            out_dtype=tl.float16,
        )
        destination_rows = tl.load(slots_ptr + offsets_m, mask=mask_m, other=0).to(
            tl.int64
        )
        # Page-table slots are arbitrary, so this final scatter deliberately
        # remains a masked pointer store; it is not TMA-compatible.
        tl.store(
            output_ptr
            + destination_rows[:, None] * stride_output_row
            + offsets_n[None, :] * stride_output_feature,
            reconstructed,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    # Prefix hits may start at any token within the first 4K chunk.  Keeping
    # this scalar out of Triton's divisibility specialization prevents a new
    # kernel compile on the first partial hit.
    @triton.jit(do_not_specialize=["first_row_start"])
    def _fused_quantized_svd_restore_batch_tma_kernel(
        u_qdata_desc,
        u_scale_ptr,
        sigma_ptr,
        right_qdata_desc,
        right_scale_ptr,
        slots_ptr,
        output_ptr,
        chunk_rows,
        features,
        first_row_start,
        blob_stride_fp16,
        stride_us_row,
        stride_us_group,
        stride_rs_row,
        stride_rs_group,
        stride_output_row,
        stride_output_feature,
        U_ROW_SCALE: tl.constexpr,
        R_ROW_SCALE: tl.constexpr,
        U_GROUP_SIZE: tl.constexpr,
        R_GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Restore one operation's uniformly padded chunks in one 3-D grid."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_j = tl.program_id(2)
        chunk_row_start = tl.where(pid_j == 0, first_row_start, 0)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        source_rows = chunk_row_start + offsets_m
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, 128)
        mask_m = source_rows < chunk_rows
        mask_n = offsets_n < features
        chunk_fp16_offset = pid_j * blob_stride_fp16

        # The singleton leading block dimension selects one serialized chunk.
        # Reshape it away after the 3-D TMA transaction so the register tile is
        # the same [M,packed-K] operand used by the per-chunk specialization.
        u_packed = u_qdata_desc.load(
            [pid_j, pid_m * BLOCK_M + chunk_row_start, 0]
        ).reshape((BLOCK_M, 64))
        u_packed = u_packed.to(tl.int32)
        u_codes = tl.interleave(u_packed & 0xF, (u_packed >> 4) & 0xF)
        u_signed = u_codes - 8
        if U_ROW_SCALE:
            u_groups = offsets_k // U_GROUP_SIZE
            u_scale = tl.load(
                u_scale_ptr
                + chunk_fp16_offset
                + source_rows[:, None] * stride_us_row
                + u_groups[None, :] * stride_us_group,
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            u_scale = tl.load(u_scale_ptr + chunk_fp16_offset)
        sigma = tl.load(sigma_ptr + chunk_fp16_offset + offsets_k)
        u = (
            u_signed.to(tl.float16)
            * u_scale.to(tl.float16)
            * sigma[None, :].to(tl.float16)
        )

        right_packed = right_qdata_desc.load([pid_j, pid_n * BLOCK_N, 0]).reshape(
            (BLOCK_N, 64)
        )
        right_packed = right_packed.to(tl.int32)
        right_codes = tl.interleave(
            right_packed & 0xF,
            (right_packed >> 4) & 0xF,
        )
        right_signed = right_codes - 8
        if R_ROW_SCALE:
            r_groups = offsets_k // R_GROUP_SIZE
            right_scale = tl.load(
                right_scale_ptr
                + chunk_fp16_offset
                + offsets_n[:, None] * stride_rs_row
                + r_groups[None, :] * stride_rs_group,
                mask=mask_n[:, None],
                other=0.0,
            )
        else:
            right_scale = tl.load(right_scale_ptr + chunk_fp16_offset)
        right = tl.trans(right_signed.to(tl.float16) * right_scale.to(tl.float16))

        reconstructed = tl.dot(
            u.to(tl.float8e4nv),
            right.to(tl.float8e4nv),
            out_dtype=tl.float16,
        )
        # Slots from all chunks are contiguous in operation order even though
        # their values are an arbitrary page-table scatter.  Chunk zero may be
        # a suffix; every later chunk is complete.
        slot_offsets = pid_j * chunk_rows - first_row_start + source_rows
        destination_rows = tl.load(
            slots_ptr + slot_offsets,
            mask=mask_m,
            other=0,
        ).to(tl.int64)
        tl.store(
            output_ptr
            + destination_rows[:, None] * stride_output_row
            + offsets_n[None, :] * stride_output_feature,
            reconstructed,
            mask=mask_m[:, None] & mask_n[None, :],
        )


_TMA_ALLOCATOR_SET = False


def _set_triton_tma_allocator() -> None:
    """Install Triton's device-side descriptor allocator once per process."""

    global _TMA_ALLOCATOR_SET
    if _TMA_ALLOCATOR_SET:
        return

    def alloc_fn(size: int, alignment: int, stream: int | None):
        del alignment, stream
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    triton.set_allocator(alloc_fn)
    _TMA_ALLOCATOR_SET = True


def _validate_tma_contract(
    factors: QuantizedSVDFactors,
    destination: torch.Tensor,
    *,
    compute_mode: str,
) -> None:
    """Fail closed unless the explicit Hopper TMA specialization can run."""

    if TensorDescriptor is None:
        raise RuntimeError("TMA SVD restore requires Triton TensorDescriptor support")
    if compute_mode != "fp8_e4m3":
        raise ValueError("TMA SVD restore requires compute_mode='fp8_e4m3'")
    if torch.version.hip is not None or torch.cuda.get_device_capability(
        destination.device
    ) < (9, 0):
        raise RuntimeError("TMA SVD restore requires NVIDIA compute capability 9.0+")
    if factors.rank != 128:
        raise ValueError("TMA SVD restore requires retained rank exactly 128")
    if factors.u.bits != 4 or factors.right.bits != 4:
        raise ValueError("TMA SVD restore requires INT4 U and INT4 R factors")

    for label, qdata in (("U", factors.u.qdata), ("R", factors.right.qdata)):
        if qdata.ndim != 2 or qdata.shape[1] != 64:
            raise ValueError(f"TMA {label} qdata must have shape [rows,64]")
        if not qdata.is_contiguous() or qdata.stride() != (64, 1):
            raise ValueError(
                f"TMA {label} qdata must be contiguous with row stride 64 bytes"
            )
        if qdata.data_ptr() % 16:
            raise ValueError(f"TMA {label} qdata base must be 16-byte aligned")


def fused_svd_restore_available() -> bool:
    """Return whether the optional Triton implementation can be launched."""

    return triton is not None


def fused_reconstruct_svd_batch_into(
    factors: Sequence[QuantizedSVDFactors],
    destination_slots: torch.Tensor,
    destination: torch.Tensor,
    *,
    first_row_start: int = 0,
    compute_mode: str = "fp8_e4m3",
) -> bool:
    """Try one 3-D TMA launch for a contiguous operation's chunk factors.

    The optimization is intentionally strict: at least two full rank-128 INT4
    chunks must be views of one CUDA allocation at a common 64-byte-aligned
    padded-blob stride.  Every factor tensor must occupy the same byte offset
    in each blob.  ``False`` means the caller must use the existing per-chunk
    path; invalid destination/slot arguments and unsupported hardware still
    fail closed with an exception.
    """

    if triton is None:
        raise RuntimeError("fused SVD restore requires Triton")
    if not isinstance(factors, Sequence):
        raise TypeError("factors must be a sequence")
    chunk_factors = tuple(factors)
    if len(chunk_factors) < 2:
        return False
    if isinstance(first_row_start, bool) or not isinstance(first_row_start, int):
        raise TypeError("first_row_start must be an integer")

    first = chunk_factors[0]
    _validate_tma_contract(first, destination, compute_mode=compute_mode)
    chunk_rows = first.chunk_tokens
    features = first.feature_dim
    if not 0 <= first_row_start < chunk_rows:
        raise ValueError("first_row_start must select a row in the first chunk")
    expected_slots = len(chunk_factors) * chunk_rows - first_row_start
    if destination_slots.ndim != 1 or destination_slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("destination_slots must be one-dimensional INT32/INT64")
    if destination_slots.numel() != expected_slots:
        raise ValueError(
            "destination_slots must cover the first chunk suffix and all "
            "subsequent chunks"
        )
    if not destination_slots.is_contiguous():
        return False
    if destination.device.type != "cuda":
        raise ValueError("batched TMA restore requires a CUDA destination")
    if destination.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused SVD restore destination must be FP16 or BF16")
    if destination.ndim not in (2, 3) or not destination.is_contiguous():
        raise ValueError(
            "destination must be contiguous [slots,features] or [slots,heads,dim]"
        )
    output = destination.view(destination.shape[0], -1)
    if output.shape[1] != features:
        raise ValueError(
            f"destination feature width must be {features}, got {output.shape[1]}"
        )
    if destination_slots.device != destination.device:
        raise ValueError("destination_slots and destination must share a device")

    def tensor_tuple(value: QuantizedSVDFactors) -> tuple[torch.Tensor, ...]:
        return (
            value.u.qdata,
            value.u.scale,
            value.sigma,
            value.right.qdata,
            value.right.scale,
        )

    def factor_signature(value: QuantizedSVDFactors) -> tuple[object, ...]:
        tensors = tensor_tuple(value)
        return (
            value.chunk_tokens,
            value.feature_dim,
            value.rank,
            value.u.bits,
            value.right.bits,
            value.u.scale_mode,
            value.right.scale_mode,
            value.u.group_size,
            value.right.group_size,
            *((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
        )

    signature = factor_signature(first)
    first_tensors = tensor_tuple(first)
    storage_base = first_tensors[0].untyped_storage().data_ptr()
    first_devices = {tensor.device for tensor in first_tensors}
    if first_devices != {destination.device}:
        raise ValueError("all SVD factors and destination must share a device")
    if any(not tensor.is_contiguous() for tensor in first_tensors):
        return False
    if (
        first.u.qdata.dtype != torch.uint8
        or first.right.qdata.dtype != torch.uint8
        or any(
            tensor.dtype != torch.float16
            for tensor in (first.u.scale, first.sigma, first.right.scale)
        )
    ):
        return False

    second_u_ptr = chunk_factors[1].u.qdata.data_ptr()
    blob_stride_bytes = second_u_ptr - first.u.qdata.data_ptr()
    if blob_stride_bytes <= 0 or blob_stride_bytes % 64:
        return False

    for chunk_index, value in enumerate(chunk_factors):
        try:
            _validate_tma_contract(value, destination, compute_mode=compute_mode)
        except (TypeError, ValueError):
            return False
        if factor_signature(value) != signature:
            return False
        tensors = tensor_tuple(value)
        if any(
            tensor.untyped_storage().data_ptr() != storage_base for tensor in tensors
        ):
            return False
        for base_tensor, tensor in zip(first_tensors, tensors):
            if tensor.data_ptr() != (
                base_tensor.data_ptr() + chunk_index * blob_stride_bytes
            ):
                return False

    # Ensure the explicit descriptors cannot address beyond the common slab.
    storage_nbytes = first.u.qdata.untyped_storage().nbytes()
    for qdata, rows in ((first.u.qdata, chunk_rows), (first.right.qdata, features)):
        end = (
            qdata.data_ptr()
            - storage_base
            + (len(chunk_factors) - 1) * blob_stride_bytes
            + rows * 64
        )
        if end > storage_nbytes:
            return False

    _set_triton_tma_allocator()
    # H100 32K-prefix sweep (J=8, C=4096, F=1024) selected 128x128 with
    # four warps: 0.04102 ms versus 0.05168 ms for 64x128/eight-warps.
    block_m = 128
    block_n = 128
    u_qdata_desc = TensorDescriptor(
        first.u.qdata,
        [len(chunk_factors), chunk_rows, 64],
        [blob_stride_bytes, 64, 1],
        [1, block_m, 64],
    )
    right_qdata_desc = TensorDescriptor(
        first.right.qdata,
        [len(chunk_factors), features, 64],
        [blob_stride_bytes, 64, 1],
        [1, block_n, 64],
    )
    grid = (
        triton.cdiv(chunk_rows, block_m),
        triton.cdiv(features, block_n),
        len(chunk_factors),
    )
    _fused_quantized_svd_restore_batch_tma_kernel[grid](
        u_qdata_desc,
        first.u.scale,
        first.sigma,
        right_qdata_desc,
        first.right.scale,
        destination_slots,
        output,
        chunk_rows,
        features,
        first_row_start,
        blob_stride_bytes // torch.float16.itemsize,
        first.u.scale.stride(0),
        first.u.scale.stride(1),
        first.right.scale.stride(0),
        first.right.scale.stride(1),
        output.stride(0),
        output.stride(1),
        U_ROW_SCALE=first.u.scale_mode == "row",
        R_ROW_SCALE=first.right.scale_mode == "row",
        U_GROUP_SIZE=first.u.group_size or first.rank,
        R_GROUP_SIZE=first.right.group_size or first.rank,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return True


def fused_reconstruct_svd_into(
    factors: QuantizedSVDFactors,
    destination_slots: torch.Tensor,
    destination: torch.Tensor,
    *,
    compute_mode: str = "fp16",
    use_tma: bool = False,
) -> None:
    """Reconstruct device-resident factors directly into paged KV slots.

    ``factors.u`` must already contain only the token rows being restored.
    ``destination`` accepts contiguous ``[slots, features]`` or
    ``[slots, heads, head_dim]`` storage.  Packed factors, slots, and output
    must reside on the same CUDA device.
    """

    if triton is None:
        raise RuntimeError("fused SVD restore requires Triton")
    if compute_mode not in SUPPORTED_SVD_RESTORE_COMPUTE:
        raise ValueError(
            f"compute_mode must be one of {SUPPORTED_SVD_RESTORE_COMPUTE}, "
            f"got {compute_mode!r}"
        )
    if destination.device.type != "cuda":
        raise ValueError("fused SVD restore requires a CUDA destination")
    fp8_compute = compute_mode == "fp8_e4m3"
    if fp8_compute:
        capability = torch.cuda.get_device_capability(destination.device)
        if torch.version.hip is not None or capability < (9, 0):
            raise RuntimeError(
                "fp8_e4m3 SVD restore requires NVIDIA compute capability 9.0+"
            )
    if destination.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused SVD restore destination must be FP16 or BF16")
    if destination.ndim not in (2, 3) or not destination.is_contiguous():
        raise ValueError(
            "destination must be contiguous [slots,features] or [slots,heads,dim]"
        )
    if destination_slots.ndim != 1 or destination_slots.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("destination_slots must be one-dimensional INT32/INT64")
    if destination_slots.numel() != factors.chunk_tokens:
        raise ValueError("destination_slots length must equal reconstructed rows")
    if destination_slots.device != destination.device:
        raise ValueError("destination_slots and destination must share a device")
    factor_tensors = (
        factors.u.qdata,
        factors.u.scale,
        factors.sigma,
        factors.right.qdata,
        factors.right.scale,
    )
    if any(tensor.device != destination.device for tensor in factor_tensors):
        raise ValueError("all SVD factors and destination must share a device")

    rows = factors.chunk_tokens
    features = factors.feature_dim
    output = destination.view(destination.shape[0], -1)
    if output.shape[1] != features:
        raise ValueError(
            f"destination feature width must be {features}, got {output.shape[1]}"
        )
    if factors.rank > 128:
        raise ValueError("fused SVD restore supports retained rank at most 128")

    if use_tma:
        _validate_tma_contract(
            factors,
            destination,
            compute_mode=compute_mode,
        )
        _set_triton_tma_allocator()
        # H100 sweep (4096x1024x128, BF16 destination) selected 64x128 with
        # eight warps: 0.01172 ms versus 0.01243 ms for 64x64/four-warps.
        block_m = 64
        block_n = 128
        # TensorDescriptor's host contract requires the innermost dimension
        # to be contiguous and every outer byte stride/base to be 16B aligned.
        u_qdata_desc = TensorDescriptor.from_tensor(
            factors.u.qdata,
            [block_m, 64],
        )
        right_qdata_desc = TensorDescriptor.from_tensor(
            factors.right.qdata,
            [block_n, 64],
        )
        grid = (triton.cdiv(rows, block_m), triton.cdiv(features, block_n))
        _fused_quantized_svd_restore_tma_kernel[grid](
            u_qdata_desc,
            factors.u.scale,
            factors.sigma,
            right_qdata_desc,
            factors.right.scale,
            destination_slots,
            output,
            rows,
            features,
            factors.u.scale.stride(0),
            factors.u.scale.stride(1),
            factors.right.scale.stride(0),
            factors.right.scale.stride(1),
            output.stride(0),
            output.stride(1),
            U_ROW_SCALE=factors.u.scale_mode == "row",
            R_ROW_SCALE=factors.right.scale_mode == "row",
            U_GROUP_SIZE=factors.u.group_size or factors.rank,
            R_GROUP_SIZE=factors.right.group_size or factors.rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
            num_stages=2,
        )
        return

    if (
        fp8_compute
        and factors.rank == 128
        and factors.u.bits == 4
        and factors.right.bits == 4
    ):
        # H100 sweep (4096x1024x128, BF16 destination) selected 64x128 with
        # four warps and two stages for the vectorized pointer-load path.
        block_m = 64
        block_n = 128
        grid = (triton.cdiv(rows, block_m), triton.cdiv(features, block_n))
        _fused_quantized_svd_restore_int4_rank128_fp8_kernel[grid](
            factors.u.qdata,
            factors.u.scale,
            factors.sigma,
            factors.right.qdata,
            factors.right.scale,
            destination_slots,
            output,
            rows,
            features,
            factors.u.qdata.stride(0),
            factors.u.qdata.stride(1),
            factors.u.scale.stride(0),
            factors.u.scale.stride(1),
            factors.right.qdata.stride(0),
            factors.right.qdata.stride(1),
            factors.right.scale.stride(0),
            factors.right.scale.stride(1),
            output.stride(0),
            output.stride(1),
            U_ROW_SCALE=factors.u.scale_mode == "row",
            R_ROW_SCALE=factors.right.scale_mode == "row",
            U_GROUP_SIZE=factors.u.group_size or factors.rank,
            R_GROUP_SIZE=factors.right.group_size or factors.rank,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
        return

    block_k = max(32 if fp8_compute else 16, triton.next_power_of_2(factors.rank))
    # M=64 is required for real Hopper FP8 WGMMA.  M=32 silently lowers the
    # E4M3 operands through an FP16 MMA path in Triton 3.7.
    block_m = 64 if fp8_compute or block_k <= 32 else 32
    block_n = 64
    grid = (triton.cdiv(rows, block_m), triton.cdiv(features, block_n))
    _fused_quantized_svd_restore_kernel[grid](
        factors.u.qdata,
        factors.u.scale,
        factors.sigma,
        factors.right.qdata,
        factors.right.scale,
        destination_slots,
        output,
        rows,
        features,
        factors.rank,
        factors.u.qdata.stride(0),
        factors.u.qdata.stride(1),
        factors.u.scale.stride(0),
        factors.u.scale.stride(1),
        factors.right.qdata.stride(0),
        factors.right.qdata.stride(1),
        factors.right.scale.stride(0),
        factors.right.scale.stride(1),
        output.stride(0),
        output.stride(1),
        BITS_U=factors.u.bits,
        BITS_R=factors.right.bits,
        U_ROW_SCALE=factors.u.scale_mode == "row",
        R_ROW_SCALE=factors.right.scale_mode == "row",
        U_GROUP_SIZE=factors.u.group_size or factors.rank,
        R_GROUP_SIZE=factors.right.group_size or factors.rank,
        FP8_COMPUTE=fp8_compute,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
