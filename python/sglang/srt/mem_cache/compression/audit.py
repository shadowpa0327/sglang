"""Expensive opt-in provenance assertions, outside compression benchmark timers."""

import hashlib

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


class CompressionAudit:
    def __init__(self, linker):
        self.linker = linker
        self.adapter = linker.adapter
        self.requests = {}
        self.protected = torch.zeros(
            self.adapter.kv.k_buffer[0].shape[0],
            dtype=torch.bool,
            device=self.adapter.device,
        )
        self.source_slots = torch.zeros_like(self.protected)
        # Buffer reads are hooked on the attention pool, forwards on the pool the
        # model runner owns; for hybrid models these are different objects.
        self.adapter.kv.compression_audit = self
        self.adapter.pool.compression_audit = self

    def reset(self):
        self.requests.clear()
        self.protected.zero_()
        self.source_slots.zero_()

    @torch.no_grad()
    def poison_native(self, cache):
        assert cache.disable and self.linker.compressed_only
        assert cache.evictable_size() == cache.protected_size() == 0
        slots = self.source_slots.nonzero().flatten()
        for buffers in (self.adapter.kv.k_buffer, self.adapter.kv.v_buffer):
            for buffer in buffers:
                buffer.index_fill_(0, slots, float("nan"))
        self.linker.record(
            {
                "event": "audit_native_poisoned",
                "native_tree_tokens": 0,
                "source_slots_poisoned": len(slots),
                "layers": len(self.adapter.kv.k_buffer),
                "poison": "NaN",
            }
        )

    def bind(self, req, indices):
        assert req.rid not in self.requests
        assert not self.protected[indices.long()].any().item(), (
            "Private destinations overlap"
        )
        self.protected[indices.long()] = True
        self.requests[req.rid] = {
            "req": req,
            "indices": indices.clone().long(),
            "hashes": {},
            "read": set(),
            "forwards": 0,
            "prompt_forward_tokens": 0,
        }

    def release(self, rid):
        item = self.requests.pop(rid, None)
        if item is None:
            return
        self.protected[item["indices"]] = False
        if item["forwards"]:
            assert len(item["read"]) == len(self.adapter.kv.k_buffer) * 2, (
                "Not every KV layer was verified at attention read"
            )
            for name, buffers in (
                ("key", self.adapter.kv.k_buffer),
                ("value", self.adapter.kv.v_buffer),
            ):
                for layer, buffer in enumerate(buffers):
                    actual = tensor_sha256(buffer.index_select(0, item["indices"]))
                    if actual != item["hashes"][(layer, name)]:
                        raise RuntimeError(
                            "Compression provenance: restored prefix changed during inference"
                        )
            self.linker.record(
                {
                    "event": "audit_request_complete",
                    "rid": rid,
                    "prefix_unchanged_after_decode": True,
                    "restored_tokens": len(item["indices"]),
                    "prompt_forward_tokens": item["prompt_forward_tokens"],
                    "model_forwards": item["forwards"],
                    "verified_kv_tensors": len(item["read"]),
                }
            )

    @torch.no_grad()
    def source(self, slots, native, key):
        """Round-trip the block's first page across every layer against `native`."""
        self.source_slots[slots.long()] = True
        block = self.linker.store.blocks[key]
        dtype_bytes = {}
        for tensor in block.payload.tensors.values():
            name = str(tensor.dtype)
            dtype_bytes[name] = (
                dtype_bytes.get(name, 0) + tensor.numel() * tensor.element_size()
            )
        # This additional source-side decode is audit work, not load timing.
        reconstructed = self.linker.store.reconstruct([key]).blocks[0]
        changed, maximum = 0, 0.0
        for name, original in native.items():
            error = (original[0].float() - reconstructed[name][0].float()).abs()
            changed += int(torch.count_nonzero(error).item())
            maximum = max(maximum, float(error.max().item()))
        self.linker.record(
            {
                "event": "audit_encoding",
                "payload_bytes_by_dtype": dtype_bytes,
                "objects": [key],
                "sample_tokens": self.adapter.page_size,
                "changed_elements": changed,
                "max_abs_roundtrip_error": maximum,
            }
        )

    @torch.no_grad()
    def verify_scatter(self, rid, transfer, restored):
        if transfer.name != PoolName.KV:
            return
        item = self.requests[rid]
        assert torch.equal(item["indices"], transfer.device_indices.long())
        for name, buffers in (
            ("key", self.adapter.kv.k_buffer),
            ("value", self.adapter.kv.v_buffer),
        ):
            for layer, buffer in enumerate(buffers):
                expected = restored.pages(name, layer)
                actual = buffer.index_select(0, item["indices"])
                if not torch.equal(actual, expected):
                    raise RuntimeError(
                        f"Compression provenance: scatter mismatch for {rid} {name} layer {layer}"
                    )
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError(
                        "Compression provenance: nonfinite reconstructed KV"
                    )
                item["hashes"][(layer, name)] = tensor_sha256(expected)
        self.linker.record(
            {
                "event": "audit_scatter_exact",
                "rid": rid,
                "tokens": len(item["indices"]),
                "verified_kv_tensors": len(item["hashes"]),
            }
        )

    @torch.no_grad()
    def read(self, layer, name, buffer):
        key = (layer - self.adapter.kv.start_layer, name)
        for rid, item in self.requests.items():
            if key in item["read"]:
                continue
            expected = item["hashes"].get(key)
            if expected is None:
                raise RuntimeError(
                    "Compression provenance: attention read before verified restoration"
                )
            actual = tensor_sha256(buffer.index_select(0, item["indices"]))
            if actual != expected:
                raise RuntimeError(
                    "Compression provenance: KV changed before attention read"
                )
            item["read"].add(key)
            self.linker.record(
                {
                    "event": "audit_attention_read",
                    "rid": rid,
                    "layer": key[0],
                    "tensor": name,
                    "sha256": actual,
                    "matches_decompressor": True,
                }
            )

    @torch.no_grad()
    def write(self, locations):
        if self.requests and self.protected[locations.long()].any().item():
            raise RuntimeError(
                "Compression provenance: model attempted to overwrite restored prefix KV"
            )

    @torch.no_grad()
    def forward(self, batch):
        if not self.requests:
            return
        row_indices = batch.req_pool_indices.cpu().tolist()
        if batch.forward_mode.is_decode():
            lengths = [1] * len(row_indices)
        elif batch.forward_mode.is_extend():
            lengths = batch.extend_seq_lens.cpu().tolist()
        else:
            raise RuntimeError(
                "Compression provenance audit requires ordinary prefill/decode"
            )
        by_row = {
            int(item["req"].kv.req_pool_idx): (rid, item)
            for rid, item in self.requests.items()
            if item["req"].kv.req_pool_idx is not None
        }
        offset = 0
        for row, length in zip(row_indices, lengths):
            if row in by_row:
                rid, item = by_row[row]
                prefix = len(item["indices"])
                mapped = self.adapter.req_pool.req_to_token[row, :prefix].long()
                if not torch.equal(mapped, item["indices"]):
                    raise RuntimeError(
                        "Compression provenance: attention block table does not reference restored KV"
                    )
                positions = batch.positions[offset : offset + length]
                if positions.numel() != length or (positions < prefix).any().item():
                    raise RuntimeError(
                        "Compression provenance: model recomputed a cached prefix position"
                    )
                writes = batch.out_cache_loc[offset : offset + length]
                self.write(writes)
                item["forwards"] += 1
                if batch.forward_mode.is_extend():
                    item["prompt_forward_tokens"] += length
                self.linker.record(
                    {
                        "event": "audit_model_forward",
                        "batch_size": len(row_indices),
                        "rid": rid,
                        "mode": "decode"
                        if batch.forward_mode.is_decode()
                        else "extend",
                        "tokens": length,
                        "restored_prefix_tokens": prefix,
                        "min_position": int(positions.min().item()),
                        "max_position": int(positions.max().item()),
                        "prefix_tokens_recomputed": 0,
                        "prefix_write_overlap": 0,
                        "restored_table_matches": True,
                    }
                )
            offset += length
        assert offset == batch.input_ids.numel(), (
            "Audit did not cover all model input tokens"
        )
