"""Block-granular compressed prefix store behind the external cache linker.

Compression follows the request, not the radix tree. Each time a prefill chunk
lands, every newly completed block of `block_pages` pages is gathered from the
request's own slots, encoded by the plugin, and stored with the raw recurrent
state at that boundary (hybrid models). Lookups and restores move whole blocks;
the incomplete tail of a prompt is recomputed. Restores are synchronous: every
queued request is restored before its forward batch can read KV.
"""

import hashlib
import json
import logging
from collections import deque
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.mem_cache.utils import get_hash_str

from kvcompress.api import canonical_json
from kvcompress.loader import load_plugin
from kvcompress.store import BlockStore

logger = logging.getLogger(__name__)

KNOWN_SETTINGS = {
    "cache_mode",
    "plugin",
    "parameters",
    "block_pages",
    "metrics_path",
    "audit",
    "audit_fault",
    "key_space",
    "store_bytes",
}


class CompletedBatchCounter:
    """Loads finish synchronously, including device scatter, before publication."""

    def __init__(self):
        self.producer_index = -1
        self.consumer_index = -1

    def set_consumer(self, index):
        if index > self.producer_index:
            raise RuntimeError("Attempted to consume an unfinished compression load")
        self.consumer_index = index

    def wait_until(self, threshold):
        pass

    def reset(self):
        self.producer_index = self.consumer_index = -1


class NativePoolAdapter:
    """Gather/scatter between a request's native slots and plugin-shaped tensors."""

    def __init__(self, params):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
        )

        self.page_size = params.page_size
        self.req_pool = params.req_to_token_pool
        kv = params.token_to_kv_pool_allocator.get_kvcache()
        self.hybrid = isinstance(kv, HybridLinearKVPool)
        # `pool` is what the model runner holds; `kv` is its full-attention part.
        self.pool = kv
        self.kv = kv.full_kv_pool if self.hybrid else kv
        # Alternate page-major / packed / quantized layouts require adapters.
        if (
            type(self.kv) is not MHATokenToKVPool
            or self.kv.dtype not in (torch.float16, torch.bfloat16)
            or self.kv.is_quantized_kv_cache
        ):
            raise ValueError(
                "Compression currently requires ordinary BF16/FP16 MHATokenToKVPool"
            )
        self.layer_ids = (
            list(kv.full_attention_layer_id_mapping)
            if self.hybrid
            else list(
                range(self.kv.start_layer, self.kv.start_layer + self.kv.layer_num)
            )
        )
        self.device = self.kv.k_buffer[0].device
        if self.hybrid and getattr(self.req_pool, "mamba_ckpt_pool", None) is not None:
            raise ValueError("Compression stores raw recurrent state; disable int8 checkpoints")
        if self.hybrid and any(
            getattr(self.req_pool.mamba_pool.mamba_cache, name, None) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            raise ValueError("ReplaySSM ring state needs a separate compression adapter")

    def context(self):
        return {
            "layout": "pages,layers,tokens,heads,dim",
            "layer_ids": self.layer_ids,
            "page_size": self.page_size,
            "key_space": "post_rope",
        }

    def slots(self, req, start, end):
        row = self.req_pool.req_to_token[req.kv.req_pool_idx, start:end]
        return row.to(device=self.device, dtype=torch.int64)

    def gather_kv(self, slots):
        """[pages, layers, page_tokens, heads, dim] for key and value."""
        if slots.numel() % self.page_size:
            raise ValueError("Slots are not a sequence of complete pages")
        pages = slots.numel() // self.page_size
        return {
            name: torch.stack(
                [
                    b.index_select(0, slots).reshape(pages, self.page_size, *b.shape[1:])
                    for b in buffers
                ],
                dim=1,
            )
            for name, buffers in (("key", self.kv.k_buffer), ("value", self.kv.v_buffer))
        }

    def scatter_kv(self, slots, block):
        for name, buffers in (("key", self.kv.k_buffer), ("value", self.kv.v_buffer)):
            for layer, buffer in enumerate(buffers):
                rows = block[name][:, layer].reshape(-1, *buffer.shape[1:])
                buffer.index_copy_(0, slots, rows)

    def _state_buffers(self):
        state = self.req_pool.mamba_pool.mamba_cache
        return {"temporal": state.temporal, **{f"conv_{i}": t for i, t in enumerate(state.conv)}}

    def _physical_state_slot(self, slot):
        virtual = torch.as_tensor(slot, device=self.device).reshape(1)
        return self.req_pool.translate_mamba_indices(virtual).long()

    def gather_state(self, slot):
        """Recurrent and convolution state of one slot: [layers, ...] per tensor."""
        phys = self._physical_state_slot(slot)
        return {
            name: tensor.index_select(1, phys).squeeze(1)
            for name, tensor in self._state_buffers().items()
        }

    def scatter_state(self, slot, state):
        phys = self._physical_state_slot(slot)
        for name, buffer in self._state_buffers().items():
            buffer.index_copy_(1, phys, state[name].unsqueeze(1).to(buffer.dtype))


class CompressionLinker(UnifiedCacheLinker):
    def __init__(self, server_args, params, *, components):
        if not set(components) <= {ComponentType.FULL, ComponentType.MAMBA}:
            raise ValueError(
                "Compression substrate supports full attention and hybrid recurrent models"
            )
        if server_args.tp_size != 1 or params.pp_size != 1 or params.attn_cp_size != 1:
            raise ValueError("Compression experiments currently use one GPU")
        if (
            getattr(server_args, "enable_lora", False)
            or params.is_eagle
            or getattr(server_args, "speculative_algorithm", None)
        ):
            raise ValueError(
                "LoRA and speculative decoding need separate compression identity adapters"
            )
        config = json.loads(server_args.prefix_compression_config or "{}")
        unknown = set(config) - KNOWN_SETTINGS
        if unknown:
            raise ValueError(f"Unknown prefix compression settings: {sorted(unknown)}")
        self.adapter = NativePoolAdapter(params)
        self.page_size = params.page_size
        self.block_pages = config.get("block_pages", 16)
        if type(self.block_pages) is not int or self.block_pages < 1:
            raise ValueError("block_pages must be a positive page count")
        self.block_tokens = self.block_pages * self.page_size
        # The scheduler caps every prefill chunk at the next block end, so chunk
        # ends coincide with block ends for every request of a batch.
        self.prefill_boundary_tokens = self.block_tokens
        chunk = server_args.chunked_prefill_size
        if self.adapter.hybrid and chunk is not None and chunk > 0:
            # The recurrent state is only observed at chunk ends, so every block
            # end must coincide with one for its checkpoint to exist.
            if self.block_tokens % chunk:
                raise ValueError(
                    "Hybrid models need block_pages * page_size to be a multiple of "
                    f"chunked_prefill_size ({self.block_tokens} vs {chunk})"
                )
        plugin, description = load_plugin(
            config.get("plugin", "identity"), config.get("parameters")
        )
        identity = {
            "model_path": server_args.model_path,
            "revision": server_args.revision,
            "model_override_args": server_args.json_model_override_args,
            "dtype": str(self.adapter.kv.dtype),
            "page_size": self.page_size,
            "block_pages": self.block_pages,
            "plugin": description,
        }
        # Store is process-local and reset on weight changes. Retain the resolved
        # model interpretation as well as the requested model/version for audit.
        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(server_args)
        model_json = model_config.hf_config.to_json_string()
        identity["model_config_sha256"] = hashlib.sha256(model_json.encode()).hexdigest()
        identity["model_config"] = json.loads(model_json)
        self.pre_rope = None
        from .pre_rope import resolve_key_space

        key_space, key_space_source = resolve_key_space(
            config.get("key_space"), plugin.key_space
        )
        identity["key_space"] = key_space
        identity["key_space_source"] = key_space_source
        identity["plugin_key_space"] = plugin.key_space
        if key_space_source == "config_override":
            logger.warning(
                "prefix_compression_config key_space=%s overrides the plugin's %s",
                key_space,
                plugin.key_space,
            )
        if key_space == "pre_rope":
            from .pre_rope import NativeRoPEDecodePlugin, PreRoPETransform

            # Invert the model's own rotary table at the block boundary. No
            # forward hook, no shadow buffers, CUDA graphs stay enabled.
            self.pre_rope = PreRoPETransform.from_model_ropes(
                self.adapter, record=self.record
            )
            plugin = NativeRoPEDecodePlugin(plugin, self.pre_rope)
            identity["pre_rope_adapter"] = {
                "version": 2,
                "source_sha256": hashlib.sha256(
                    Path(__file__).with_name("pre_rope.py").read_bytes()
                ).hexdigest(),
                "method": "fp32-inverse-rotation-from-model-rope-table",
                "rope": self.pre_rope.table.description,
            }
        self.store = BlockStore(
            plugin,
            identity,
            self.block_pages,
            store_bytes=config.get("store_bytes", 4 << 30),
        )
        self.layer_done_counter = CompletedBatchCounter()
        if self.adapter.hybrid:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.queued = {}
        self.completed_loads = deque()
        # rid -> (hashed prompt pages, chained hash of the last hashed page)
        self.progress = {}
        self.cache_mode = config.get("cache_mode", "compressed")
        if self.cache_mode not in {"none", "native", "compressed"}:
            raise ValueError("cache_mode must be none, native or compressed")
        self.writes_enabled = self.cache_mode == "compressed"
        self.compressed_only = self.cache_mode == "compressed"
        self.metrics_path = config.get("metrics_path")
        self.events = deque(maxlen=10000)
        self.request_events = {}
        self.audit = None
        self.audit_fault = config.get("audit_fault")
        if self.audit_fault not in (None, "skip_scatter") or (
            self.audit_fault and not config.get("audit")
        ):
            raise ValueError("audit_fault requires audit=True and must be skip_scatter")
        if config.get("audit"):
            if not server_args.disable_cuda_graph:
                raise ValueError("Byte-level audit requires disabled CUDA graphs")
            from .audit import CompressionAudit

            self.audit = CompressionAudit(self)
        pool = self._allocate_store()
        self.record(
            {"event": "init", "identity": identity, "configuration": config, "store": pool}
        )

    @torch.no_grad()
    def _allocate_store(self):
        """Reserve the pool from one synthetic block; codecs emit a fixed layout."""
        slots = torch.arange(self.block_tokens, device=self.adapter.device)
        sample = {
            name: torch.randn(t.shape, dtype=t.dtype, device=t.device)
            for name, t in self.adapter.gather_kv(slots).items()
        }
        context = {
            **self.adapter.context(),
            "token_ids": [0] * self.block_tokens,
            "start_position": 0,
            "block_index": 0,
        }
        if self.pre_rope is not None:
            context["key_space"] = "pre_rope"
        state = self.adapter.gather_state(0) if self.adapter.hybrid else None
        return self.store.allocate(sample, context=context, state=state)

    def record(self, event):
        self.events.append(event)
        rids = [event["rid"]] if "rid" in event else event.get("request_ids", [])
        for rid in rids:
            self.request_events.setdefault(rid, []).append(event)
        if self.metrics_path:
            path = Path(self.metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(canonical_json(event) + "\n")

    # ---- source side: compress blocks as the request's prefill completes them ----

    @torch.no_grad()
    def on_request_progress(self, req, processed_tokens, *, finished):
        """Store every newly completed block of this request's prompt.

        `processed_tokens` prompt tokens have KV in the request's slots. For a
        hybrid model the live recurrent state matches that position only before
        the first decode step, so blocks completed later are left to the tail.
        """
        hashed_pages, last_hash = (
            self.progress.pop(req.rid, (0, None))
            if finished
            else self.progress.get(req.rid, (0, None))
        )
        if not self.writes_enabled or req.kv.req_pool_idx is None:
            return
        processed = min(processed_tokens, len(req.origin_input_ids))
        complete = processed // self.block_tokens
        done = hashed_pages // self.block_pages
        if complete <= done:
            return
        state_valid = len(req.output_ids) <= 1
        start = done * self.block_tokens
        hashes = get_hash_str(
            req.origin_input_ids[start : complete * self.block_tokens],
            last_hash,
            page_size=self.page_size,
        )
        for index in range(done, complete):
            offset = (index - done + 1) * self.block_pages
            key = hashes[offset - 1]
            parent = hashes[offset - self.block_pages - 1] if index > done else last_hash
            end = (index + 1) * self.block_tokens
            if self.store.has(key):
                continue
            if self.adapter.hybrid and (not state_valid or end != processed):
                self.record({"event": "block_skipped_no_state", "rid": req.rid, "block": index})
                continue
            self._compress_block(req, index, key, parent)
        if not finished:
            self.progress[req.rid] = (complete * self.block_pages, hashes[-1])

    def _compress_block(self, req, index, key, parent):
        start, end = index * self.block_tokens, (index + 1) * self.block_tokens
        slots = self.adapter.slots(req, start, end)
        native = self.adapter.gather_kv(slots)
        # The audit compares one page; keep only that page so the post-RoPE keys
        # can be released once de-rotated.
        sample = (
            {name: t[:1].clone() for name, t in native.items()}
            if self.audit is not None
            else None
        )
        context = {
            **self.adapter.context(),
            "token_ids": list(req.origin_input_ids[start:end]),
            "start_position": start,
            "block_index": index,
        }
        inputs = native
        if self.pre_rope is not None:
            context["key_space"] = "pre_rope"
            keys = native.pop("key")
            inputs = {
                **native,
                "key": self.pre_rope.derotate(
                    keys, start_position=start, audit=self.audit is not None
                ),
            }
            del keys
        state = (
            self.adapter.gather_state(req.kv.mamba_pool_idx) if self.adapter.hybrid else None
        )
        record = self.store.insert(
            key, inputs, context=context, state=state, parent=parent
        )
        if self.adapter.device.type == "cuda":
            torch.cuda.current_stream(self.adapter.device).synchronize()
        for evicted in record.pop("evicted"):
            self.record(
                {
                    "event": "evict",
                    "rid": req.rid,
                    "key": evicted,
                    "tokens": self.block_tokens,
                    "stored_blocks": len(self.store.blocks),
                }
            )
        self.record({"event": "compress", "rid": req.rid, "block": index, **record})
        if self.audit is not None:
            self.audit.source(slots, sample, key)

    # ---- target side: whole-block lookup and synchronous restore ----

    def lookup(self, rid, transfers):
        full = next(t for t in transfers if t.name == PoolName.KV)
        # Page 0 of the transfer is the first uncached page; blocks are aligned
        # to the prompt start, so a misaligned native hit simply finds nothing.
        result = []
        for end in range(self.block_pages, len(full.keys) + 1, self.block_pages):
            if not self.store.has(full.keys[end - 1]):
                break
            result.append(end)
        self.record({"event": "lookup", "rid": rid, "hit_pages": result[-1] if result else 0})
        return result

    def offload(self, transfers):
        # Tree write-through is not a compression trigger; nothing is queued.
        return False

    def load(self, rid, transfers):
        if rid in self.queued:
            raise ValueError(f"Duplicate queued load: {rid}")
        self.queued[rid] = transfers
        return True

    @torch.no_grad()
    def start_layer_wise_loading(self):
        if not self.queued:
            return -1
        rids = list(self.queued)
        for rid, transfers in self.queued.items():
            kv = next(t for t in transfers if t.name == PoolName.KV)
            if len(kv.keys) % self.block_pages:
                raise ValueError("Restore request is not a sequence of whole blocks")
            block_keys = kv.keys[self.block_pages - 1 :: self.block_pages]
            restored = self.store.reconstruct(block_keys)
            self.record({"event": "decompress", "rid": rid, **restored.measurement})
            if self.audit_fault != "skip_scatter":
                slots = kv.device_indices.to(device=self.adapter.device, dtype=torch.int64)
                for i, block in enumerate(restored.blocks):
                    self.adapter.scatter_kv(
                        slots[i * self.block_tokens : (i + 1) * self.block_tokens], block
                    )
                for transfer in transfers:
                    if transfer.name == PoolName.MAMBA:
                        state = self.store.state(block_keys[-1])
                        if state is None:
                            raise RuntimeError("Block has no recurrent state for a hybrid restore")
                        self.adapter.scatter_state(transfer.device_indices[0], state)
            if self.audit is not None:
                self.audit.verify_scatter(rid, kv, restored)
        if self.adapter.device.type == "cuda":
            torch.cuda.current_stream(self.adapter.device).synchronize()
        self.layer_done_counter.producer_index += 1
        self.completed_loads.append(rids)
        self.queued.clear()
        self.record({"event": "load_batch", "request_ids": rids, "batch_size": len(rids)})
        return self.layer_done_counter.producer_index

    def cancel_queued_load(self, rid):
        if self.compressed_only and rid in self.queued:
            del self.queued[rid]
            return True
        # The wrapper has already published destinations into the radix tree.
        # Finish those loads even if a request is cancelled, to keep them valid.
        return False

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        return self.completed_loads.popleft()

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise RuntimeError("Compression never queues tree offloads")

    def reset(self):
        if self.audit is not None:
            self.audit.reset()
        self.queued.clear()
        self.completed_loads.clear()
        self.progress.clear()
        self.layer_done_counter.reset()
        self.store.reset()
        self.request_events.clear()
        self.writes_enabled = self.cache_mode == "compressed"
        self.compressed_only = self.cache_mode == "compressed"
        self.record({"event": "reset"})

    def close(self):
        self.reset()
