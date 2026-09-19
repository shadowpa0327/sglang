"""Compressed KV storage behind SGLang's external-cache linker.

``compression_unit`` selects one of two independent paths:

``block``
    Compress completed fixed-size blocks incrementally. Lookup, eviction, and
    reconstruction are block-granular.

``request``
    Compress one complete restorable span when SGLang finalizes the request.
    The payload stays opaque and is reconstructed and evicted as one unit. Only
    an exact match of the complete original input can restore it.

Restores are synchronous: every queued request is reconstructed before its
forward batch can read KV.
"""

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.mem_cache.utils import get_hash_str

from kvcompress.api import canonical_json
from kvcompress.loader import load_plugin
from kvcompress.request_store import RequestStore, RequestStoreFull, RequestTooLarge
from kvcompress.store import BlockStore

KNOWN_SETTINGS = {
    "cache_mode",
    "plugin",
    "parameters",
    "block_pages",
    "metrics_path",
    "audit",
    "audit_fault",
    "store_bytes",
    "compression_unit",
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
            MLATokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.mem_cache.unified_memory_pool import (
            UnifiedMHATokenToKVPool,
            UnifiedMLATokenToKVPool,
        )

        self.page_size = params.page_size
        self.req_pool = params.req_to_token_pool
        self.allocator = params.token_to_kv_pool_allocator
        pool = self.allocator.get_kvcache()
        self.has_swa = isinstance(pool, SWAKVPool)
        self.hybrid = (
            isinstance(pool, HybridLinearKVPool)
            or getattr(self.req_pool, "mamba_pool", None) is not None
        )
        # `pool` is what the model runner holds; `kv` is its full-attention part.
        self.pool = pool
        self.kv = (
            pool.full_kv_pool
            if isinstance(pool, (HybridLinearKVPool, SWAKVPool))
            else pool
        )
        self.swa_kv = pool.swa_kv_pool if self.has_swa else None
        # MLA stores one joint latent per token instead of a key/value pair, so
        # it is a different tensor contract rather than a different layout. A
        # sliding window is never the MLA half of anything, so only the full
        # pool is allowed to be one.
        self.is_mla = isinstance(self.kv, MLATokenToKVPool)
        # Alternate page-major / packed / quantized layouts require adapters.
        for name, kv_pool in (("full", self.kv), ("swa", self.swa_kv)):
            if kv_pool is None:
                continue
            if name == "full" and self.is_mla:
                ordinary = type(kv_pool) is MLATokenToKVPool or isinstance(
                    kv_pool, UnifiedMLATokenToKVPool
                )
            else:
                ordinary = isinstance(kv_pool, MHATokenToKVPool) and (
                    type(kv_pool) is MHATokenToKVPool
                    or isinstance(kv_pool, UnifiedMHATokenToKVPool)
                )
            if (
                not ordinary
                or kv_pool.dtype not in (torch.float16, torch.bfloat16)
                # MLA pools carry no such flag; FP4/FP8 MLA is a subclass and
                # is already excluded by the identity check above.
                or getattr(kv_pool, "is_quantized_kv_cache", False)
            ):
                raise ValueError(
                    "Compression currently requires ordinary BF16/FP16 MHA or "
                    f"MLA storage for the {name} pool"
                )
        if self.has_swa:
            full = sorted(
                (local, global_id)
                for global_id, (local, is_swa) in pool.layers_mapping.items()
                if not is_swa
            )
            swa = sorted(
                (local, global_id)
                for global_id, (local, is_swa) in pool.layers_mapping.items()
                if is_swa
            )
            self.layer_ids = [global_id for _, global_id in full]
            self.swa_layer_ids = [global_id for _, global_id in swa]
        elif isinstance(pool, HybridLinearKVPool):
            self.layer_ids = list(pool.full_attention_layer_id_mapping)
            self.swa_layer_ids = []
        else:
            self.layer_ids = list(
                range(self.kv.start_layer, self.kv.start_layer + self.kv.layer_num)
            )
            self.swa_layer_ids = []
        self.sliding_window_size = (
            int(params.sliding_window_size) if self.has_swa else None
        )
        self.swa_window_pages = (
            math.ceil(self.sliding_window_size / self.page_size) if self.has_swa else 0
        )
        self.temporal_layer_ids = (
            list(self.req_pool.mamba_pool.mamba_layer_ids) if self.hybrid else []
        )
        self.temporal_kind = None
        self.device = self._kv_components()[0][1][0].device
        if self.hybrid and getattr(self.req_pool, "mamba_ckpt_pool", None) is not None:
            raise ValueError(
                "External compression requires native recurrent checkpoints; "
                "disable the engine int8 checkpoint pool"
            )
        if self.hybrid and any(
            getattr(self.req_pool.mamba_pool.mamba_cache, name, None) is not None
            for name in ("replayssm_d", "replayssm_k", "replayssm_g")
        ):
            raise ValueError(
                "ReplaySSM ring state needs a separate compression adapter"
            )

    def context(self):
        if self.is_mla:
            # One joint latent per token per layer, heads folded to 1. The
            # leading kv_lora_rank columns carry no rotation and the trailing
            # qk_rope_head_dim ones are already rotated, so there is no single
            # pre-RoPE space to invert into: the latent is published as stored.
            specs = {
                "latent": {
                    "component": "full_attention",
                    "layout": "pages,layers,tokens,heads,dim",
                    "layer_ids": self.layer_ids,
                    "extent": "span",
                    "dtype": str(self.kv.dtype).removeprefix("torch."),
                    "key_space": "post_rope",
                    "kv_lora_rank": int(self.kv.kv_lora_rank),
                    "qk_rope_head_dim": int(self.kv.qk_rope_head_dim),
                },
            }
        else:
            specs = {
                "key": {
                    "component": "full_attention",
                    "layout": "pages,layers,tokens,heads,dim",
                    "layer_ids": self.layer_ids,
                    "extent": "span",
                    "dtype": str(self.kv.k_buffer[0].dtype).removeprefix("torch."),
                    "key_space": "pre_rope",
                },
                "value": {
                    "component": "full_attention",
                    "layout": "pages,layers,tokens,heads,dim",
                    "layer_ids": self.layer_ids,
                    "extent": "span",
                    "dtype": str(self.kv.v_buffer[0].dtype).removeprefix("torch."),
                },
            }
        if self.has_swa:
            specs.update(
                {
                    "swa_key": {
                        "component": "sliding_attention",
                        "layout": "pages,layers,tokens,heads,dim",
                        "layer_ids": self.swa_layer_ids,
                        "extent": "window",
                        "dtype": str(self.swa_kv.k_buffer[0].dtype).removeprefix(
                            "torch."
                        ),
                        "window_tokens": self.sliding_window_size,
                        "stored_window_tokens": self.swa_window_pages * self.page_size,
                        "key_space": "pre_rope",
                    },
                    "swa_value": {
                        "component": "sliding_attention",
                        "layout": "pages,layers,tokens,heads,dim",
                        "layer_ids": self.swa_layer_ids,
                        "extent": "window",
                        "dtype": str(self.swa_kv.v_buffer[0].dtype).removeprefix(
                            "torch."
                        ),
                        "window_tokens": self.sliding_window_size,
                        "stored_window_tokens": self.swa_window_pages * self.page_size,
                    },
                }
            )
        if self.hybrid:
            temporal_kind = getattr(self, "temporal_kind", None) or "recurrent"
            temporal_layer_ids = getattr(self, "temporal_layer_ids", None)
            if temporal_layer_ids is None:
                temporal_layer_ids = list(
                    range(self._state_buffers()["temporal"].shape[0])
                )
            specs["temporal"] = {
                "component": "recurrent",
                "layout": (
                    "layers,value_heads,value_dim,key_dim"
                    if temporal_kind in {"gdn", "kda", "lightning"}
                    else "layers,heads,head_dim,state_dim"
                ),
                "layer_ids": temporal_layer_ids,
                "extent": "endpoint",
                "kind": temporal_kind,
                "dtype": str(
                    self._state_buffers()["temporal"].dtype
                ).removeprefix("torch."),
            }
        return {
            "layout": "pages,layers,tokens,heads,dim",
            "layer_ids": self.layer_ids,
            "page_size": self.page_size,
            "key_space": "post_rope" if self.is_mla else "pre_rope",
            "tensor_specs": specs,
        }

    def slots(self, req, start, end):
        row = self.req_pool.req_to_token[req.kv.req_pool_idx, start:end]
        return row.to(device=self.device, dtype=torch.int64)

    def _full_buffer_slots(self, slots):
        """Translate request-row ids when unified memory uses virtual slots."""
        if getattr(self.kv, "kv_cache_layout", None) == "page_major":
            return self.allocator.translate_kv_loc_for_kernel(slots)
        return slots

    @staticmethod
    def _gather_buffers(buffers, slots, *, pages, page_size):
        return torch.stack(
            [
                buffer.index_select(0, slots).reshape(
                    pages, page_size, *buffer.shape[1:]
                )
                for buffer in buffers
            ],
            dim=1,
        )

    def _kv_components(self):
        """The full-attention tensors this pool exposes, as (name, buffers).

        MHA has two half-width buffers per layer; MLA has one joint latent
        buffer whose trailing ``qk_rope_head_dim`` columns are the rotated part.
        Both present as [tokens, heads, dim], so everything downstream of here
        reads the same five-axis block either way.
        """
        if not self.is_mla:
            return (("key", self.kv.k_buffer), ("value", self.kv.v_buffer))
        buffers = self.kv.kv_buffer
        if self.kv.store_dtype != self.kv.dtype:
            buffers = [buffer.view(self.kv.dtype) for buffer in buffers]
        return (("latent", buffers),)

    def gather_kv(self, slots):
        """[pages, layers, page_tokens, heads, dim] per full-attention tensor."""
        if slots.numel() % self.page_size:
            raise ValueError("Slots are not a sequence of complete pages")
        pages = slots.numel() // self.page_size
        slots = self._full_buffer_slots(slots)
        return {
            name: self._gather_buffers(
                buffers, slots, pages=pages, page_size=self.page_size
            )
            for name, buffers in self._kv_components()
        }

    def scatter_kv(self, slots, block):
        slots = self._full_buffer_slots(slots)
        for name, buffers in self._kv_components():
            for layer, buffer in enumerate(buffers):
                rows = block[name][:, layer].reshape(-1, *buffer.shape[1:])
                buffer.index_copy_(0, slots, rows)

    def gather_swa(self, full_slots):
        """Gather a live trailing window addressed by its full-pool slots."""
        if not self.has_swa:
            return {}
        if full_slots.numel() % self.page_size:
            raise ValueError("SWA slots are not a sequence of complete pages")
        slots = self.pool.translate_loc_from_full_to_swa(full_slots)
        if bool((slots <= 0).any().item()):
            raise RuntimeError("Trailing SWA window contains an unmapped cache page")
        pages = full_slots.numel() // self.page_size
        return {
            name: self._gather_buffers(
                buffers, slots, pages=pages, page_size=self.page_size
            )
            for name, buffers in (
                ("swa_key", self.swa_kv.k_buffer),
                ("swa_value", self.swa_kv.v_buffer),
            )
        }

    def _new_swa(self, factory):
        if not self.has_swa:
            return {}
        pages = self.swa_window_pages
        return {
            name: factory(
                (pages, len(buffers), self.page_size, *buffers[0].shape[1:]),
                dtype=buffers[0].dtype,
                device=buffers[0].device,
            )
            for name, buffers in (
                ("swa_key", self.swa_kv.k_buffer),
                ("swa_value", self.swa_kv.v_buffer),
            )
        }

    def empty_swa(self):
        """One fixed-shape placeholder for an absent block checkpoint."""
        return self._new_swa(torch.zeros)

    def sample_swa(self):
        """One allocation-safe random layout probe (no preceding zero tensor)."""
        return self._new_swa(torch.randn)

    def pad_swa(self, tensors):
        """Left-align a short early-prefix window in the fixed block schema."""
        if not tensors or tensors["swa_key"].shape[0] == self.swa_window_pages:
            return tensors
        padded = self.empty_swa()
        pages = tensors["swa_key"].shape[0]
        for name, tensor in tensors.items():
            padded[name][:pages].copy_(tensor)
        return padded

    def scatter_swa(self, slots, block, *, indices_from_full=False):
        # Static PoolName.SWA transfers already carry SWA-physical ids. Unified
        # transfers deliberately reuse FULL virtual ids and mark that fact; only
        # those ids need the full->SWA kernel-space translation here.
        if indices_from_full:
            slots = self.allocator.translate_loc_from_full_to_swa(slots)
        for name, buffers in (
            ("swa_key", self.swa_kv.k_buffer),
            ("swa_value", self.swa_kv.v_buffer),
        ):
            pages = slots.numel() // self.page_size
            tensor = block[name][:pages]
            for layer, buffer in enumerate(buffers):
                rows = tensor[:, layer].reshape(-1, *buffer.shape[1:])
                buffer.index_copy_(0, slots, rows)

    def _state_buffers(self):
        state = self.req_pool.mamba_pool.mamba_cache
        return {
            "temporal": state.temporal,
            **{f"conv_{i}": t for i, t in enumerate(state.conv)},
        }

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

    def split_state(self, state):
        if state is None:
            return None, None
        return state["temporal"], {
            name: tensor for name, tensor in state.items() if name != "temporal"
        }

    def empty_temporal(self):
        """One transient zero checkpoint with the plugin-visible shape."""
        temporal = self._state_buffers()["temporal"]
        return torch.zeros(
            (temporal.shape[0], *temporal.shape[2:]),
            dtype=temporal.dtype,
            device=temporal.device,
        )

    def scatter_state(self, slot, state):
        phys = self._physical_state_slot(slot)
        for name, buffer in self._state_buffers().items():
            buffer.index_copy_(1, phys, state[name].unsqueeze(1).to(buffer.dtype))


class CompressionLinker(UnifiedCacheLinker):
    @staticmethod
    def _rope_selector(model_config, layer_ids, head_dim):
        """Resolve per-component RoPE parameters for mixed FULL/SWA models."""
        config = model_config.hf_text_config
        layer_types = getattr(config, "layer_types", None)
        rope_parameters = getattr(config, "rope_parameters", None)
        if not layer_types or not isinstance(rope_parameters, dict):
            return None
        kinds = {layer_types[layer_id] for layer_id in layer_ids}
        if len(kinds) != 1:
            return None
        parameters = rope_parameters.get(next(iter(kinds)))
        if not isinstance(parameters, dict):
            return None
        rope_type = parameters.get("rope_type", parameters.get("type", "default"))
        selector = {"base": parameters.get("rope_theta", 10000.0)}
        if rope_type == "proportional":
            selector["class"] = "Gemma4RotaryEmbedding"
            # Gemma4 records the partial width in ``rope_angles`` but exposes
            # a padded, cross-mixed table whose public rotary_dim is head_dim.
            selector["rotary_dim"] = int(head_dim)
        elif rope_type == "default":
            selector["class"] = "RotaryEmbedding"
            selector["rotary_dim"] = int(
                head_dim * parameters.get("partial_rotary_factor", 1.0)
            )
        return selector

    def __init__(self, server_args, params, *, components):
        if not set(components) <= {
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.MAMBA,
        }:
            raise ValueError(
                "Compression substrate supports full attention, sliding-window "
                "attention, and hybrid recurrent models"
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
        if "scope" in config:
            raise ValueError(
                "scope is no longer supported; use compression_unit=block or request"
            )
        unknown = set(config) - KNOWN_SETTINGS
        if unknown:
            raise ValueError(f"Unknown prefix compression settings: {sorted(unknown)}")
        self.adapter = NativePoolAdapter(params)
        self.page_size = params.page_size
        self.compression_unit = config.get("compression_unit", "block")
        if self.compression_unit not in {"block", "request"}:
            raise ValueError("compression_unit must be block or request")
        if self.compression_unit == "request" and "block_pages" in config:
            raise ValueError("block_pages only applies to compression_unit=block")
        self.block_pages = (
            config.get("block_pages", 16) if self.compression_unit == "block" else None
        )
        if self.block_pages is not None and (
            type(self.block_pages) is not int or self.block_pages < 1
        ):
            raise ValueError("block_pages must be a positive page count")
        self.block_tokens = (
            self.block_pages * self.page_size if self.block_pages is not None else None
        )
        chunk = server_args.chunked_prefill_size
        if self.compression_unit == "block":
            self.restoration_boundary_tokens = (
                math.lcm(self.block_tokens, chunk)
                if chunk is not None and chunk > 0
                else self.block_tokens
            )
        else:
            self.restoration_boundary_tokens = (
                math.lcm(self.page_size, chunk)
                if (self.adapter.hybrid or self.adapter.has_swa)
                and chunk is not None
                and chunk > 0
                else self.page_size
            )
        # Stateful request mode needs callbacks at exact recurrent/SWA
        # checkpoints; plain full-attention request mode does no unfinished work.
        self.prefill_boundary_tokens = (
            self.restoration_boundary_tokens
            if self.compression_unit == "block"
            or self.adapter.hybrid
            or self.adapter.has_swa
            else None
        )
        plugin, description = load_plugin(
            config.get("plugin", "identity"), config.get("parameters")
        )
        accounting_root = plugin
        identity = {
            "model_path": server_args.model_path,
            "revision": server_args.revision,
            "model_override_args": server_args.json_model_override_args,
            "dtype": str(self.adapter.kv.dtype),
            "page_size": self.page_size,
            "block_pages": self.block_pages,
            "restoration_boundary_tokens": self.restoration_boundary_tokens,
            "plugin": description,
        }
        # Store is process-local and reset on weight changes. Retain the resolved
        # model interpretation as well as the requested model/version for audit.
        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(server_args)
        model_json = model_config.hf_config.to_json_string()
        identity["model_config_sha256"] = hashlib.sha256(
            model_json.encode()
        ).hexdigest()
        identity["model_config"] = json.loads(model_json)
        if self.adapter.hybrid:
            from sglang.srt.configs.hybrid_arch import (
                hybrid_gdn_config,
                hybrid_lightning_config,
                kimi_linear_config,
            )

            if hybrid_gdn_config(model_config) is not None:
                self.adapter.temporal_kind = "gdn"
            elif kimi_linear_config(model_config) is not None:
                self.adapter.temporal_kind = "kda"
            elif hybrid_lightning_config(model_config) is not None:
                self.adapter.temporal_kind = "lightning"
            else:
                self.adapter.temporal_kind = "mamba2"

        from sglang.srt.layers.rotary_embedding import factory
        from .pre_rope import NativeRoPEDecodePlugin, PreRoPETransform

        # Compression has one canonical key space. Invert the model's own
        # rotary table at the unit boundary; a model with no RoPE uses identity.
        text_config = model_config.hf_text_config
        # An empty rope cache is the stronger witness: Nemotron-H carries a vestigial
        # rope_theta that no layer reads, so the config test alone sends it to a selector
        # that finds zero tables and refuses to start. The model is loaded before
        # init_memory_pools builds this linker, so the cache is populated by now.
        allow_no_rope = not factory._ROPE_DICT or not any(
            getattr(text_config, name, None)
            for name in ("rope_parameters", "rope_scaling", "rope_theta")
        )
        # MLA rotates only the trailing qk_rope_head_dim columns of a latent
        # that is otherwise not a key at all, so there is nothing to invert
        # here: the contract already says the latent is published post-RoPE.
        if self.adapter.is_mla:
            self.pre_rope = None
        else:
            full_selector = self._rope_selector(
                model_config, self.adapter.layer_ids, self.adapter.kv.head_dim
            )
            self.pre_rope = PreRoPETransform.from_model_ropes(
                self.adapter,
                record=self.record,
                allow_no_rope=allow_no_rope,
                component="full",
                selector=full_selector,
            )
        self.swa_pre_rope = None
        if self.adapter.has_swa:
            swa_selector = self._rope_selector(
                model_config,
                self.adapter.swa_layer_ids,
                self.adapter.swa_kv.head_dim,
            )
            self.swa_pre_rope = PreRoPETransform.from_model_ropes(
                self.adapter,
                record=self.record,
                allow_no_rope=allow_no_rope,
                component="swa",
                selector=swa_selector,
            )
        identity["key_space"] = "post_rope" if self.adapter.is_mla else "pre_rope"
        transforms = {
            name: transform
            for name, transform in (
                ("key", self.pre_rope),
                ("swa_key", self.swa_pre_rope),
            )
            if transform is not None
        }
        if transforms:
            plugin = NativeRoPEDecodePlugin(plugin, transforms)
            ropes = {
                name: transform.table.description
                for name, transform in transforms.items()
            }
            identity["pre_rope_adapter"] = {
                "version": 3,
                "source_sha256": hashlib.sha256(
                    Path(__file__).with_name("pre_rope.py").read_bytes()
                ).hexdigest(),
                "method": "fp32-inverse-rotation-from-model-rope-table",
                "rope": ropes.get("key"),
                "ropes": ropes,
            }
        else:
            identity["pre_rope_adapter"] = {
                "version": 3,
                "method": "identity-no-rope",
            }
        identity["compression_unit"] = self.compression_unit
        self.cache_mode = config.get("cache_mode", "compressed")
        if self.cache_mode not in {"none", "native", "compressed"}:
            raise ValueError("cache_mode must be none, native or compressed")
        self.writes_enabled = self.cache_mode == "compressed"
        self.compressed_only = self.cache_mode == "compressed"
        # Native mode still relies on the tree's ordinary SWA recovery. Only
        # compressed-only loads restore SWA from this backend.
        self.restores_swa = self.adapter.has_swa and self.compressed_only
        store_bytes = config.get("store_bytes", 4 << 30)
        self.store = (
            BlockStore(
                plugin,
                identity,
                self.block_pages,
                store_bytes=store_bytes,
                accounting_root=accounting_root,
            )
            if self.compression_unit == "block"
            else RequestStore(
                plugin,
                identity,
                store_bytes=store_bytes,
                accounting_root=accounting_root,
            )
        )
        self.layer_done_counter = CompletedBatchCounter()
        if self.adapter.hybrid:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.queued = {}
        self.completed_loads = deque()
        # Block mode: rid -> (hashed pages, chained hash of the last page).
        self.progress = {}
        # Request mode: rid -> (restorable token count, recurrent checkpoint).
        self.checkpoints = {}
        # Request mode: exact key -> restorable page count, and rid -> matched key.
        self.request_pages = {}
        self.request_hits = {}
        self.metrics_path = config.get("metrics_path")
        self.events = deque(maxlen=10000)
        self.request_events = {}
        self.audit = None
        self.audit_fault = config.get("audit_fault")
        if self.compression_unit == "request" and config.get("audit"):
            raise ValueError(
                "audit=true is not supported with compression_unit=request"
            )
        if self.audit_fault not in (None, "skip_scatter") or (
            self.audit_fault and not config.get("audit")
        ):
            raise ValueError("audit_fault requires audit=True and must be skip_scatter")
        if config.get("audit"):
            if not server_args.disable_cuda_graph:
                raise ValueError("Byte-level audit requires disabled CUDA graphs")
            if self.adapter.is_mla:
                # The audit hooks set_kv_buffer per k/v buffer pair; MLA writes
                # one joint latent and would read half the expected tensors.
                raise ValueError("Byte-level audit does not cover MLA pools yet")
            from .audit import CompressionAudit

            self.audit = CompressionAudit(self)
        pool = self._initialize_store()
        self.record(
            {
                "event": "init",
                "identity": identity,
                "configuration": config,
                "store": pool,
            }
        )

    def _initialize_store(self):
        if self.compression_unit == "block" and self.writes_enabled:
            return self._allocate_store()
        return self.store.description()

    def _add_temporal(
        self,
        tensors,
        context,
        state,
        *,
        checkpoint_position,
        placeholder=None,
    ):
        """Expose temporal to the plugin and retain convolution state losslessly."""
        if not self.adapter.hybrid:
            return tensors, context, None
        temporal, conv = self.adapter.split_state(state)
        valid = temporal is not None
        if temporal is None:
            temporal = placeholder
            if temporal is None:
                temporal = self.adapter.empty_temporal()
        tensors["temporal"] = temporal
        context["temporal_valid"] = valid
        context["tensor_specs"]["temporal"]["checkpoint_position"] = (
            checkpoint_position if valid else None
        )
        return tensors, context, conv

    def _add_swa(
        self,
        tensors,
        context,
        swa,
        *,
        checkpoint_position,
        start_position=None,
        fixed_layout=False,
        placeholder=None,
    ):
        """Expose a trailing SWA checkpoint, padding only block-store entries."""
        if not self.adapter.has_swa:
            return tensors, context
        valid = swa is not None
        valid_pages = swa["swa_key"].shape[0] if valid else 0
        if swa is None:
            swa = placeholder if placeholder is not None else self.adapter.empty_swa()
        elif fixed_layout:
            swa = self.adapter.pad_swa(swa)
        tensors.update(swa)
        if start_position is None:
            start_position = max(
                0,
                checkpoint_position - self.adapter.swa_window_pages * self.page_size,
            )
        context["swa_valid"] = valid
        for name in ("swa_key", "swa_value"):
            context["tensor_specs"][name].update(
                {
                    "checkpoint_position": checkpoint_position if valid else None,
                    "start_position": start_position,
                    "valid_pages": valid_pages,
                }
            )
        return tensors, context

    def _gather_swa_checkpoint(self, req, end):
        """Copy the live, page-aligned SWA window ending at ``end``."""
        start = max(0, end - self.adapter.swa_window_pages * self.adapter.page_size)
        full_slots = self.adapter.slots(req, start, end)
        tensors = self.adapter.gather_swa(full_slots)
        if getattr(self, "swa_pre_rope", None) is not None:
            native = tensors.pop("swa_key")
            tensors["swa_key"] = self.swa_pre_rope.derotate(
                native, start_position=start, audit=self.audit is not None
            )
        return tensors, start

    @torch.no_grad()
    def _allocate_store(self):
        """Reserve the pool from one synthetic block; codecs emit a fixed layout."""
        if self.compression_unit != "block":
            raise RuntimeError("Only block compression preallocates a fixed layout")
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
        if self.adapter.has_swa:
            sample, context = self._add_swa(
                sample,
                context,
                None,
                checkpoint_position=0,
                start_position=0,
                fixed_layout=True,
                placeholder=self.adapter.sample_swa(),
            )
        state = self.adapter.gather_state(0) if self.adapter.hybrid else None
        sample, context, conv = self._add_temporal(
            sample, context, state, checkpoint_position=0
        )
        # The first call is a layout probe, never a restorable checkpoint.
        context["temporal_valid"] = False
        context["tensor_specs"].get("temporal", {}).update(
            {"checkpoint_position": None}
        )
        return self.store.allocate(sample, context=context, state=conv)

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

    # ---- source side: gather from live request slots, then compress ----

    @torch.no_grad()
    def on_request_progress(self, req, processed_tokens, *, finished):
        """Observe live KV progress and persist according to the configured unit."""
        if self.compression_unit == "request":
            self._on_request_progress(req, processed_tokens, finished=finished)
            return
        self._on_block_progress(req, processed_tokens, finished=finished)

    def _on_block_progress(self, req, processed_tokens, *, finished):
        """Compress each newly completed block independently."""
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
        checkpoint_index = (
            complete - 1
            if (self.adapter.hybrid or self.adapter.has_swa)
            and state_valid
            and processed == complete * self.block_tokens
            else None
        )
        checkpoint = (
            self.adapter.gather_state(req.kv.mamba_pool_idx)
            if checkpoint_index is not None and self.adapter.hybrid
            else None
        )
        start = done * self.block_tokens
        hashes = get_hash_str(
            req.origin_input_ids[start : complete * self.block_tokens],
            last_hash,
            page_size=self.page_size,
        )
        missing = []
        for index in range(done, complete):
            offset = (index - done + 1) * self.block_pages
            key = hashes[offset - 1]
            parent = (
                hashes[offset - self.block_pages - 1] if index > done else last_hash
            )
            state = checkpoint if index == checkpoint_index else None
            swa_valid = self.adapter.has_swa and index == checkpoint_index
            block = (index, key, parent, state, swa_valid)
            if self.store.has(key):
                needs_upgrade = (
                    state is not None and not self.store.has_checkpoint(key)
                ) or (swa_valid and not self.store.has_swa(key))
                if needs_upgrade:
                    begin = time.perf_counter_ns()
                    inputs, context, _, _ = self._gather_block(
                        req, index, swa_checkpoint=swa_valid
                    )
                    inputs, context, conv = self._add_temporal(
                        inputs,
                        context,
                        state,
                        checkpoint_position=(index + 1) * self.block_tokens,
                    )
                    record = self.store.upgrade(
                        key, inputs, context=context, state=conv
                    )
                    if self.adapter.device.type == "cuda":
                        torch.cuda.current_stream(self.adapter.device).synchronize()
                    self.record(
                        {
                            "event": "state_checkpoint",
                            "rid": req.rid,
                            "block": index,
                            **record,
                            "latency_ms": (time.perf_counter_ns() - begin) / 1e6,
                            "recompressed": True,
                        }
                    )
            else:
                missing.append(block)
        self._gather_and_commit_blocks(req, missing)
        if not finished:
            self.progress[req.rid] = (complete * self.block_pages, hashes[-1])

    @torch.no_grad()
    def _on_request_progress(self, req, processed_tokens, *, finished):
        """Keep stateful checkpoints during prefill; encode only at finalization."""
        if not self._request_is_keyable(req):
            self.checkpoints.pop(req.rid, None)
            if finished:
                self.record(
                    {
                        "event": "compress_skipped",
                        "rid": req.rid,
                        "compression_unit": "request",
                        "reason": "custom_input_embeddings",
                    }
                )
            return
        if not finished:
            self._hold_request_checkpoint(req, processed_tokens)
            return
        held = self.checkpoints.pop(req.rid, None)
        if not self.writes_enabled or req.kv.req_pool_idx is None:
            return
        key = self._request_key(req)
        if self.store.has(key):
            return

        max_end = self._request_cacheable_end(req, processed_tokens)
        if self.adapter.hybrid or self.adapter.has_swa:
            if held is not None and held[0] <= max_end:
                end, state, swa, swa_start = held
            elif not req.output_ids and max_end == processed_tokens:
                end = max_end
                state = (
                    self.adapter.gather_state(req.kv.mamba_pool_idx)
                    if self.adapter.hybrid
                    else None
                )
                swa, swa_start = (
                    self._gather_swa_checkpoint(req, end)
                    if self.adapter.has_swa
                    else (None, None)
                )
            else:
                self.record(
                    {
                        "event": "compress_skipped",
                        "rid": req.rid,
                        "compression_unit": "request",
                        "reason": (
                            "no_restorable_recurrent_checkpoint"
                            if self.adapter.hybrid
                            else "no_restorable_swa_checkpoint"
                        ),
                    }
                )
                return
        else:
            end, state, swa, swa_start = max_end, None, None, None
        if end <= 0:
            return

        begin = time.perf_counter_ns()
        tensors, context, _, _ = self._gather_span(req, 0, end)
        tensors, context = self._add_swa(
            tensors,
            context,
            swa,
            checkpoint_position=end,
            start_position=swa_start,
        )
        tensors, context, conv = self._add_temporal(
            tensors, context, state, checkpoint_position=end
        )
        try:
            record = self.store.insert(key, tensors, context=context, state=conv)
        except (RequestTooLarge, RequestStoreFull) as error:
            self.record(
                {
                    "event": "compress_rejected",
                    "rid": req.rid,
                    "key": key,
                    "compression_unit": "request",
                    "restorable_tokens": end,
                    "reason": str(error),
                }
            )
            return
        if self.adapter.device.type == "cuda":
            torch.cuda.current_stream(self.adapter.device).synchronize()
        latency = (time.perf_counter_ns() - begin) / 1e6
        event_record = dict(record)
        for evicted in event_record.pop("evicted"):
            evicted_pages = self.request_pages.pop(evicted, 0)
            self.record(
                {
                    "event": "evict",
                    "rid": req.rid,
                    "key": evicted,
                    "compression_unit": "request",
                    "tokens": evicted_pages * self.page_size,
                    "stored_requests": len(self.store.requests),
                }
            )
        self.request_pages[key] = record["pages"]
        self.record(
            {
                "event": "compress",
                "rid": req.rid,
                "key": key,
                "compression_unit": "request",
                "restorable_tokens": end,
                **event_record,
                "latency_ms": latency,
                "timer": "wall_sync",
            }
        )

    @torch.no_grad()
    def _hold_request_checkpoint(self, req, processed_tokens):
        """Retain the deepest exact recurrent/SWA boundary seen before decode."""
        if (
            not (self.adapter.hybrid or self.adapter.has_swa)
            or not self.writes_enabled
            or req.kv.req_pool_idx is None
            or req.output_ids
        ):
            return
        processed = min(processed_tokens, len(req.origin_input_ids))
        if (
            processed <= 0
            or processed % self.page_size
            or processed > self._request_cacheable_end(req, processed_tokens)
        ):
            return
        state = (
            self.adapter.gather_state(req.kv.mamba_pool_idx)
            if self.adapter.hybrid
            else None
        )
        swa, swa_start = (
            self._gather_swa_checkpoint(req, processed)
            if self.adapter.has_swa
            else (None, None)
        )
        self.checkpoints[req.rid] = (processed, state, swa, swa_start)

    def _request_cacheable_end(self, req, processed_tokens):
        """Largest page-aligned prefix an ordinary admission can restore."""
        processed = min(processed_tokens, len(req.origin_input_ids))
        # SGLang always recomputes at least the final input token for logits.
        limit = min(processed, max(len(req.origin_input_ids) - 1, 0))
        return limit // self.page_size * self.page_size

    @staticmethod
    def _request_key(req):
        """Exact complete-input identity, including SGLang's cache namespace."""
        encoded = canonical_json(
            {
                "version": 1,
                "token_ids": list(req.origin_input_ids),
                "extra_key": getattr(req, "extra_key", None),
                "cache_salt": getattr(req, "cache_salt", None),
            }
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _request_is_keyable(req):
        """Token identity is insufficient when callers replace token embeddings."""
        return (
            getattr(req, "input_embeds", None) is None
            and getattr(req, "positional_embed_overrides", None) is None
        )

    def _context(self, req, start, end, *, block_index=None):
        context = {
            **self.adapter.context(),
            "token_ids": list(req.origin_input_ids[start:end]),
            "start_position": start,
        }
        if block_index is not None:
            context["block_index"] = block_index
        return context

    def _gather_span(self, req, start, end, *, sample_blocks=()):
        """Gather one logical request span, regardless of its physical KV slots."""
        slots = self.adapter.slots(req, start, end)
        native = self.adapter.gather_kv(slots)
        start_page = start // self.page_size
        samples = {}
        for index in sample_blocks:
            first_page = index * self.block_pages - start_page
            samples[index] = {
                name: tensor[first_page : first_page + 1].clone()
                for name, tensor in native.items()
            }
        if self.pre_rope is not None:
            keys = native.pop("key")
            native["key"] = self.pre_rope.derotate(
                keys, start_position=start, audit=self.audit is not None
            )
            del keys
        return native, self._context(req, start, end), slots, samples

    def _gather_block(self, req, index, *, swa_checkpoint=False, swa_placeholder=None):
        """Gather one block for the independent block-compression path."""
        start, end = index * self.block_tokens, (index + 1) * self.block_tokens
        native, context, slots, samples = self._gather_span(
            req,
            start,
            end,
            sample_blocks=(index,) if self.audit is not None else (),
        )
        context["block_index"] = index
        swa, swa_start = (
            self._gather_swa_checkpoint(req, end) if swa_checkpoint else (None, None)
        )
        native, context = self._add_swa(
            native,
            context,
            swa,
            checkpoint_position=end,
            start_position=swa_start,
            fixed_layout=True,
            placeholder=swa_placeholder,
        )
        return native, context, slots, samples.get(index)

    def _record_compression(self, req, encoded, latency, *, blocks_encoded):
        """Publish records for independently compressed blocks."""
        for (index, key, _, _, _), record in encoded:
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
            self.record(
                {
                    "event": "compress",
                    "rid": req.rid,
                    "block": index,
                    **record,
                    "blocks_encoded": blocks_encoded,
                    "latency_ms": latency / len(encoded),
                    "timer": "wall_sync",
                }
            )

    def _gather_and_commit_blocks(self, req, blocks):
        """Gather and compress each missing block independently."""
        if not blocks:
            return
        begin = time.perf_counter_ns()
        swa_placeholder = (
            self.adapter.empty_swa()
            if self.adapter.has_swa and any(not block[-1] for block in blocks)
            else None
        )
        gathered = [
            self._gather_block(
                req,
                index,
                swa_checkpoint=swa_valid,
                swa_placeholder=swa_placeholder,
            )
            for index, _, _, _, swa_valid in blocks
        ]
        placeholder = (
            self.adapter.empty_temporal()
            if self.adapter.hybrid
            and any(state is None for _, _, _, state, _ in blocks)
            else None
        )
        prepared = []
        for (index, _, _, state, _), (inputs, context, slots, sample) in zip(
            blocks, gathered
        ):
            inputs, context, conv = self._add_temporal(
                inputs,
                context,
                state,
                checkpoint_position=(index + 1) * self.block_tokens,
                placeholder=placeholder,
            )
            prepared.append((inputs, context, slots, sample, conv))
        records = [
            self.store.insert(key, inputs, context=context, state=conv, parent=parent)
            for (_, key, parent, _, _), (inputs, context, _, _, conv) in zip(
                blocks, prepared
            )
        ]
        if self.adapter.device.type == "cuda":
            torch.cuda.current_stream(self.adapter.device).synchronize()
        latency = (time.perf_counter_ns() - begin) / 1e6
        self._record_compression(
            req, list(zip(blocks, records)), latency, blocks_encoded=1
        )
        if self.audit is not None:
            for (_, key, _, _, _), (_, _, slots, sample, _) in zip(blocks, prepared):
                self.audit.source(slots, sample, key)

    # ---- target side: exact lookup and synchronous restore ----

    def lookup(self, rid, transfers):
        """Block-mode lookup retained for generic linker compatibility."""
        if self.compression_unit != "block":
            raise RuntimeError("Request compression requires request-aware lookup")
        full = next(t for t in transfers if t.name == PoolName.KV)
        # Page 0 of the transfer is the first uncached page; blocks are aligned
        # to the prompt start, so a misaligned native hit simply finds nothing.
        result = []
        for end in range(self.block_pages, len(full.keys) + 1, self.block_pages):
            key = full.keys[end - 1]
            if not self.store.has(key):
                break
            has_state = not self.adapter.hybrid or self.store.has_checkpoint(key)
            has_swa = not self.adapter.has_swa or self.store.has_swa(key)
            if has_state and has_swa:
                result.append(end)
        self.record(
            {"event": "lookup", "rid": rid, "hit_pages": result[-1] if result else 0}
        )
        return result

    def lookup_request(self, req, transfers, *, device_hit_pages=0):
        if self.compression_unit == "block":
            return self.lookup(req.rid, transfers)
        full = next(t for t in transfers if t.name == PoolName.KV)
        key = self._request_key(req)
        if device_hit_pages:
            self._release_request_hit(req.rid)
            self.record(
                {
                    "event": "lookup",
                    "rid": req.rid,
                    "key": key,
                    "compression_unit": "request",
                    "exact_request_hit": False,
                    "hit_pages": 0,
                    "hit_tokens": 0,
                    "reason": "native_prefix_present",
                }
            )
            return []
        if not self._request_is_keyable(req):
            self._release_request_hit(req.rid)
            self.record(
                {
                    "event": "lookup",
                    "rid": req.rid,
                    "key": key,
                    "compression_unit": "request",
                    "exact_request_hit": False,
                    "hit_pages": 0,
                    "hit_tokens": 0,
                    "reason": "custom_input_embeddings",
                }
            )
            return []
        pages = self.request_pages.get(key, 0)
        previous = self.request_hits.get(req.rid)
        eligible = pages > 0 and pages <= len(full.keys)
        if not eligible:
            self._release_request_hit(req.rid)
            hit = False
        elif previous == key:
            hit = True
        else:
            self._release_request_hit(req.rid)
            hit = self.store.pin(key)
            if hit:
                self.request_hits[req.rid] = key
        if hit and self.adapter.hybrid and not self.store.has_checkpoint(key):
            self._release_request_hit(req.rid)
            hit = False
        if hit and self.adapter.has_swa and not self.store.has_swa(key):
            self._release_request_hit(req.rid)
            hit = False
        self.record(
            {
                "event": "lookup",
                "rid": req.rid,
                "key": key,
                "compression_unit": "request",
                "exact_request_hit": hit,
                "hit_pages": pages if hit else 0,
                "hit_tokens": pages * self.page_size if hit else 0,
            }
        )
        return [pages] if hit else []

    def offload(self, transfers):
        # Tree write-through is not a compression trigger; nothing is queued.
        return False

    def load(self, rid, transfers):
        if rid in self.queued:
            raise ValueError(f"Duplicate queued load: {rid}")
        if self.compression_unit == "request" and rid not in self.request_hits:
            raise ValueError(f"No exact request hit recorded for rid={rid!r}")
        self.queued[rid] = transfers
        return True

    @torch.no_grad()
    def start_layer_wise_loading(self):
        if not self.queued:
            return -1
        rids = list(self.queued)
        for rid, transfers in self.queued.items():
            if self.compression_unit == "request":
                self._restore_request(rid, transfers)
            else:
                self._restore_blocks(rid, transfers)
        if self.adapter.device.type == "cuda":
            torch.cuda.current_stream(self.adapter.device).synchronize()
        self.layer_done_counter.producer_index += 1
        self.completed_loads.append(rids)
        self.queued.clear()
        self.record(
            {"event": "load_batch", "request_ids": rids, "batch_size": len(rids)}
        )
        return self.layer_done_counter.producer_index

    def _restore_blocks(self, rid, transfers):
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
                if transfer.name == PoolName.SWA:
                    if not self.store.has_swa(block_keys[-1]):
                        raise RuntimeError(
                            "Block has no sliding-window state for an SWA restore"
                        )
                    self.adapter.scatter_swa(
                        transfer.device_indices,
                        restored.blocks[-1],
                        indices_from_full=transfer.indices_from_pool == PoolName.KV,
                    )
                if transfer.name == PoolName.MAMBA:
                    conv = self.store.state(block_keys[-1])
                    temporal = restored.blocks[-1].get("temporal")
                    if (
                        not self.store.has_checkpoint(block_keys[-1])
                        or temporal is None
                    ):
                        raise RuntimeError(
                            "Block has no recurrent state for a hybrid restore"
                        )
                    self.adapter.scatter_state(
                        transfer.device_indices[0],
                        {"temporal": temporal, **(conv or {})},
                    )
        if self.audit is not None:
            self.audit.verify_scatter(rid, kv, restored)

    def _restore_request(self, rid, transfers):
        key = self.request_hits.pop(rid)
        try:
            kv = next(t for t in transfers if t.name == PoolName.KV)
            expected_pages = self.request_pages[key]
            if len(kv.keys) != expected_pages:
                raise ValueError(
                    "Request restore destination does not cover the stored span"
                )
            restored = self.store.reconstruct(key)
            self.record(
                {
                    "event": "decompress",
                    "rid": rid,
                    "key": key,
                    "compression_unit": "request",
                    **restored.measurement,
                }
            )
            slots = kv.device_indices.to(device=self.adapter.device, dtype=torch.int64)
            if slots.numel() != restored.measurement["tokens"]:
                raise ValueError(
                    "Request restore slot count does not match decoded payload"
                )
            self.adapter.scatter_kv(slots, restored.tensors)
            for transfer in transfers:
                if transfer.name == PoolName.SWA:
                    if not self.store.has_swa(key):
                        raise RuntimeError(
                            "Request has no sliding-window state for an SWA restore"
                        )
                    self.adapter.scatter_swa(
                        transfer.device_indices,
                        restored.tensors,
                        indices_from_full=transfer.indices_from_pool == PoolName.KV,
                    )
                if transfer.name == PoolName.MAMBA:
                    conv = self.store.state(key)
                    temporal = restored.tensors.get("temporal")
                    if not self.store.has_checkpoint(key) or temporal is None:
                        raise RuntimeError(
                            "Request has no recurrent state for a hybrid restore"
                        )
                    self.adapter.scatter_state(
                        transfer.device_indices[0],
                        {"temporal": temporal, **(conv or {})},
                    )
        finally:
            self.store.unpin(key)

    def _release_request_hit(self, rid):
        hits = getattr(self, "request_hits", None)
        if hits is None:
            return
        key = hits.pop(rid, None)
        if key is not None:
            self.store.unpin(key)

    def cancel_queued_load(self, rid):
        if self.compressed_only and rid in self.queued:
            del self.queued[rid]
            self._release_request_hit(rid)
            return True
        self._release_request_hit(rid)
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
        self.checkpoints.clear()
        self.request_pages.clear()
        self.request_hits.clear()
        self.layer_done_counter.reset()
        self.store.reset()
        self.request_events.clear()
        self.writes_enabled = self.cache_mode == "compressed"
        self.compressed_only = self.cache_mode == "compressed"
        self.record({"event": "reset"})

    def close(self):
        self.reset()
