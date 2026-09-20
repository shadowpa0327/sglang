"""The complete timer must include codec preparation AND engine scatter."""
import time
from types import SimpleNamespace

import pytest
import torch

from kvcompress.api import CompressedPayload, KVCompressionPlugin
from kvcompress.request_store import RequestStore
from kvcompress.store import BlockStore
from sglang.srt.mem_cache.compression.linker import CompressionLinker
from sglang.srt.mem_cache.hicache_storage import PoolName


class SlowPreparation(KVCompressionPlugin):
    @property
    def configuration(self):
        return {"scheme": "test-delayed-prepare"}

    def compress(self, tensors, *, context):
        return CompressedPayload({name: value.clone() for name, value in tensors.items()}, {})

    def prepare_decompression(self, payload, *, out):
        time.sleep(0.025)

    def decompress(self, payload, *, out, scratch):
        for name in out:
            out[name].copy_(payload.tensors[name])


@pytest.mark.parametrize("mode", ["request", "block"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_complete_timer_includes_prepare_and_scatter(mode, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    plugin = SlowPreparation()
    tensors = {name: torch.ones((1, 1, 4, 1, 2), device=device) for name in ("key", "value")}
    context = {"page_size": 4, "layer_ids": [0]}
    store = (RequestStore(plugin, {}, store_bytes=4096) if mode == "request" else
             BlockStore(plugin, {}, block_pages=1, store_bytes=4096))
    store.insert("key", tensors, context=context)
    linker = CompressionLinker.__new__(CompressionLinker)
    published = {}

    def scatter(slots, block):
        time.sleep(0.025)
        published.update({name: tensor.clone() for name, tensor in block.items()})

    linker.adapter = SimpleNamespace(device=device, scatter_kv=scatter)
    linker.store = store
    linker.audit = None
    linker.audit_fault = None
    linker.block_pages = 1
    linker.block_tokens = 4
    events = []
    linker.record = events.append
    transfer = SimpleNamespace(name=PoolName.KV, keys=["key"], device_indices=torch.arange(4, device=device))
    if mode == "request":
        store.pin("key")
        linker.request_hits = {"r": "key"}
        linker.request_pages = {"key": 1}
        linker._restore_request("r", [transfer])
    else:
        linker._restore_blocks("r", [transfer])
    event = events[-1]
    assert event["event"] == "decompress"
    assert event["restore_timer"] == "synchronized_wall"
    assert event["restore_total_ms"] - event["latency_ms"] >= 40.0
    assert event["restored_bytes"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert all(torch.equal(published[name], tensor) for name, tensor in tensors.items())
