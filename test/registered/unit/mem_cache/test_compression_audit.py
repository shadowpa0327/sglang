from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.compression.audit import CompressionAudit
from sglang.srt.mem_cache.compression.linker import NativePoolAdapter
from kvcompress.loader import load_plugin
from kvcompress.store import BlockStore
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer


def fixture():
    torch.manual_seed(12)
    adapter = NativePoolAdapter.__new__(NativePoolAdapter)
    adapter.page_size, adapter.device = 2, torch.device("cpu")
    adapter.kv = SimpleNamespace(
        start_layer=0,
        k_buffer=[torch.randn(32, 1, 8).bfloat16() for _ in range(2)],
        v_buffer=[torch.randn(32, 1, 8).bfloat16() for _ in range(2)],
    )
    adapter.pool = adapter.kv
    adapter.req_pool = SimpleNamespace(
        req_to_token=torch.zeros(1, 32, dtype=torch.int64)
    )
    plugin, identity = load_plugin("int8")
    store = BlockStore(plugin, identity, 2)
    events = []
    linker = SimpleNamespace(
        adapter=adapter, record=events.append, store=store, compressed_only=True
    )
    audit = CompressionAudit(linker)
    slots = torch.arange(2, 6)
    native = adapter.gather_kv(slots)
    store.insert("b", native, context={"page_size": 2, "layer_ids": [0, 1]})
    audit.source(slots, native, "b")
    audit.poison_native(
        SimpleNamespace(
            disable=True, evictable_size=lambda: 0, protected_size=lambda: 0
        )
    )
    target = PoolTransfer(
        name=PoolName.KV, keys=["a", "b"], device_indices=torch.arange(8, 12)
    )
    request = SimpleNamespace(rid="r", kv=SimpleNamespace(req_pool_idx=0))
    adapter.req_pool.req_to_token[0, :4] = target.device_indices
    audit.bind(request, target.device_indices)
    restored = store.reconstruct(["b"])
    return audit, adapter, target, restored, events


def batch():
    return SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        forward_mode=SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True),
        extend_seq_lens=torch.tensor([2]),
        positions=torch.tensor([4, 5]),
        out_cache_loc=torch.tensor([12, 13]),
        input_ids=torch.tensor([7, 8]),
    )


def test_poisoned_source_roundtrip_scatter_reads_and_suffix_forward():
    audit, adapter, target, restored, events = fixture()
    assert torch.isnan(adapter.kv.k_buffer[0][2:6]).all()
    assert events[0]["payload_bytes_by_dtype"]["torch.int8"] > 0
    assert events[0]["changed_elements"] > 0
    adapter.scatter_kv(target.device_indices, restored.blocks[0])
    audit.verify_scatter("r", target, restored)
    audit.forward(batch())
    for name, buffers in (("key", adapter.kv.k_buffer), ("value", adapter.kv.v_buffer)):
        for layer, buffer in enumerate(buffers):
            audit.read(layer, name, buffer)
    audit.release("r")
    assert events[-1]["prefix_unchanged_after_decode"]
    assert events[-1]["prompt_forward_tokens"] == 2
    assert events[-1]["verified_kv_tensors"] == 4


def test_missing_scatter_is_rejected_even_after_decompress_was_called():
    audit, adapter, target, restored, _ = fixture()
    with pytest.raises(RuntimeError, match="scatter mismatch"):
        audit.verify_scatter("r", target, restored)


def test_cached_prefix_recomputation_is_rejected():
    audit, adapter, target, restored, _ = fixture()
    b = batch()
    b.positions[0] = 3
    with pytest.raises(RuntimeError, match="recomputed"):
        audit.forward(b)


def test_model_write_into_prefix_is_rejected():
    audit, adapter, target, restored, _ = fixture()
    with pytest.raises(RuntimeError, match="overwrite"):
        audit.write(target.device_indices[:1])


def test_wrong_attention_block_table_is_rejected():
    audit, adapter, target, restored, _ = fixture()
    adapter.req_pool.req_to_token[0, 0] = 31
    with pytest.raises(RuntimeError, match="block table"):
        audit.forward(batch())


def test_change_between_restore_and_attention_is_rejected():
    audit, adapter, target, restored, _ = fixture()
    adapter.scatter_kv(target.device_indices, restored.blocks[0])
    audit.verify_scatter("r", target, restored)
    adapter.kv.k_buffer[0][8].zero_()
    with pytest.raises(RuntimeError, match="changed before attention"):
        audit.read(0, "key", adapter.kv.k_buffer[0])
