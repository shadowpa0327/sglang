from collections import deque
from types import SimpleNamespace

import pytest
import torch

from kvcompress.loader import load_plugin
from kvcompress.store import BlockStore
from sglang.srt.mem_cache.compression.linker import (
    CompletedBatchCounter,
    CompressionLinker,
    NativePoolAdapter,
)
IdentityPlugin = type(load_plugin("identity")[0])
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.utils import get_hash_str

PAGE, BLOCK_PAGES = 2, 2
BLOCK = PAGE * BLOCK_PAGES


def make_adapter(hybrid=False, slots=64):
    torch.manual_seed(3)
    adapter = NativePoolAdapter.__new__(NativePoolAdapter)
    adapter.page_size, adapter.device, adapter.hybrid = PAGE, torch.device("cpu"), hybrid
    adapter.layer_ids = [0, 1]
    adapter.kv = SimpleNamespace(
        start_layer=0,
        k_buffer=[torch.randn(slots, 2, 4).bfloat16() for _ in range(2)],
        v_buffer=[torch.randn(slots, 2, 4).bfloat16() for _ in range(2)],
    )
    adapter.req_pool = SimpleNamespace(req_to_token=torch.zeros(4, slots, dtype=torch.int64))
    if hybrid:
        adapter.req_pool.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.randn(3, 8, 5).bfloat16(),
                conv=[torch.randn(3, 8, 6).bfloat16()],
            )
        )
        adapter.req_pool.translate_mamba_indices = lambda x: x
    return adapter


def make_linker(adapter, plugin=None, store_bytes=1 << 20, *, preallocate=True):
    linker = CompressionLinker.__new__(CompressionLinker)
    linker.adapter = adapter
    linker.page_size, linker.block_pages, linker.block_tokens = PAGE, BLOCK_PAGES, BLOCK
    linker.store = BlockStore(
        plugin or IdentityPlugin(), {}, BLOCK_PAGES, store_bytes=store_bytes
    )
    linker.pre_rope = linker.audit = linker.audit_fault = None
    linker.progress, linker.queued = {}, {}
    linker.completed_loads = deque()
    linker.layer_done_counter = CompletedBatchCounter()
    linker.writes_enabled, linker.compressed_only = True, True
    linker.cache_mode = "compressed"
    linker.request_events = {}
    linker.events = []
    linker.record = linker.events.append
    if preallocate:
        sample = adapter.gather_kv(torch.arange(BLOCK))
        state = adapter.gather_state(0) if adapter.hybrid else None
        linker.store.allocate(sample, context=adapter.context(), state=state)
    return linker


def make_req(adapter, rid, tokens, row, first_slot, mamba_slot=None):
    adapter.req_pool.req_to_token[row, : len(tokens)] = torch.arange(
        first_slot, first_slot + len(tokens)
    )
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(tokens),
        output_ids=[],
        kv=SimpleNamespace(req_pool_idx=row, mamba_pool_idx=mamba_slot),
    )


def block_keys(tokens, blocks):
    return get_hash_str(tokens[: blocks * BLOCK], None, page_size=PAGE)[
        BLOCK_PAGES - 1 :: BLOCK_PAGES
    ]


def test_blocks_complete_chunk_by_chunk_and_tail_is_never_stored():
    adapter = make_adapter()
    linker = make_linker(adapter)
    req = make_req(adapter, "a", range(100, 110), row=0, first_slot=10)
    for processed in (3, 4, 7, 8, 10):
        linker.on_request_progress(req, processed, finished=False)
    stored = [e["key"] for e in linker.events if e["event"] == "compress"]
    assert stored == block_keys(req.origin_input_ids, 2)
    assert linker.progress["a"] == (4, block_keys(req.origin_input_ids, 2)[-1])
    linker.on_request_progress(req, 10, finished=True)
    assert "a" not in linker.progress and len(linker.store.blocks) == 2
    assert [linker.store.blocks[k].parent for k in stored] == [None, stored[0]]
    record = linker.store.blocks[stored[0]].record
    assert record["tokens"] == BLOCK and record["compression_ratio"] < 1  # identity + metadata
    event = next(e for e in linker.events if e["event"] == "compress")
    assert event["latency_ms"] >= 0 and event["timer"] == "wall_sync"


def test_shared_prefix_compresses_only_new_blocks_with_prefix_closed_keys():
    adapter = make_adapter()
    linker = make_linker(adapter)
    first = make_req(adapter, "a", [1, 2, 3, 4, 5, 6, 7, 8], row=0, first_slot=0)
    linker.on_request_progress(first, 8, finished=True)
    second = make_req(adapter, "b", [1, 2, 3, 4, 9, 9, 9, 9], row=1, first_slot=20)
    linker.on_request_progress(second, 8, finished=True)
    compressed = [e for e in linker.events if e["event"] == "compress"]
    assert [(e["rid"], e["block"]) for e in compressed] == [("a", 0), ("a", 1), ("b", 1)]
    assert block_keys(first.origin_input_ids, 1) == block_keys(second.origin_input_ids, 1)
    assert len(linker.store.blocks) == 3


def test_lookup_is_whole_blocks_from_the_first_uncached_page():
    adapter = make_adapter()
    linker = make_linker(adapter)
    req = make_req(adapter, "a", range(10), row=0, first_slot=0)
    linker.on_request_progress(req, 10, finished=True)
    pages = get_hash_str(list(range(10)), None, page_size=PAGE)
    kv = lambda keys: [PoolTransfer(name=PoolName.KV, keys=keys)]
    assert linker.lookup("t", kv(pages)) == [2, 4]  # pages 4 (the tail) never hit
    assert linker.lookup("t", kv(pages[:3])) == [2]
    linker.store.evict(pages[3])
    assert linker.lookup("t", kv(pages)) == [2]
    # A native hit that is not block aligned leaves nothing restorable.
    assert linker.lookup("t", kv(pages[1:])) == []


@pytest.mark.parametrize("hybrid", [False, True])
def test_restore_scatters_whole_blocks_and_state_into_private_slots(hybrid):
    adapter = make_adapter(hybrid=hybrid)
    linker = make_linker(adapter)
    req = make_req(adapter, "a", range(8), row=0, first_slot=0, mamba_slot=2 if hybrid else None)
    # Chunk-aligned progress: a hybrid block's state exists only at its own end.
    for processed in (4, 8):
        linker.on_request_progress(req, processed, finished=False)
    keys = get_hash_str(list(range(8)), None, page_size=PAGE)
    expected = adapter.gather_kv(torch.arange(8))
    if hybrid:
        expected_state = adapter.gather_state(2)
        for tensor in adapter.req_pool.mamba_pool.mamba_cache.conv + [
            adapter.req_pool.mamba_pool.mamba_cache.temporal
        ]:
            tensor[:, 2].zero_()
    for buffer in adapter.kv.k_buffer + adapter.kv.v_buffer:
        buffer[:8].fill_(float("nan"))
    transfers = [PoolTransfer(name=PoolName.KV, keys=keys, device_indices=torch.arange(30, 38))]
    if hybrid:
        transfers.append(
            PoolTransfer(name=PoolName.MAMBA, keys=keys[-1:], device_indices=torch.tensor([5]))
        )
    assert linker.load("t", transfers)
    assert linker.start_layer_wise_loading() == 0
    restored = adapter.gather_kv(torch.arange(30, 38))
    for name in expected:
        assert torch.equal(restored[name], expected[name])
    if hybrid:
        for name, tensor in adapter.gather_state(5).items():
            assert torch.equal(tensor, expected_state[name])
    assert linker.pop_completed_load() == ["t"]
    decompress = next(e for e in linker.events if e["event"] == "decompress")
    assert decompress["blocks"] == 2 and decompress["tokens"] == 8


def test_hybrid_chunk_can_store_multiple_blocks_with_one_checkpoint():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter)
    req = make_req(adapter, "a", range(12), row=0, first_slot=0, mamba_slot=1)
    # One 8-token chunk contains two 4-token blocks. Both KV blocks are useful;
    # only the chunk-ending block owns the recurrent checkpoint.
    linker.on_request_progress(req, 8, finished=False)
    events = [(e["event"], e.get("block")) for e in linker.events]
    assert events == [("compress", 0), ("compress", 1)]
    keys = block_keys(req.origin_input_ids, 2)
    assert linker.store.state(keys[0]) is None
    assert linker.store.state(keys[1]) is not None
    pages = get_hash_str(req.origin_input_ids, None, page_size=PAGE)
    kv = [PoolTransfer(name=PoolName.KV, keys=pages)]
    assert linker.lookup("whole", kv) == [4]
    assert linker.lookup("half", [PoolTransfer(name=PoolName.KV, keys=pages[:2])]) == []
    # After a decode step the state has moved past every prompt boundary.
    req.output_ids = [7, 7]
    linker.on_request_progress(req, 12, finished=True)
    assert [e["event"] for e in linker.events[4:]] == ["compress"]
    assert linker.store.state(block_keys(req.origin_input_ids, 3)[-1]) is None


def test_hybrid_existing_kv_block_can_gain_a_later_checkpoint():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter)
    long = make_req(adapter, "long", range(8), row=0, first_slot=0, mamba_slot=1)
    linker.on_request_progress(long, 8, finished=False)
    key = block_keys(long.origin_input_ids, 1)[0]
    assert linker.store.state(key) is None

    short = make_req(adapter, "short", range(4), row=1, first_slot=20, mamba_slot=2)
    linker.on_request_progress(short, 4, finished=False)
    assert linker.store.state(key) is not None
    assert linker.events[-1]["event"] == "state_checkpoint"


def test_frozen_store_and_unadmitted_requests_write_nothing():
    adapter = make_adapter()
    linker = make_linker(adapter)
    req = make_req(adapter, "a", range(8), row=0, first_slot=0)
    linker.writes_enabled = False
    linker.on_request_progress(req, 8, finished=False)
    linker.writes_enabled = True
    req.kv.req_pool_idx = None
    linker.on_request_progress(req, 8, finished=False)
    assert not linker.store.blocks and not linker.events


def test_store_owns_bytes_and_accounts_for_payload_metadata_and_state():
    plugin, identity = load_plugin("identity")
    store = BlockStore(plugin, identity, 2, store_bytes=1 << 20)
    source = torch.randn(2, 2, 16, 4, 8).bfloat16()
    state = {"temporal": torch.ones(3, 5)}
    expected = source.clone()
    record = store.insert("k", {"key": source}, context={"page_size": 16}, state=state)
    source.zero_()
    state["temporal"].zero_()
    restored = store.reconstruct(["k"])
    assert torch.equal(restored.blocks[0]["key"], expected)
    assert torch.equal(restored.pages("key", 1), expected[:, 1].reshape(-1, 4, 8))
    assert store.state("k")["temporal"].eq(1).all()
    assert record["original_bytes"] == expected.nbytes
    assert record["state_bytes"] == 3 * 5 * 4
    assert record["metadata_bytes"] > 0
    assert record["compressed_bytes"] == record["payload_bytes"] + record["metadata_bytes"]
    assert record["payload_bytes"] == expected.nbytes
    assert store.capacity == (1 << 20) // (expected.nbytes + record["state_bytes"])


def test_failed_or_malformed_insert_publishes_nothing():
    class Fails(IdentityPlugin):
        def compress(self, tensors, *, context):
            raise ValueError("intentional failure")

    store = BlockStore(Fails(), {}, 2, store_bytes=1 << 20)
    with pytest.raises(ValueError, match="intentional"):
        store.insert("k", {"key": torch.ones(2, 32)}, context={})
    assert not store.blocks and store.pool is None
    store = BlockStore(IdentityPlugin(), {}, 2, store_bytes=1 << 20)
    with pytest.raises(ValueError, match="leading block_pages"):
        store.insert("k", {"key": torch.ones(3, 32)}, context={})
    store.insert("k", {"key": torch.ones(2, 32)}, context={})
    with pytest.raises(ValueError, match="already stored"):
        store.insert("k", {"key": torch.ones(2, 32)}, context={})


def test_int8_bound_and_zero_rows():
    plugin, identity = load_plugin("int8")
    store = BlockStore(plugin, identity, 4, store_bytes=1 << 20)
    tensor = torch.randn(4, 128, 128).bfloat16()
    tensor[0].zero_()
    record = store.insert("k", {"key": tensor}, context={})
    result = store.reconstruct(["k"]).blocks[0]["key"]
    assert result[0].count_nonzero() == 0
    assert (result.float() - tensor.float()).abs().max() < 0.04
    assert record["compressed_bytes"] < record["original_bytes"]


def test_file_plugin_fingerprints_code_and_parameters(tmp_path):
    path = tmp_path / "algorithm.py"
    path.write_text(
        "from kvcompress.loader import load_plugin\nIdentityPlugin = type(load_plugin('identity')[0])\n"
        "def create_plugin(config):\n    return IdentityPlugin()\n"
    )
    _, a = load_plugin(str(path), {"rank": 4})
    _, b = load_plugin(str(path), {"rank": 8})
    assert a["fingerprint"] != b["fingerprint"]
    path.write_text(path.read_text() + "# changed\n")
    _, c = load_plugin(str(path), {"rank": 4})
    assert a["fingerprint"] != c["fingerprint"]


def _put(store, key, parent=None, value=1.0):
    return store.insert(
        key, {"key": torch.full((2, 32), float(value))}, context={}, parent=parent
    )


def test_pool_evicts_oldest_leaf_and_reuses_slots():
    store = BlockStore(IdentityPlugin(), {}, 2, store_bytes=3 * 2 * 32 * 4)
    _put(store, "a0", value=1)
    _put(store, "a1", "a0", value=2)
    _put(store, "b0", value=3)
    assert store.capacity == 3 and not store.free
    # a0 is the oldest block but has a continuation: its tail a1 goes first.
    assert _put(store, "b1", "b0", value=4)["evicted"] == ["a1"]
    assert set(store.blocks) == {"a0", "b0", "b1"} and store.evictions == 1
    # a0 is now a leaf and the oldest block.
    assert _put(store, "c0", value=5)["evicted"] == ["a0"]
    restored = store.reconstruct(["b0", "b1"]).blocks
    assert restored[0]["key"].eq(3).all() and restored[1]["key"].eq(4).all()
    assert store.reconstruct(["c0"]).blocks[0]["key"].eq(5).all()
    with pytest.raises(KeyError, match="evicted"):
        store.reconstruct(["a1"])
    store.reset()
    assert not store.blocks and len(store.free) == 3 and store.pool is not None


def test_lookup_hits_refresh_recency():
    store = BlockStore(IdentityPlugin(), {}, 2, store_bytes=2 * 2 * 32 * 4)
    _put(store, "a")
    _put(store, "b")
    assert store.has("a") and not store.has("zzz")
    assert _put(store, "c")["evicted"] == ["b"]


def test_pool_rejects_layout_drift_and_tiny_budgets():
    store = BlockStore(IdentityPlugin(), {}, 2, store_bytes=1 << 20)
    _put(store, "k")
    with pytest.raises(ValueError, match="layout"):
        store.insert("j", {"key": torch.ones(2, 16)}, context={})
    with pytest.raises(ValueError, match="layout"):
        store.insert("j", {"key": torch.ones(2, 32)}, context={}, state={"s": torch.ones(2)})
    with pytest.raises(ValueError, match="holds no block"):
        BlockStore(IdentityPlugin(), {}, 2, store_bytes=8).insert(
            "k", {"key": torch.ones(2, 32)}, context={}
        )


def test_full_pool_evicts_and_records_at_the_linker():
    adapter = make_adapter()
    linker = make_linker(adapter, store_bytes=256)  # exactly one block
    req = make_req(adapter, "a", range(8), row=0, first_slot=0)
    linker.on_request_progress(req, 8, finished=True)
    keys = block_keys(req.origin_input_ids, 2)
    assert [(e["event"], e["key"]) for e in linker.events] == [
        ("compress", keys[0]),
        ("evict", keys[0]),
        ("compress", keys[1]),
    ]
    assert linker.events[1]["rid"] == "a" and linker.store.evictions == 1
    assert "evicted" not in linker.events[2]


@pytest.mark.parametrize("hybrid", [False, True])
def test_startup_probe_reserves_the_pool_for_real_blocks(hybrid):
    adapter = make_adapter(hybrid=hybrid)
    linker = make_linker(adapter, store_bytes=4096, preallocate=False)
    pool = linker._allocate_store()
    assert pool["capacity_blocks"] == 4096 // pool["bytes_per_block"] >= 1
    assert (pool["state_bytes_per_block"] > 0) == hybrid
    req = make_req(adapter, "a", range(8), row=0, first_slot=0, mamba_slot=1 if hybrid else None)
    for processed in (4, 8):
        linker.on_request_progress(req, processed, finished=False)
    assert len(linker.store.blocks) == 2


def test_compressed_only_match_never_queries_native_tree_or_session():
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    empty, hit = object(), object()
    cache.tree_core = SimpleNamespace(empty_match_result=empty)
    seen = []
    cache.linker = SimpleNamespace(
        cache_linker=SimpleNamespace(compressed_only=True),
        match=lambda key, req, result: seen.append(result) or hit,
    )
    cache.disable = True
    params = MatchPrefixParams(key=RadixKey([1, 2, 3]), req=object())
    assert cache.match_prefix(params) is hit
    assert seen == [empty]
    assert not cache.supports_fast_match_prefix()
    assert cache.match_prefix(MatchPrefixParams(key=params.key)) is empty


@pytest.mark.parametrize("hybrid", [False, True])
def test_private_load_allocates_each_duplicate_separately_without_tree_insert(hybrid):
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        ExternalCacheHitMarker,
        UnifiedCacheLinkerWrapper,
    )

    allocated = [0]

    def build_full(phase, node, keys):
        start = allocated[0]
        allocated[0] += 16
        return PoolTransfer(
            name=PoolName.KV, keys=keys, device_indices=torch.arange(start, start + 16)
        )

    components = [SimpleNamespace(build_external_linker_transfer=build_full)]
    if hybrid:

        def build_state(phase, node, keys):
            return PoolTransfer(
                name=PoolName.MAMBA,
                keys=keys[-1:],
                device_indices=torch.tensor([allocated[0]]),
            )

        components.append(SimpleNamespace(build_external_linker_transfer=build_state))
    empty = SimpleNamespace(
        device_indices=torch.empty(0, dtype=torch.int64), last_device_node=0
    )
    cache = SimpleNamespace(
        disable=True,
        tree_core=SimpleNamespace(empty_match_result=empty),
        _components_tuple=components,
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper.cache_linker = SimpleNamespace(
        compressed_only=True, record=lambda event: None
    )
    hit = ExternalCacheHitMarker(
        prefix_key=RadixKey(list(range(16))), tail_hashes=["a"], device_hit_len=0
    )
    wrapper.hit_markers = {"a": hit, "b": hit}
    wrapper.private_loads = {}
    queued = []
    wrapper._queue_load = lambda rid, node, transfers: queued.append((rid, transfers))
    indices = []
    for rid in ("a", "b"):
        req = SimpleNamespace(rid=rid, kv=SimpleNamespace(holds_mamba=False))
        slots, node = wrapper.load_back(req)
        indices.append(set(slots.tolist()))
        again, _ = wrapper.load_back(req)
        assert again is slots
        assert node == 0
        if hybrid:
            assert req.kv.mamba_cow_src_index is None
            assert not req.kv.mamba_needs_clear
    assert not indices[0].intersection(indices[1])
    assert [rid for rid, _ in queued] == ["a", "b"]


@pytest.mark.parametrize("has_row", [False, True])
def test_private_abort_cancels_load_and_frees_only_unadmitted_slots(has_row):
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

    linker = CompressionLinker.__new__(CompressionLinker)
    linker.compressed_only = True
    linker.queued = {"r": []}
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache_linker = linker
    wrapper.hit_markers = {}
    slots = torch.arange(16)
    wrapper.private_loads = {
        "r": (SimpleNamespace(kv=SimpleNamespace(holds_kv=has_row)), slots)
    }
    wrapper.pending_loads = {"r": (0, None)}
    freed = []
    wrapper.cache = SimpleNamespace(
        dec_lock_ref=lambda *args: None,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda value: freed.append(value)
        ),
    )
    wrapper.release_request("r")
    assert not linker.queued and not wrapper.pending_loads and not wrapper.private_loads
    assert len(freed) == (0 if has_row else 1)
    if freed:
        assert freed[0] is slots
