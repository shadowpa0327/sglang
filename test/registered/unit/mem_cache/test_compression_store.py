from collections import deque
from types import SimpleNamespace

import pytest
import torch

from kvcompress.loader import load_plugin
from kvcompress.request_store import RequestStore
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


def make_adapter(hybrid=False, swa=False, slots=128):
    torch.manual_seed(3)
    adapter = NativePoolAdapter.__new__(NativePoolAdapter)
    adapter.page_size, adapter.device, adapter.hybrid = (
        PAGE,
        torch.device("cpu"),
        hybrid,
    )
    adapter.layer_ids = [0, 1]
    adapter.temporal_layer_ids = [0, 1, 2]
    adapter.temporal_kind = "gdn"
    adapter.has_swa = swa
    adapter.swa_layer_ids = [2, 3, 4] if swa else []
    adapter.sliding_window_size = 6 if swa else None
    adapter.swa_window_pages = 3 if swa else 0
    adapter.kv = SimpleNamespace(
        start_layer=0,
        k_buffer=[torch.randn(slots, 2, 4).bfloat16() for _ in range(2)],
        v_buffer=[torch.randn(slots, 2, 4).bfloat16() for _ in range(2)],
    )
    adapter.allocator = SimpleNamespace()
    if swa:
        adapter.swa_kv = SimpleNamespace(
            k_buffer=[torch.randn(slots, 3, 5).bfloat16() for _ in range(3)],
            v_buffer=[torch.randn(slots, 3, 6).bfloat16() for _ in range(3)],
        )
        adapter.pool = SimpleNamespace(
            translate_loc_from_full_to_swa=lambda indices: indices + 64
        )
    else:
        adapter.swa_kv = None
        adapter.pool = adapter.kv
    adapter.req_pool = SimpleNamespace(
        req_to_token=torch.zeros(4, slots, dtype=torch.int64)
    )
    if hybrid:
        adapter.req_pool.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.randn(3, 8, 5).bfloat16(),
                conv=[torch.randn(3, 8, 6).bfloat16()],
            )
        )
        adapter.req_pool.translate_mamba_indices = lambda x: x
    return adapter


def make_linker(
    adapter,
    plugin=None,
    store_bytes=1 << 20,
    *,
    preallocate=True,
    compression_unit="block",
):
    linker = CompressionLinker.__new__(CompressionLinker)
    linker.adapter = adapter
    linker.page_size = PAGE
    linker.block_pages = BLOCK_PAGES if compression_unit == "block" else None
    linker.block_tokens = BLOCK if compression_unit == "block" else None
    linker.restoration_boundary_tokens = BLOCK
    linker.compression_unit = compression_unit
    linker.store = (
        BlockStore(
            plugin or IdentityPlugin(),
            {},
            BLOCK_PAGES,
            store_bytes=store_bytes,
        )
        if compression_unit == "block"
        else RequestStore(plugin or IdentityPlugin(), {}, store_bytes=store_bytes)
    )
    linker.pre_rope = linker.swa_pre_rope = linker.audit = linker.audit_fault = None
    linker.progress, linker.checkpoints, linker.queued = {}, {}, {}
    linker.request_pages, linker.request_hits = {}, {}
    linker.completed_loads = deque()
    linker.layer_done_counter = CompletedBatchCounter()
    linker.writes_enabled, linker.compressed_only = True, True
    linker.cache_mode = "compressed"
    linker.request_events = {}
    linker.events = []
    linker.record = linker.events.append
    if preallocate and compression_unit == "block":
        sample = adapter.gather_kv(torch.arange(BLOCK))
        context = adapter.context()
        sample, context = linker._add_swa(
            sample,
            context,
            None,
            checkpoint_position=BLOCK,
            fixed_layout=True,
        )
        state = adapter.gather_state(0) if adapter.hybrid else None
        sample, context, conv = linker._add_temporal(
            sample, context, state, checkpoint_position=BLOCK
        )
        linker.store.allocate(sample, context=context, state=conv)
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
    assert (
        record["tokens"] == BLOCK and record["compression_ratio"] < 1
    )  # identity + metadata
    event = next(e for e in linker.events if e["event"] == "compress")
    assert event["latency_ms"] >= 0 and event["timer"] == "wall_sync"


def test_request_unit_gathers_one_restorable_span_only_at_finish():
    adapter = make_adapter()
    linker = make_linker(adapter, compression_unit="request")
    gathered = []
    gather_kv = adapter.gather_kv

    def record_gather(slots):
        gathered.append(slots.clone())
        return gather_kv(slots)

    adapter.gather_kv = record_gather
    req = make_req(adapter, "a", range(10), row=0, first_slot=10)
    linker.on_request_progress(req, 8, finished=False)
    assert gathered == []
    linker.on_request_progress(req, 10, finished=True)
    assert len(gathered) == 1
    assert torch.equal(gathered[0], torch.arange(10, 18))
    assert len(linker.store.requests) == 1
    event = next(event for event in linker.events if event["event"] == "compress")
    assert event["compression_unit"] == "request"
    assert event["restorable_tokens"] == 8
    assert event["pages"] == 4


def test_request_unit_requires_the_complete_input_identity():
    adapter = make_adapter()
    linker = make_linker(adapter, compression_unit="request")
    stored = make_req(
        adapter, "stored", [1, 2, 3, 4, 5, 6, 7, 8, 9], row=0, first_slot=0
    )
    linker.on_request_progress(stored, 9, finished=True)
    keys = get_hash_str(stored.origin_input_ids[:8], None, page_size=PAGE)
    transfers = [PoolTransfer(name=PoolName.KV, keys=keys)]

    exact = make_req(adapter, "exact", stored.origin_input_ids, row=1, first_slot=20)
    assert linker.lookup_request(exact, transfers, device_hit_pages=1) == []
    assert linker.events[-1]["reason"] == "native_prefix_present"
    assert linker.store.description()["pinned_requests"] == 0
    assert linker.lookup_request(exact, transfers) == [4]
    assert linker.store.description()["pinned_requests"] == 1
    changed_tail = make_req(
        adapter,
        "changed",
        [1, 2, 3, 4, 5, 6, 7, 8, 99],
        row=2,
        first_slot=30,
    )
    assert linker.lookup_request(changed_tail, transfers) == []
    assert linker.events[-1]["exact_request_hit"] is False
    namespaced = make_req(
        adapter, "namespaced", stored.origin_input_ids, row=3, first_slot=40
    )
    namespaced.cache_salt = "different-tenant"
    assert linker.lookup_request(namespaced, transfers) == []
    embedded = make_req(
        adapter, "embedded", stored.origin_input_ids, row=3, first_slot=40
    )
    embedded.input_embeds = torch.randn(9, 8)
    assert linker.lookup_request(embedded, transfers) == []
    assert linker.events[-1]["reason"] == "custom_input_embeddings"
    linker._release_request_hit(exact.rid)
    assert linker.store.description()["pinned_requests"] == 0


def test_request_larger_than_the_budget_is_reported_without_failing_the_request():
    adapter = make_adapter()
    linker = make_linker(adapter, store_bytes=64, compression_unit="request")
    req = make_req(adapter, "large", range(9), row=0, first_slot=0)

    linker.on_request_progress(req, 9, finished=True)

    assert not linker.store.requests
    assert linker.events == [
        {
            "event": "compress_rejected",
            "rid": "large",
            "key": linker._request_key(req),
            "compression_unit": "request",
            "restorable_tokens": 8,
            "reason": "Request needs 512 bytes but store_bytes=64",
        }
    ]


def test_request_unit_does_not_store_token_ambiguous_embedding_inputs():
    adapter = make_adapter()
    linker = make_linker(adapter, compression_unit="request")
    req = make_req(adapter, "embedded", range(9), row=0, first_slot=0)
    req.positional_embed_overrides = object()

    linker.on_request_progress(req, 9, finished=True)

    assert not linker.store.requests
    assert linker.events[-1]["reason"] == "custom_input_embeddings"


def test_shared_prefix_compresses_only_new_blocks_with_prefix_closed_keys():
    adapter = make_adapter()
    linker = make_linker(adapter)
    first = make_req(adapter, "a", [1, 2, 3, 4, 5, 6, 7, 8], row=0, first_slot=0)
    linker.on_request_progress(first, 8, finished=True)
    second = make_req(adapter, "b", [1, 2, 3, 4, 9, 9, 9, 9], row=1, first_slot=20)
    linker.on_request_progress(second, 8, finished=True)
    compressed = [e for e in linker.events if e["event"] == "compress"]
    assert [(e["rid"], e["block"]) for e in compressed] == [
        ("a", 0),
        ("a", 1),
        ("b", 1),
    ]
    assert block_keys(first.origin_input_ids, 1) == block_keys(
        second.origin_input_ids, 1
    )
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
    req = make_req(
        adapter, "a", range(8), row=0, first_slot=0, mamba_slot=2 if hybrid else None
    )
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
    transfers = [
        PoolTransfer(name=PoolName.KV, keys=keys, device_indices=torch.arange(30, 38))
    ]
    if hybrid:
        transfers.append(
            PoolTransfer(
                name=PoolName.MAMBA, keys=keys[-1:], device_indices=torch.tensor([5])
            )
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


@pytest.mark.parametrize("hybrid", [False, True])
def test_swa_checkpoint_uses_trailing_window_and_restores_swa_pool(hybrid):
    adapter = make_adapter(hybrid=hybrid, swa=True)
    linker = make_linker(adapter)
    req = make_req(
        adapter, "a", range(8), row=0, first_slot=0, mamba_slot=2 if hybrid else None
    )
    expected_full = adapter.gather_kv(torch.arange(8))
    expected_swa = adapter.gather_swa(torch.arange(2, 8))
    expected_state = adapter.gather_state(2) if hybrid else None

    # One callback completes two blocks. Only the callback-ending block owns
    # the SWA (and optional Mamba) checkpoint used to resume the prefix.
    linker.on_request_progress(req, 8, finished=False)
    keys = block_keys(req.origin_input_ids, 2)
    assert not linker.store.has_swa(keys[0])
    assert linker.store.has_swa(keys[1])
    assert linker.lookup(
        "target",
        [
            PoolTransfer(
                name=PoolName.KV,
                keys=get_hash_str(list(range(8)), None, page_size=PAGE),
            )
        ],
    ) == [4]

    for buffer in adapter.kv.k_buffer + adapter.kv.v_buffer:
        buffer[100:108].fill_(float("nan"))
    for buffer in adapter.swa_kv.k_buffer + adapter.swa_kv.v_buffer:
        buffer[90:96].fill_(float("nan"))
    transfers = [
        PoolTransfer(
            name=PoolName.KV,
            keys=get_hash_str(list(range(8)), None, page_size=PAGE),
            device_indices=torch.arange(100, 108),
        ),
        PoolTransfer(
            name=PoolName.SWA,
            keys=get_hash_str(list(range(8)), None, page_size=PAGE)[-3:],
            device_indices=torch.arange(90, 96),
        ),
    ]
    if hybrid:
        transfers.append(
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=[keys[-1]],
                device_indices=torch.tensor([5]),
            )
        )

    linker.load("target", transfers)
    linker.start_layer_wise_loading()
    restored_full = adapter.gather_kv(torch.arange(100, 108))
    for name in expected_full:
        assert torch.equal(restored_full[name], expected_full[name])
    restored_swa = {
        name: adapter._gather_buffers(
            buffers, torch.arange(90, 96), pages=3, page_size=PAGE
        )
        for name, buffers in (
            ("swa_key", adapter.swa_kv.k_buffer),
            ("swa_value", adapter.swa_kv.v_buffer),
        )
    }
    for name in expected_swa:
        assert torch.equal(restored_swa[name], expected_swa[name])
    if hybrid:
        for name, tensor in adapter.gather_state(5).items():
            assert torch.equal(tensor, expected_state[name])


def test_short_swa_prefix_is_left_aligned_and_scatter_ignores_padding():
    adapter = make_adapter(swa=True)
    linker = make_linker(adapter)
    req = make_req(adapter, "short", range(4), row=0, first_slot=0)
    expected = adapter.gather_swa(torch.arange(4))

    linker.on_request_progress(req, 4, finished=False)
    key = block_keys(req.origin_input_ids, 1)[0]
    decoded = linker.store.reconstruct([key]).blocks[0]
    assert decoded["swa_key"].shape[0] == adapter.swa_window_pages
    assert decoded["swa_key"][2:].count_nonzero() == 0

    # Static PoolName.SWA destinations are already SWA-physical. The adapter's
    # allocator intentionally has no translation method, catching double maps.
    target = torch.arange(90, 94)
    adapter.scatter_swa(target, decoded)
    restored = {
        name: adapter._gather_buffers(
            buffers, target, pages=2, page_size=adapter.page_size
        )
        for name, buffers in (
            ("swa_key", adapter.swa_kv.k_buffer),
            ("swa_value", adapter.swa_kv.v_buffer),
        )
    }
    for name in expected:
        assert torch.equal(restored[name], expected[name])


def test_unified_swa_scatter_translates_full_virtual_destinations_once():
    adapter = make_adapter(swa=True)
    adapter.allocator.translate_loc_from_full_to_swa = lambda slots: slots + 64
    block = adapter.gather_swa(torch.arange(4))
    virtual = torch.arange(10, 14)

    adapter.scatter_swa(virtual, block, indices_from_full=True)

    physical = virtual + 64
    restored = {
        name: adapter._gather_buffers(
            buffers, physical, pages=2, page_size=adapter.page_size
        )
        for name, buffers in (
            ("swa_key", adapter.swa_kv.k_buffer),
            ("swa_value", adapter.swa_kv.v_buffer),
        )
    }
    for name in block:
        assert torch.equal(restored[name], block[name])


def test_swa_gather_rejects_reserved_or_tombstoned_slot_zero():
    adapter = make_adapter(swa=True)
    adapter.pool.translate_loc_from_full_to_swa = lambda slots: torch.zeros_like(slots)

    with pytest.raises(RuntimeError, match="unmapped cache page"):
        adapter.gather_swa(torch.arange(4))


@pytest.mark.parametrize("hybrid", [False, True])
def test_request_unit_reconstructs_and_scatters_the_whole_stored_span(hybrid):
    adapter = make_adapter(hybrid=hybrid)
    linker = make_linker(adapter, compression_unit="request")
    source = make_req(
        adapter,
        "source",
        range(9),
        row=0,
        first_slot=0,
        mamba_slot=2 if hybrid else None,
    )
    expected = adapter.gather_kv(torch.arange(8))
    expected_state = adapter.gather_state(2) if hybrid else None
    if hybrid:
        linker.on_request_progress(source, 8, finished=False)
        source.output_ids = [17]
    linker.on_request_progress(source, 9, finished=True)
    stored_key = linker._request_key(source)
    if hybrid:
        assert set(linker.store.state(stored_key)) == {"conv_0"}
        assert "temporal" in linker.store.requests[stored_key].input_layout

    target = make_req(
        adapter,
        "target",
        range(9),
        row=1,
        first_slot=30,
        mamba_slot=5 if hybrid else None,
    )
    pages = get_hash_str(list(range(8)), None, page_size=PAGE)
    transfers = [
        PoolTransfer(
            name=PoolName.KV,
            keys=pages,
            device_indices=torch.arange(30, 38),
        )
    ]
    if hybrid:
        transfers.append(
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=pages[-1:],
                device_indices=torch.tensor([5]),
            )
        )
    assert linker.lookup_request(target, transfers) == [4]
    assert linker.store.description()["pinned_requests"] == 1
    assert linker.load(target.rid, transfers)
    assert linker.start_layer_wise_loading() == 0
    assert linker.store.description()["pinned_requests"] == 0
    restored = adapter.gather_kv(torch.arange(30, 38))
    for name in expected:
        assert torch.equal(restored[name], expected[name])
    if hybrid:
        for name, tensor in adapter.gather_state(5).items():
            assert torch.equal(tensor, expected_state[name])
    event = next(event for event in linker.events if event["event"] == "decompress")
    assert event["compression_unit"] == "request"
    assert event["requests"] == 1 and event["tokens"] == 8


def test_request_unit_unfinished_callbacks_only_keep_deepest_hybrid_checkpoint():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter, compression_unit="request")
    req = make_req(adapter, "a", range(13), row=0, first_slot=0, mamba_slot=2)
    gathered = []
    gather_kv = adapter.gather_kv

    def record_gather(slots):
        gathered.append(slots.clone())
        return gather_kv(slots)

    adapter.gather_kv = record_gather
    linker.on_request_progress(req, 4, finished=False)
    linker.on_request_progress(req, 8, finished=False)
    assert linker.checkpoints[req.rid][0] == 8
    assert gathered == []
    req.output_ids = [17]
    linker.on_request_progress(req, 13, finished=True)
    assert len(gathered) == 1
    assert torch.equal(gathered[0], torch.arange(8))
    assert next(iter(linker.store.requests.values())).record["tokens"] == 8


def test_request_unit_retains_swa_window_from_the_prefill_checkpoint():
    adapter = make_adapter(swa=True)
    linker = make_linker(adapter, compression_unit="request")
    req = make_req(adapter, "a", range(9), row=0, first_slot=0)
    expected = adapter.gather_swa(torch.arange(2, 8))

    linker.on_request_progress(req, 8, finished=False)
    for buffer in adapter.swa_kv.k_buffer + adapter.swa_kv.v_buffer:
        buffer[66:72].zero_()
    req.output_ids = [17]
    linker.on_request_progress(req, 9, finished=True)

    key = linker._request_key(req)
    assert linker.store.has_swa(key)
    restored = linker.store.reconstruct(key).tensors
    for name in expected:
        assert torch.equal(restored[name], expected[name])


def test_request_unit_does_not_recompress_or_skip_an_exact_existing_entry():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter, compression_unit="request")
    source = make_req(adapter, "source", range(9), row=0, first_slot=0, mamba_slot=2)
    linker.on_request_progress(source, 8, finished=False)
    source.output_ids = [17]
    linker.on_request_progress(source, 9, finished=True)

    replay = make_req(adapter, "replay", range(9), row=1, first_slot=20, mamba_slot=3)
    replay.output_ids = [17, 18]
    before = len(linker.events)
    linker.on_request_progress(replay, 9, finished=True)

    assert len(linker.events) == before
    assert linker.store.description()["requests_compressed"] == 1


def test_short_hybrid_request_without_an_earlier_checkpoint_is_not_stored():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter, compression_unit="request")
    req = make_req(adapter, "short", range(5), row=0, first_slot=0, mamba_slot=2)
    req.output_ids = [17]

    linker.on_request_progress(req, 5, finished=True)

    assert not linker.store.requests
    assert linker.events == [
        {
            "event": "compress_skipped",
            "rid": "short",
            "compression_unit": "request",
            "reason": "no_restorable_recurrent_checkpoint",
        }
    ]


def test_hybrid_request_ignores_non_restorable_prefill_checkpoint():
    adapter = make_adapter(hybrid=True)
    linker = make_linker(adapter, compression_unit="request")
    req = make_req(adapter, "unaligned", range(9), row=0, first_slot=0, mamba_slot=2)

    linker.on_request_progress(req, 6, finished=False)
    req.output_ids = [17]
    linker.on_request_progress(req, 9, finished=True)

    assert not linker.store.requests
    assert linker.events[-1]["reason"] == "no_restorable_recurrent_checkpoint"


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
    assert not linker.store.has_checkpoint(keys[0])
    assert linker.store.has_checkpoint(keys[1])
    expected_temporal = adapter.gather_state(1)["temporal"]
    assert torch.equal(
        linker.store.reconstruct([keys[1]]).blocks[0]["temporal"],
        expected_temporal,
    )
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
    old_metadata = linker.store.blocks[key].record["metadata_bytes"]
    cumulative_metadata = linker.store.description()["metadata_bytes_compressed"]

    short = make_req(adapter, "short", range(4), row=1, first_slot=20, mamba_slot=2)
    expected_temporal = adapter.gather_state(2)["temporal"]
    linker.on_request_progress(short, 4, finished=False)
    assert linker.store.state(key) is not None
    assert linker.store.has_checkpoint(key)
    assert torch.equal(
        linker.store.reconstruct([key]).blocks[0]["temporal"], expected_temporal
    )
    event = linker.events[-1]
    assert event["event"] == "state_checkpoint"
    assert event["exposed_bytes"] == (
        event["exposed_kv_bytes"] + event["exposed_temporal_bytes"]
    )
    assert event["payload_bytes"] == (
        event["kv_payload_bytes"] + event["temporal_payload_bytes"]
    )
    assert event["metadata_bytes"] > 0
    assert linker.store.description()["metadata_bytes_compressed"] == (
        cumulative_metadata - old_metadata + event["metadata_bytes"]
    )


def test_quantized_temporal_payload_is_reencoded_when_checkpoint_arrives():
    adapter = make_adapter(hybrid=True)
    plugin, _ = load_plugin("temporal_quant", {"bits": 8})
    linker = make_linker(adapter, plugin=plugin)
    long = make_req(adapter, "long", range(8), row=0, first_slot=0, mamba_slot=1)
    linker.on_request_progress(long, 8, finished=False)
    key = block_keys(long.origin_input_ids, 1)[0]
    assert not linker.store.has_checkpoint(key)

    short = make_req(adapter, "short", range(4), row=1, first_slot=20, mamba_slot=2)
    expected = adapter.gather_state(2)["temporal"]
    linker.on_request_progress(short, 4, finished=False)

    restored = linker.store.reconstruct([key]).blocks[0]["temporal"]
    scale = expected.abs().amax(dim=-2, keepdim=True) / 127
    assert torch.all((restored - expected).abs() <= scale / 2 + 1e-2)
    assert linker.store.has_checkpoint(key)
    assert linker.events[-1]["recompressed"] is True


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
    assert (
        record["compressed_bytes"] == record["payload_bytes"] + record["metadata_bytes"]
    )
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
        store.insert(
            "j", {"key": torch.ones(2, 32)}, context={}, state={"s": torch.ones(2)}
        )
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
@pytest.mark.parametrize("swa", [False, True])
def test_startup_probe_reserves_the_pool_for_real_blocks(hybrid, swa):
    adapter = make_adapter(hybrid=hybrid, swa=swa)
    linker = make_linker(adapter, store_bytes=4096, preallocate=False)
    pool = linker._allocate_store()
    assert pool["capacity_blocks"] == 4096 // pool["bytes_per_block"] >= 1
    assert (pool["state_bytes_per_block"] > 0) == hybrid
    assert (pool["swa_bytes_per_block"] > 0) == swa
    req = make_req(
        adapter, "a", range(8), row=0, first_slot=0, mamba_slot=1 if hybrid else None
    )
    for processed in (4, 8):
        linker.on_request_progress(req, processed, finished=False)
    assert len(linker.store.blocks) == 2


@pytest.mark.parametrize("cache_mode", ["none", "native"])
def test_non_compressed_modes_do_not_allocate_the_block_store(cache_mode):
    linker = CompressionLinker.__new__(CompressionLinker)
    linker.compression_unit = "block"
    linker.cache_mode = cache_mode
    linker.writes_enabled = False
    linker.store = SimpleNamespace(description=lambda: {"capacity_blocks": 0})
    linker._allocate_store = lambda: pytest.fail("allocated a disabled store")

    assert linker._initialize_store() == {"capacity_blocks": 0}


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


def test_compression_swa_restore_disables_the_legacy_reprefill_window():
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.components = {ComponentType.SWA: SimpleNamespace(sliding_window_size=1023)}
    cache.cache_controller = object()
    cache.tree_core = SimpleNamespace(has_swa_host_pool=False)
    cache.linker = SimpleNamespace(
        cache_linker=SimpleNamespace(restores_swa=False, compressed_only=True)
    )
    assert cache.swa_reprefill_tail_tokens() == 1023
    cache.linker.cache_linker.restores_swa = True
    assert cache.swa_reprefill_tail_tokens() == 0
    cache.linker.cache_linker.compressed_only = False
    assert cache.swa_reprefill_tail_tokens() == 1023


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

    def prepare(phase, req, full, transfer, prefix_len, **kwargs):
        return transfer

    components = [
        SimpleNamespace(
            build_external_linker_transfer=build_full,
            update_external_linker_load=prepare,
        )
    ]
    if hybrid:

        def build_state(phase, node, keys):
            return PoolTransfer(
                name=PoolName.MAMBA,
                keys=keys[-1:],
                device_indices=torch.tensor([allocated[0]]),
            )

        components.append(
            SimpleNamespace(
                build_external_linker_transfer=build_state,
                update_external_linker_load=prepare,
            )
        )
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


def test_private_swa_load_runs_prepare_to_install_the_mapping():
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        ExternalLinkerLoadPhase,
    )
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        ExternalCacheHitMarker,
        UnifiedCacheLinkerWrapper,
    )

    full_slots = torch.arange(20, 28)
    swa_slots = torch.arange(50, 54)
    prepared = []

    def update(phase, req, full, transfer, prefix_len, **kwargs):
        assert phase == ExternalLinkerLoadPhase.PREPARE
        if transfer.name == PoolName.SWA:
            prepared.append((full.device_indices[-4:].clone(), transfer.device_indices))
        return transfer

    components = [
        SimpleNamespace(
            build_external_linker_transfer=lambda phase, node, keys: PoolTransfer(
                name=PoolName.KV, keys=keys, device_indices=full_slots
            ),
            update_external_linker_load=update,
        ),
        SimpleNamespace(
            build_external_linker_transfer=lambda phase, node, keys: PoolTransfer(
                name=PoolName.SWA, keys=keys[-2:], device_indices=swa_slots
            ),
            update_external_linker_load=update,
        ),
    ]
    empty = SimpleNamespace(
        device_indices=torch.empty(0, dtype=torch.int64), last_device_node=0
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        disable=True,
        tree_core=SimpleNamespace(empty_match_result=empty),
        _components_tuple=components,
    )
    wrapper.cache_linker = SimpleNamespace(
        compressed_only=True, record=lambda event: None
    )
    wrapper.hit_markers = {
        "r": ExternalCacheHitMarker(
            prefix_key=RadixKey(list(range(8))),
            tail_hashes=["a", "b", "c", "d"],
            device_hit_len=0,
        )
    }
    wrapper.private_loads = {}
    wrapper._queue_load = lambda *args: None
    req = SimpleNamespace(rid="r", kv=SimpleNamespace())

    restored, _ = wrapper.load_back(req)

    assert torch.equal(restored, full_slots)
    assert wrapper.is_request_owned_load(req.rid)
    assert len(prepared) == 1
    assert torch.equal(prepared[0][0], full_slots[-4:])
    assert prepared[0][1] is swa_slots


def test_private_tri_pool_prepares_swa_before_allocating_mamba():
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        ExternalLinkerLoadPhase,
    )
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        ExternalCacheHitMarker,
        UnifiedCacheLinkerWrapper,
    )

    order = []
    swa_bound = [False]

    def make_component(name, transfer):
        def build(phase, node, keys):
            order.append(f"build_{name}")
            if name == "mamba":
                assert swa_bound[0], "Mamba allocation raced ahead of SWA binding"
            transfer.keys = keys[-1:] if name == "mamba" else keys
            return transfer

        def update(phase, req, full, current, prefix_len, **kwargs):
            assert phase == ExternalLinkerLoadPhase.PREPARE
            order.append(f"prepare_{name}")
            if name == "swa":
                swa_bound[0] = True
            return current

        return SimpleNamespace(
            build_external_linker_transfer=build,
            update_external_linker_load=update,
        )

    components = [
        make_component(
            "full",
            PoolTransfer(name=PoolName.KV, device_indices=torch.arange(20, 28)),
        ),
        make_component(
            "swa",
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(40, 44)),
        ),
        make_component(
            "mamba",
            PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([3])),
        ),
    ]
    empty = SimpleNamespace(
        device_indices=torch.empty(0, dtype=torch.int64), last_device_node=0
    )
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(
        disable=True,
        tree_core=SimpleNamespace(empty_match_result=empty),
        _components_tuple=components,
        req_to_token_pool=SimpleNamespace(
            mamba_allocator=SimpleNamespace(free=lambda slots: None)
        ),
    )
    wrapper.cache_linker = SimpleNamespace(
        compressed_only=True, record=lambda event: None
    )
    wrapper.hit_markers = {
        "r": ExternalCacheHitMarker(
            prefix_key=RadixKey(list(range(8))),
            tail_hashes=["a", "b", "c", "d"],
            device_hit_len=0,
        )
    }
    wrapper.private_loads = {}
    wrapper._queue_load = lambda *args: None
    req = SimpleNamespace(
        rid="r",
        kv=SimpleNamespace(
            holds_mamba=False,
            mamba_pool_idx=None,
            mamba_cow_src_index=None,
            mamba_needs_clear=True,
        ),
    )

    wrapper.load_back(req)

    assert order == [
        "build_full",
        "prepare_full",
        "build_swa",
        "prepare_swa",
        "build_mamba",
        "prepare_mamba",
    ]


def test_swa_external_load_allocates_static_and_binds_unified_slots():
    from sglang.srt.mem_cache.unified_cache.components.swa_component import (
        SWAComponent,
    )
    from sglang.srt.mem_cache.unified_cache.components.tree_component import (
        ExternalLinkerLoadPhase,
        LinkerTransferPhase,
    )

    class Allocator:
        device = torch.device("cpu")

        def __init__(self, start):
            self.start = start
            self.allocations = []
            self.bindings = []
            self.frees = []

        def available_size(self):
            return 100

        def alloc(self, count):
            self.allocations.append(count)
            return torch.arange(self.start, self.start + count)

        def alloc_with_virtual(self, pages):
            self.bindings.append(pages.clone())

        def free(self, slots):
            self.frees.append(slots.clone())

    def component(swa_allocator, unified):
        result = SWAComponent.__new__(SWAComponent)
        mappings = []
        token_allocator = SimpleNamespace(
            swa_attn_allocator=swa_allocator,
            set_full_to_swa_mapping=lambda full, swa: mappings.append(
                (full.clone(), swa.clone())
            ),
        )
        result.cache = SimpleNamespace(
            page_size=2,
            token_to_kv_pool_allocator=token_allocator,
            evict=lambda params: None,
        )
        result.sliding_window_size = 6
        result._unified_allocator = lambda: unified
        return result, mappings

    static_allocator = Allocator(50)
    static, mappings = component(static_allocator, None)
    transfer = static.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b", "c", "d"]
    )
    assert transfer.keys == ["b", "c", "d"]
    assert static_allocator.allocations == [6]
    full = PoolTransfer(name=PoolName.KV, device_indices=torch.arange(20, 28))
    static.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE,
        SimpleNamespace(kv=None),
        full,
        transfer,
        8,
    )
    assert torch.equal(mappings[0][0], full.device_indices[-6:])
    assert torch.equal(mappings[0][1], torch.arange(50, 56))

    unified_swa = Allocator(0)
    unified = SimpleNamespace(swa_attn_allocator=unified_swa)
    shared, mappings = component(unified_swa, unified)
    transfer = shared.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b", "c", "d"]
    )
    assert transfer.device_indices.numel() == 0
    shared.update_external_linker_load(
        ExternalLinkerLoadPhase.PREPARE,
        SimpleNamespace(kv=None),
        full,
        transfer,
        8,
    )
    assert torch.equal(transfer.device_indices, full.device_indices[-6:])
    assert transfer.indices_from_pool == PoolName.KV
    assert torch.equal(unified_swa.bindings[0], torch.tensor([11, 12, 13]))
    assert mappings == []
    canonical = torch.arange(70, 76)
    shared.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(kv=None),
        full,
        transfer,
        8,
        insert_result=SimpleNamespace(),
        canonical_full=canonical,
    )
    assert transfer.device_indices is canonical
    assert transfer.indices_from_pool == PoolName.KV


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
