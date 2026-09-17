from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from kvcompress.loader import load_plugin
from kvcompress.request_store import RequestStore
from kvcompress.store import BlockStore
from sglang.srt.mem_cache.compression.pre_rope import (
    NativeRoPEDecodePlugin,
    PreRoPETransform,
    RotaryTable,
)

PLUGIN = (
    Path(__file__).resolve().parents[4]
    / "examples/prefix_compression/xkv_svd_plugin.py"
)
CPU = torch.device("cpu")


def make_rope(neox=True, head=8, rotary=8, max_pos=4096, base=10000):
    return RotaryEmbedding(head, rotary, max_pos, base, neox, torch.bfloat16)


def test_exact_svd_matches_xkv_matrix_layout_and_factor_bytes():
    torch.manual_seed(9)
    plugin, _ = load_plugin(
        str(PLUGIN), {"layer_group_size": 3, "rank_k": 3, "rank_v": 5}
    )
    tensors = {name: torch.randn(2, 5, 8, 2, 4).bfloat16() for name in ("key", "value")}
    payload = plugin.compress(tensors, context={})
    out = {name: torch.empty_like(t) for name, t in tensors.items()}
    scratch = plugin.prepare_decompression(payload, out=out)
    # Two widths (three layers and the trailing two), shared across K and V.
    assert len({t.data_ptr() for t in scratch.values()}) == 2
    plugin.decompress(payload, out=out, scratch=scratch)
    nbytes = 0
    for group in payload.metadata["groups"]:
        name, first, count, rank = (
            group["name"],
            group["first"],
            group["layers"],
            group["rank"],
        )
        # Independent xKV view: concatenate layers as heads, then transpose tokens first.
        source = (
            tensors[name][:, first : first + count]
            .permute(1, 3, 0, 2, 4)
            .reshape(count * 2, 16, 4)
        )
        matrix = source.transpose(0, 1).reshape(16, count * 8).float()
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        expected = (u[:, :rank] * s[:rank].sqrt()).bfloat16() @ (
            s[:rank].sqrt()[:, None] * vh[:rank]
        ).bfloat16()
        actual = (
            out[name][:, first : first + count]
            .permute(0, 2, 1, 3, 4)
            .reshape_as(expected)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        nbytes += rank * (16 + count * 8) * 2
    assert payload.tensor_bytes == nbytes


def test_randomized_solver_rng_isolation_and_partial_group_restoration():
    plugin, identity = load_plugin(
        str(PLUGIN), {"solver": "randomized", "rank_k": 3, "rank_v": 3}
    )
    tensors = {name: torch.randn(4, 4, 8, 2, 8).bfloat16() for name in ("key", "value")}
    state = torch.random.get_rng_state().clone()
    first = plugin.compress(tensors, context={"start_position": 512})
    second = plugin.compress(tensors, context={"start_position": 512})
    assert torch.equal(state, torch.random.get_rng_state())
    for name in first.tensors:
        assert torch.equal(first.tensors[name], second.tensors[name])
    store = BlockStore(plugin, identity, 4, store_bytes=1 << 20)
    record = store.insert("k", tensors, context={})
    assert record["compressed_bytes"] < record["original_bytes"]
    for tensor in tensors.values():
        tensor.fill_(float("nan"))
    decoded = store.reconstruct(["k"])
    assert decoded.measurement["pages"] == 4
    assert all(torch.isfinite(t).all() for t in decoded.blocks[0].values())


@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("rotary", [8, 4])
def test_table_matches_and_inverts_the_models_native_rotation(neox, rotary):
    torch.manual_seed(1)
    rope = make_rope(neox, 8, rotary)
    table = RotaryTable(rope, CPU)
    positions = torch.arange(1000, 1016)
    key = torch.randn(16, 2, 8)
    _, native = rope.forward_native(positions, torch.zeros(16, 1, 8), key.clone())
    cos, sin = table.rows(positions)
    out, scratch = torch.empty_like(key), torch.empty(16, 2, rotary // 2)
    table.rotate(key, cos, sin, inverse=False, out=out, scratch=scratch)
    torch.testing.assert_close(out, native, atol=1e-5, rtol=1e-5)
    back = torch.empty_like(key)
    table.rotate(out, cos, sin, inverse=True, out=back, scratch=scratch)
    torch.testing.assert_close(back, key, atol=1e-5, rtol=1e-5)
    if rotary < 8:
        assert torch.equal(out[..., rotary:], key[..., rotary:])


@pytest.mark.parametrize("offset", [0, 1024])
def test_pre_rope_store_roundtrip_uses_absolute_positions(offset):
    torch.manual_seed(2)
    events = []
    transform = PreRoPETransform(
        RotaryTable(make_rope(), CPU), layers=2, page_size=4, record=events.append
    )
    inner, identity = load_plugin("identity")
    wrapper = NativeRoPEDecodePlugin(inner, transform)
    store = BlockStore(wrapper, identity, 4, store_bytes=1 << 20)
    native = torch.randn(4, 2, 4, 1, 8).bfloat16()
    pre = transform.derotate(native, start_position=offset, audit=True)
    assert events[-1]["event"] == "audit_pre_rope_roundtrip"
    assert events[-1]["max_abs_error"] <= events[-1]["bound"]
    assert pre.dtype == native.dtype and not torch.equal(pre, native)
    context = {
        "key_space": "pre_rope",
        "page_size": 4,
        "start_position": offset,
        "token_ids": list(range(offset, offset + 16)),
        "layer_ids": [0, 1],
    }
    temporal = torch.randn(3, 2, 4, 8)
    tensors = {"key": pre, "value": native.clone(), "temporal": temporal}
    store.insert("k", tensors, context=context)
    decoded = store.reconstruct(["k"])
    restored = decoded.blocks[0]["key"][:3]
    eps = torch.finfo(torch.bfloat16).eps
    tolerance = 2 * eps * native.float().abs().max()
    assert (restored.float() - native[:3].float()).abs().max() <= tolerance
    assert torch.equal(decoded.blocks[0]["value"][:3], native[:3])
    assert torch.equal(decoded.blocks[0]["temporal"], temporal)
    # A wrong absolute frame must be visible: claim the keys sit 8 tokens later.
    shifted = BlockStore(wrapper, identity, 4, store_bytes=1 << 20)
    later = {**context, "start_position": offset + 8}
    shifted.insert("k", tensors, context=later)
    decoded = shifted.reconstruct(["k"])
    wrong = decoded.blocks[0]["key"][:3]
    assert (wrong.float() - native[:3].float()).abs().max() > 10 * tolerance


def test_request_store_roundtrip_rerotates_one_complete_span():
    torch.manual_seed(4)
    transform = PreRoPETransform(RotaryTable(make_rope(), CPU), layers=2, page_size=4)
    inner, identity = load_plugin("identity")
    wrapper = NativeRoPEDecodePlugin(inner, transform)
    store = RequestStore(wrapper, identity, store_bytes=1 << 20)
    offset = 512
    native = torch.randn(4, 2, 4, 1, 8).bfloat16()
    pre = transform.derotate(native, start_position=offset)
    temporal = torch.randn(3, 2, 4, 8)
    store.insert(
        "request-key",
        {"key": pre, "value": native.clone(), "temporal": temporal},
        context={
            "key_space": "pre_rope",
            "page_size": 4,
            "start_position": offset,
            "token_ids": list(range(16)),
            "layer_ids": [0, 1],
        },
    )
    tensors = store.reconstruct("request-key").tensors
    restored = tensors["key"]
    tolerance = 2 * torch.finfo(torch.bfloat16).eps * native.float().abs().max()
    assert (restored.float() - native.float()).abs().max() <= tolerance
    assert torch.equal(tensors["temporal"], temporal)


def test_full_and_swa_keys_use_independent_rope_tables_and_positions():
    torch.manual_seed(5)
    transforms = {
        "key": PreRoPETransform(
            RotaryTable(make_rope(base=10000), CPU), layers=2, page_size=4
        ),
        "swa_key": PreRoPETransform(
            RotaryTable(make_rope(base=1000000), CPU), layers=3, page_size=4
        ),
    }
    inner, identity = load_plugin("identity")
    wrapper = NativeRoPEDecodePlugin(inner, transforms)
    store = RequestStore(wrapper, identity, store_bytes=1 << 20)
    full_native = torch.randn(4, 2, 4, 1, 8).bfloat16()
    swa_native = torch.randn(2, 3, 4, 1, 8).bfloat16()
    full_start, swa_start = 512, 520
    tensors = {
        "key": transforms["key"].derotate(full_native, start_position=full_start),
        "value": torch.randn_like(full_native),
        "swa_key": transforms["swa_key"].derotate(swa_native, start_position=swa_start),
        "swa_value": torch.randn_like(swa_native),
    }
    context = {
        "key_space": "pre_rope",
        "page_size": 4,
        "start_position": full_start,
        "layer_ids": [0, 3],
        "tensor_specs": {
            "key": {"layer_ids": [0, 3]},
            "swa_key": {"layer_ids": [1, 2, 4], "start_position": swa_start},
        },
    }
    store.insert("k", tensors, context=context)
    assert store.requests["k"].payload.tensor_roles == {
        "key": "kv",
        "value": "kv",
        "swa_key": "swa",
        "swa_value": "swa",
    }
    restored = store.reconstruct("k").tensors
    eps = torch.finfo(torch.bfloat16).eps
    full_bound = 2 * eps * full_native.float().abs().max()
    swa_bound = 2 * eps * swa_native.float().abs().max()
    assert (restored["key"].float() - full_native.float()).abs().max() <= full_bound
    assert (restored["swa_key"].float() - swa_native.float()).abs().max() <= swa_bound
    assert torch.equal(restored["value"], tensors["value"])
    assert torch.equal(restored["swa_value"], tensors["swa_value"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused RoPE needs CUDA")
def test_cuda_fused_path_matches_model_kernel_and_torch_path():
    torch.manual_seed(3)
    dev = torch.device("cuda")
    rope = make_rope(head=128, rotary=64, max_pos=8192).to(dev)
    table = RotaryTable(rope, device=dev)
    assert table.fused is not None
    transform = PreRoPETransform(table, layers=2, page_size=16)
    native = torch.randn(4, 2, 16, 2, 128, device=dev).bfloat16()
    pre = transform.derotate(native, start_position=700, audit=True)
    scratch = transform.prepare_rerotation(pre.clone(), start_position=700)
    restored = pre.clone()
    transform.rerotate_(restored, scratch)
    # Re-rotation is the model kernel itself: compare with rope() on pre-RoPE keys.
    positions = torch.arange(700, 700 + 64, device=dev)
    dummy = torch.zeros(64, 1, 128, device=dev).bfloat16()
    for layer in range(2):
        key = pre[:, layer].reshape(64, 2, 128).clone()
        _, expected = rope(positions, dummy.clone(), key)
        assert torch.equal(restored[:, layer].reshape(64, 2, 128), expected)
    tolerance = 2 * torch.finfo(torch.bfloat16).eps * native.float().abs().max()
    assert (restored.float() - native.float()).abs().max() <= tolerance
    # Torch FP32 path on CPU agrees with the fused path to within one rounding.
    cpu_table = RotaryTable(rope.cpu(), device=torch.device("cpu"))
    cpu = PreRoPETransform(cpu_table, layers=2, page_size=16)
    pre_cpu = cpu.derotate(native.cpu(), start_position=700)
    assert (pre_cpu.float() - pre.cpu().float()).abs().max() <= tolerance


def test_transform_rejects_uncovered_positions_and_non_1d_tables():
    table = RotaryTable(make_rope(max_pos=64), CPU)
    transform = PreRoPETransform(table, layers=1, page_size=4)
    key = torch.randn(2, 1, 4, 1, 8).bfloat16()
    transform.derotate(key, start_position=56)
    with pytest.raises(ValueError, match="does not cover"):
        transform.derotate(key, start_position=60)
    with pytest.raises(ValueError, match="does not match layers"):
        wrong_layers = torch.randn(2, 3, 4, 1, 8).bfloat16()
        transform.derotate(wrong_layers, start_position=0)
    with pytest.raises(ValueError, match="needs a RotaryEmbedding"):
        RotaryTable(SimpleNamespace(cos_sin_cache=torch.zeros(4, 8)), CPU)


def test_plugins_use_the_api_v2_canonical_pre_rope_contract():
    svd, _ = load_plugin(str(PLUGIN), {"rank_k": 2, "rank_v": 2})
    assert svd.api_version == 2
    assert load_plugin("identity")[0].api_version == 2
    assert load_plugin("int8")[0].api_version == 2


def test_discovery_requires_exactly_one_live_table_for_the_pool_head_size(monkeypatch):
    from sglang.srt.layers.rotary_embedding import factory

    adapter = SimpleNamespace(
        kv=SimpleNamespace(head_dim=8), device=CPU, layer_ids=[0, 1], page_size=4
    )
    monkeypatch.setattr(factory, "_ROPE_DICT", {})
    with pytest.raises(ValueError, match="exactly one"):
        PreRoPETransform.from_model_ropes(adapter)
    assert PreRoPETransform.from_model_ropes(adapter, allow_no_rope=True) is None
    other = make_rope(head=16, rotary=16)
    monkeypatch.setattr(factory, "_ROPE_DICT", {("a",): make_rope(), ("b",): other})
    transform = PreRoPETransform.from_model_ropes(adapter)
    assert transform.table.head_size == 8 and transform.layers == 2
    assert transform.table.description["class"] == "RotaryEmbedding"
    twin = make_rope(neox=False)
    monkeypatch.setattr(factory, "_ROPE_DICT", {("a",): make_rope(), ("c",): twin})
    with pytest.raises(ValueError, match="exactly one"):
        PreRoPETransform.from_model_ropes(adapter)


def test_discovery_selects_component_specific_tables_with_equal_head_sizes(monkeypatch):
    from sglang.srt.layers.rotary_embedding import factory

    adapter = SimpleNamespace(
        kv=SimpleNamespace(head_dim=8),
        swa_kv=SimpleNamespace(head_dim=8),
        device=CPU,
        layer_ids=[0, 2],
        swa_layer_ids=[1, 3],
        page_size=4,
    )
    full = make_rope(base=10000)
    swa = make_rope(base=1000000)
    monkeypatch.setattr(factory, "_ROPE_DICT", {("full",): full, ("swa",): swa})
    full_transform = PreRoPETransform.from_model_ropes(
        adapter, component="full", selector={"base": 10000}
    )
    swa_transform = PreRoPETransform.from_model_ropes(
        adapter, component="swa", selector={"base": 1000000}
    )
    assert full_transform.table.description["class"] == "RotaryEmbedding"
    assert swa_transform.table.description["base"] == 1000000
    assert swa_transform.table.cos.shape == full_transform.table.cos.shape
    assert not torch.equal(swa_transform.table.cos, full_transform.table.cos)


def test_gemma4_proportional_selector_matches_the_padded_runtime_table():
    from sglang.srt.mem_cache.compression.linker import CompressionLinker

    config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            layer_types=["full_attention"],
            rope_parameters={
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000,
                    "partial_rotary_factor": 0.25,
                }
            },
        )
    )

    assert CompressionLinker._rope_selector(config, [0], 512) == {
        "base": 1_000_000,
        "class": "Gemma4RotaryEmbedding",
        "rotary_dim": 512,
    }


def test_gemma_rope_selectors_follow_full_and_sliding_layer_types():
    from sglang.srt.mem_cache.compression.linker import CompressionLinker

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.5,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                },
            },
        )
    )

    assert CompressionLinker._rope_selector(model_config, [2], 512) == {
        "base": 1000000.0,
        "rotary_dim": 512,
        "class": "Gemma4RotaryEmbedding",
    }
    assert CompressionLinker._rope_selector(model_config, [0, 1], 512) == {
        "base": 10000.0,
        "rotary_dim": 256,
        "class": "RotaryEmbedding",
    }
