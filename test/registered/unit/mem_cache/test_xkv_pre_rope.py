from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from kvcompress.loader import load_plugin
from kvcompress.store import BlockStore
from sglang.srt.mem_cache.compression.pre_rope import (
    NativeRoPEDecodePlugin,
    PreRoPETransform,
    RotaryTable,
    resolve_key_space,
)

PLUGIN = (
    Path(__file__).resolve().parents[4]
    / "examples/prefix_compression/xkv_svd_plugin.py"
)
CPU = torch.device("cpu")


def make_rope(neox=True, head=8, rotary=8, max_pos=4096):
    return RotaryEmbedding(head, rotary, max_pos, 10000, neox, torch.bfloat16)


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
    tensors = {"key": pre, "value": native.clone()}
    store.insert("k", tensors, context=context)
    decoded = store.reconstruct(["k"])
    restored = decoded.blocks[0]["key"][:3]
    eps = torch.finfo(torch.bfloat16).eps
    tolerance = 2 * eps * native.float().abs().max()
    assert (restored.float() - native[:3].float()).abs().max() <= tolerance
    assert torch.equal(decoded.blocks[0]["value"][:3], native[:3])
    # A wrong absolute frame must be visible: claim the keys sit 8 tokens later.
    shifted = BlockStore(wrapper, identity, 4, store_bytes=1 << 20)
    later = {**context, "start_position": offset + 8}
    shifted.insert("k", tensors, context=later)
    decoded = shifted.reconstruct(["k"])
    wrong = decoded.blocks[0]["key"][:3]
    assert (wrong.float() - native[:3].float()).abs().max() > 10 * tolerance


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


def test_key_space_follows_the_plugin_unless_config_overrides():
    assert resolve_key_space(None, None) == ("post_rope", "default")
    assert resolve_key_space("auto", None) == ("post_rope", "default")
    assert resolve_key_space("auto", "pre_rope") == ("pre_rope", "plugin")
    assert resolve_key_space("pre_rope", None) == ("pre_rope", "config")
    assert resolve_key_space("pre_rope", "pre_rope") == ("pre_rope", "config")
    override = resolve_key_space("post_rope", "pre_rope")
    assert override == ("post_rope", "config_override")
    with pytest.raises(ValueError, match="auto, pre_rope or post_rope"):
        resolve_key_space("sideways", None)
    with pytest.raises(ValueError, match="Plugin key_space"):
        resolve_key_space("auto", "sideways")
    svd, _ = load_plugin(str(PLUGIN), {"rank_k": 2, "rank_v": 2})
    assert svd.key_space == "pre_rope"
    assert load_plugin("identity")[0].key_space is None
    assert load_plugin("int8")[0].key_space is None


def test_discovery_requires_exactly_one_live_table_for_the_pool_head_size(monkeypatch):
    from sglang.srt.layers.rotary_embedding import factory

    adapter = SimpleNamespace(
        kv=SimpleNamespace(head_dim=8), device=CPU, layer_ids=[0, 1], page_size=4
    )
    monkeypatch.setattr(factory, "_ROPE_DICT", {})
    with pytest.raises(ValueError, match="exactly one"):
        PreRoPETransform.from_model_ropes(adapter)
    other = make_rope(head=16, rotary=16)
    monkeypatch.setattr(factory, "_ROPE_DICT", {("a",): make_rope(), ("b",): other})
    transform = PreRoPETransform.from_model_ropes(adapter)
    assert transform.table.head_size == 8 and transform.layers == 2
    assert transform.table.description["class"] == "RotaryEmbedding"
    twin = make_rope(neox=False)
    monkeypatch.setattr(factory, "_ROPE_DICT", {("a",): make_rope(), ("c",): twin})
    with pytest.raises(ValueError, match="exactly one"):
        PreRoPETransform.from_model_ropes(adapter)
