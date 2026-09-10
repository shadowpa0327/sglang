from pathlib import Path

import pytest
import torch

from kvcompress.loader import load_plugin
from kvcompress.store import BlockStore

PLUGIN = (
    Path(__file__).resolve().parents[4]
    / "examples/prefix_compression/packed_quant_plugin.py"
)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_size_exact_codebook_and_constant_rows(bits, dtype):
    plugin, _ = load_plugin(str(PLUGIN), {"bits": bits})
    # Every code is exactly representable; packing and reconstruction must be exact.
    codes = torch.arange(128).remainder(2**bits).to(dtype)
    source = torch.stack([codes, torch.zeros_like(codes), torch.full_like(codes, -3.5)])
    payload = plugin.compress({"key": source}, context={})
    assert payload.tensors["key/packed"].dtype == torch.uint8
    assert payload.tensors["key/packed"].numel() == source.numel() * bits // 8
    assert payload.tensor_bytes == source.numel() * bits // 8 + 3 * 8
    out = {"key": torch.empty_like(source)}
    plugin.decompress(
        payload, out=out, scratch=plugin.prepare_decompression(payload, out=out)
    )
    assert torch.equal(out["key"], source)


@pytest.mark.parametrize("bits", [2, 4])
def test_grouped_partial_restore_owns_packed_payload_and_bounds_error(bits):
    torch.manual_seed(5)
    plugin, identity = load_plugin(str(PLUGIN), {"bits": bits})
    store = BlockStore(plugin, identity, 4, store_bytes=1 << 20)
    source = torch.randn(4, 2, 8, 2, 128).bfloat16()
    original = source.clone()
    record = store.insert("k", {"key": source}, context={})
    source.fill_(float("nan"))
    reconstructed = store.reconstruct(["k"])
    actual = reconstructed.blocks[0]["key"][:2]
    expected = original[:2].float()
    step = (expected.amax(-1, keepdim=True) - expected.amin(-1, keepdim=True)) / (
        2**bits - 1
    )
    # Half a quantization step plus BF16 output rounding error.
    bound = step / 2 + expected.abs().amax(-1, keepdim=True) / 256
    assert torch.all((actual.float() - expected).abs() <= bound + 1e-6)
    assert reconstructed.measurement["pages"] == 4
    assert record["compressed_bytes"] < record["original_bytes"]


def test_invalid_input_and_configuration():
    with pytest.raises(ValueError, match="bits"):
        load_plugin(str(PLUGIN), {"bits": 3})
    plugin, _ = load_plugin(str(PLUGIN), {"bits": 4})
    with pytest.raises(ValueError, match="divisible"):
        plugin.compress({"key": torch.ones(2, 3)}, context={})
    with pytest.raises(ValueError, match="finite"):
        plugin.compress({"key": torch.full((2, 128), float("nan"))}, context={})
