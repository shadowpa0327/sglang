"""A BF16 copy of NVIDIA's Nemotron-3.5-Super LiLiCorr drafter (ModelOpt W4A16_NVFP4) for this SGLang, whose LiLiCorr
path builds unquantized layers for a LiLiCorrDraftModel config and refuses the NVFP4-packed weights: every quantized
linear (uint8 weight + FP8 per-16 block scale + FP32 global scale) is dequantized with SGLang's own dequantize_nvfp4
(ModelOpt's layout, low nibble first) and written in BF16; the scales and the quantization config are dropped. The
drafter is W4A16 (BF16 activations), so its arithmetic is unchanged up to rounding the weights to BF16.
usage: lilicorr_bf16.py <drafter dir> <out dir>"""
import json, shutil, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tensors = load_file(str(src / "model.safetensors"))
quantized = [k[: -len(".weight_scale_2")] for k in tensors if k.endswith(".weight_scale_2")]
# NVIDIA's names for the LiLiCorr head's sequential blocks -> this SGLang's nn.Sequential indices (models/lilicorr.py:
# feature_mlp = LayerNorm(5), Linear, SiLU, Linear; each layer's mlp = Linear, SiLU, Linear); every other name matches
RENAME = [("lilicorr.feature_norm.", "lilicorr.feature_mlp.0."), ("lilicorr.feature_mlp.up_proj.", "lilicorr.feature_mlp.1."),
          ("lilicorr.feature_mlp.down_proj.", "lilicorr.feature_mlp.3."), (".mlp.up_proj.", ".mlp.0."), (".mlp.down_proj.", ".mlp.2.")]


def rename(name: str) -> str:
    if not name.startswith("lilicorr."):
        return name
    for a, b in RENAME:
        if a in name:
            return name.replace(a, b)
    return name


result = {}
for name, t in tensors.items():
    base = name.rsplit(".", 1)[0]
    if base in quantized and name.rsplit(".", 1)[1] in ("weight_scale", "weight_scale_2", "input_scale"):
        continue  # the quantization scales of a quantized linear (its bias, if any, stays)
    if base in quantized and name.endswith(".weight"):
        w = dequantize_nvfp4(t, tensors[base + ".weight_scale"], tensors[base + ".weight_scale_2"], out_dtype=torch.bfloat16)
        assert w.shape[1] == t.shape[1] * 2, (name, tuple(t.shape), tuple(w.shape))
        result[rename(name)] = w.contiguous()
    else:
        result[rename(name)] = t
save_file(result, str(out / "model.safetensors"), metadata={"format": "pt"})
config = json.loads((src / "config.json").read_text())
config.pop("quantization_config", None)
(out / "config.json").write_text(json.dumps(config, indent=2))
for f in src.iterdir():
    if f.name not in ("model.safetensors", "config.json", "hf_quant_config.json") and f.is_file():
        shutil.copy(f, out / f.name)
print(f"{len(quantized)} linears dequantized to BF16, {len(result)} tensors -> {out}")
for k in [rename(q + ".weight")[: -len(".weight")] for q in quantized[:3]] + ["lm_head", "lilicorr.context_proj"]:
    if k + ".weight" in result:
        w = result[k + ".weight"].float()
        print(f"  {k}: {tuple(w.shape)} mean |w| {w.abs().mean():.4g}, max |w| {w.abs().max():.4g}")
