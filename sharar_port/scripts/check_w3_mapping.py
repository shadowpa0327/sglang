"""Relative error of each W3 copy (dequantized) against the original 27B checkpoint weight: checks names and row order."""
import json, os, sys, torch
from safetensors import safe_open
sys.path.insert(0, "/data/ccchang/sglang-relax")
from sglang.srt.layers.quantization.sharar_w3 import w3
from sglang.srt.layers.quantization.sharar_w3.attach import Checkpoint

TGT, W3DIR = sys.argv[1], sys.argv[2]
layers = [int(x) for x in sys.argv[3].split(",")]
idx = json.load(open(f"{TGT}/model.safetensors.index.json"))["weight_map"]
P = "model.language_model."
def get(name):
    with safe_open(f"{TGT}/{idx[name]}", "pt", device="cuda") as f:
        return f.get_tensor(name)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
def orig(base):
    w = get(base + ".weight")
    if w.dtype == torch.float8_e4m3fn:
        return w.float() * get(base + ".weight_scale").float()
    # NVFP4: two codes a byte (low nibble first), E4M3 scale per 16, FP32 global scale_2
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    v = torch.stack([E2M1[lo], E2M1[hi]], -1).reshape(w.shape[0], -1)
    s = get(base + ".weight_scale").float().repeat_interleave(16, 1)
    return v * s * get(base + ".weight_scale_2").float()
ck = Checkpoint(W3DIR)
def deq(name):
    t, cb = ck.load(name)
    lv = w3._codebook(cb)
    W = w3.W3Weight(*(t[k].cuda() for k in ("planes", "scales", "row_scale")), t["row_scale"].shape[0], t["planes"].shape[1] * 128, lv, w3.lut_of(lv))
    return w3.dequantize(W)  # includes the row scale
def rel(a, b): return ((a - b).norm() / b.norm()).item()
for i in layers:
    L = f"{P}layers.{i}."
    if f"{L}linear_attn.in_proj_qkv.weight" in idx:
        pairs = {f"layers.{i}.mixer.in_qkv": [L + "linear_attn.in_proj_qkv"], f"layers.{i}.mixer.in_z": [L + "linear_attn.in_proj_z"],
                 f"layers.{i}.mixer.out": [L + "linear_attn.out_proj"]}
    else:
        pairs = {f"layers.{i}.mixer.qkv": [L + "self_attn.q_proj", L + "self_attn.k_proj", L + "self_attn.v_proj"],
                 f"layers.{i}.mixer.o": [L + "self_attn.o_proj"]}
    pairs[f"layers.{i}.mlp.gate_up"] = [L + "mlp.gate_proj", L + "mlp.up_proj"]
    pairs[f"layers.{i}.mlp.down"] = [L + "mlp.down_proj"]
    for name, bases in pairs.items():
        o = torch.cat([orig(b) for b in bases])
        d = deq(name)
        print(f"{name:28s} {tuple(d.shape)} rel_err={rel(d, o):.3f}", flush=True)
if os.environ.get("DEBUG"):
    name="layers.0.mlp.down"; o=orig(P+"layers.0.mlp.down_proj"); t,cb=ck.load(name)
    print("orig norm", o.norm().item(), "rs", t["row_scale"][:4], "scales", t["scales"].flatten()[:8])
    d=deq(name); print("deq norm", d.norm().item(), "corr", (d.flatten()@o.flatten()/(d.norm()*o.norm())).item())
