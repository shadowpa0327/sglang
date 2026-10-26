"""Re-fit Flash-Next's 3-bit trunk copies with sharar's GPTQ (w3_calibration.quantize) from SGLang-captured covariances
(attach.py `_Capture`: cov_<key>.pt, each the input covariance shared by the checkpoint matrices it lists).
Writes a sharar W3 checkpoint (format 1) the port serves with SGLANG_SHARAR_W3.
usage: fit_trunk.py CAPTURE_DIR OUTPUT_DIR [inputs.json]"""
import glob, hashlib, json, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from sharar.weights.w3_calibration import quantize
from sharar.weights.w3_checkpoint import FORMAT_VERSION, save_weight

CAP, OUT = Path(sys.argv[1]), Path(sys.argv[2])
INPUTS = sys.argv[3] if len(sys.argv) > 3 else None
SNAP = "/data/ccchang/hf_home/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
REV = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
P = "model.language_model."
idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    with safe_open(f"{SNAP}/{idx[name]}", "pt", device="cpu") as f:
        return f.get_tensor(name)


def source(name: str) -> torch.Tensor:
    """The BF16 checkpoint rows of sharar's matrix `name` (its row order: q|k|v, gate|up)."""
    if name == "lm_head":
        parts = ["lm_head.weight"]
    else:
        _, i, block, what = name.split(".", 3)
        L = f"{P}layers.{i}."
        parts = {"mixer.in_qkv": [L + "linear_attn.in_proj_qkv.weight"], "mixer.in_z": [L + "linear_attn.in_proj_z.weight"],
                 "mixer.out": [L + "linear_attn.out_proj.weight"],
                 "mixer.qkv": [L + f"self_attn.{p}_proj.weight" for p in "qkv"], "mixer.o": [L + "self_attn.o_proj.weight"],
                 "mlp.shared_gate_up": [L + "mlp.shared_expert.gate_proj.weight", L + "mlp.shared_expert.up_proj.weight"],
                 "mlp.shared_down": [L + "mlp.shared_expert.down_proj.weight"],
                 "ple.key_proj": [L + "ple.key_proj.weight"], "ple.value_proj": [L + "ple.value_proj.weight"]}[f"{block}.{what}"]
    return torch.cat([tensor(p) for p in parts]).float()


OUT.mkdir(parents=True, exist_ok=True)
manifest = {"format": FORMAT_VERSION, "target": "flashnext", "target_revision": REV,
            "source": "checkpoint BF16 trunk; GPTQ (sharar w3_calibration.quantize) on SGLang-captured covariances",
            "capture": json.loads((CAP / "capture.json").read_text()),
            "inputs_sha256": hashlib.sha256(open(INPUTS, "rb").read()).hexdigest() if INPUTS else None,
            "damping": 0.01, "weights": {}, "reports": {}}
started = time.perf_counter()
covs = sorted(glob.glob(str(CAP / "cov_*.pt")))
for j, path in enumerate(covs):
    c = torch.load(path)
    H = c["H"].cuda()
    for name in c["names"]:
        w = source(name).cuda()
        if w.shape[1] != H.shape[0]:
            raise SystemExit(f"{name}: weight K {w.shape[1]} vs covariance {H.shape[0]}")
        w3w, report = quantize(w, H, damping=0.01)
        report["rows"] = c["rows"]
        manifest["weights"][name] = save_weight(OUT / f"{name}.safetensors", w3w)
        manifest["reports"][name] = report
        del w, w3w
    del H
    print(f"  {j + 1}/{len(covs)} covariances fitted ({time.perf_counter() - started:.0f}s)", flush=True)
manifest["seconds"] = round(time.perf_counter() - started)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"done: {len(manifest['weights'])} matrices -> {OUT}", flush=True)
