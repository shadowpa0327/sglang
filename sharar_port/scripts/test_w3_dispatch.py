import sys, torch
sys.path.insert(0, "/data/ccchang/sglang-relax")
from sglang.srt.layers.quantization.sharar_w3 import w3
torch.manual_seed(0)
bad = 0
for n, k in [(8192, 2560), (2560, 3072), (6656, 2560), (640, 2560), (124160, 2560), (10240, 2560), (2560, 2560)]:
    p = w3.pack(torch.randn(n, k, device="cuda") * 0.02); ref_w = w3.dequantize(p)
    for m in (1, 3, 16, 17, 24, 33, 48, 63, 64, 65, 96, 127, 128):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        y = w3.w3_linear(x, p).float(); ref = x.float() @ ref_w.T
        e = ((y - ref).norm() / ref.norm()).item(); bad += e > 0.04
        if e > 0.04 or m in (1, 64, 128): print(n, k, m, f"{e:.4f}", w3._h100_choice(n, k, m, x.device))
    # static FP8 activations
    xq = (torch.randn(8, k, device="cuda") * 3).clamp(-448, 448).to(torch.float8_e4m3fn); s = torch.tensor([0.01], device="cuda")
    y = w3.w3_linear(xq, p, x_scale=s).float(); ref = (xq.float() * 0.01) @ ref_w.T
    print(n, k, "fp8-static", f"{((y - ref).norm() / ref.norm()).item():.2e}")
print("BAD", bad)
