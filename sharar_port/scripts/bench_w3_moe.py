"""Time sharar's 3-bit routed-experts kernel for one Flash-Next layer (this rank's 256 experts at TP2) on H100,
and check it against the torch reference."""
import sys, time, torch
sys.path.insert(0, "/data/ccchang/sglang-relax")
from sglang.srt.layers.quantization.sharar_w3.attach import Checkpoint, load_experts
from sglang.srt.layers.quantization.sharar_w3.w3_moe import w3_moe, reference
ck = Checkpoint("/data/ccchang/hf_home/hub/models--yemara--Qwen3.8-Flash-Next-NVFP4-sharar-w3/snapshots/694e2790772b36297b52bb1dffdc33281d766671")
ex = load_experts(ck, 0, 512, 2, 0, "cuda")  # rank 0 of TP2: experts 0..255
torch.manual_seed(0)
for m in (8, 16, 32, 64, 128, 256, 1024):
    x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(512, device="cuda")[:10] for _ in range(m)]).int()
    w = torch.softmax(torch.randn(m, 10, device="cuda"), -1)
    y = w3_moe(ex, x, ids, w)
    if m <= 16:
        mask = (ids < 256)
        ref = reference(ex, x, torch.where(mask, ids, 0), torch.where(mask, w, 0.0))
        print("err", ((y.float() - ref.float()).norm() / ref.float().norm()).item())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): w3_moe(ex, x, ids, w)
    torch.cuda.synchronize(); print(f"M={m:5d} w3_moe {(time.perf_counter()-t)/20*1e6:8.1f} us")
