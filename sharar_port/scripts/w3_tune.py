"""Sweep W3 kernel split/wide configurations on H100 with L2-cold weights; print the best per (N, K, M)."""
import os, sys, json, torch
os.environ.setdefault("SHARAR_CUDA_ARCH", "9.0a")
sys.path.insert(0, "/home/ccchang/sharar")
from sharar.kernels import w3
from pathlib import Path
if os.environ.get('W3_SRC'): w3._SRC = Path(os.environ['W3_SRC'])

dev = "cuda"
WIDE_CFGS = [2144, 2244, 4124, 4142, 4143, 4144, 4243, 8144, 8183]
SHAPES = [tuple(map(int, s.split("x"))) for s in sys.argv[1].split(",")]
MS = [int(m) for m in sys.argv[2].split(",")]
mod = w3._module()

def bench(fn, iters=30):
    for _ in range(3): fn(0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

out = {}
for n, k in SHAPES:
    W = torch.randn(n, k, device=dev) * 0.02
    p0 = w3.pack(W)
    nbytes = p0.planes.numel() * 4 + p0.scales.numel()
    copies = max(1, int(3 * 50e6 // nbytes) + 1)  # cycle > L2
    ps = [p0] + [w3.W3Weight(**{f: (getattr(p0, f).clone() if torch.is_tensor(getattr(p0, f)) else getattr(p0, f)) for f in p0.__dataclass_fields__}) for _ in range(copies - 1)]
    Wb = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) for _ in range(max(1, int(3 * 50e6 // (n * k * 2)) + 1))]
    chunks = k // 128
    for m in MS:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        xq, xs = w3.quant_groups(x)
        ref = w3.w3_linear(xq, p0, xs, splits=1).float() if m <= 64 else None
        res = []
        if m <= 64:
            for sp in (1, 2, 4, 8, 16):
                if chunks % sp: continue
                ws, cnt = w3._workspace(x.device, sp * m * n if sp > 1 else 0, n // 128)
                f = lambda i: mod.w3_gemm(xq, xs, ps[i % copies].planes, ps[i % copies].scales, ps[i % copies].row_scale, p0.lut, sp, ws, cnt)
                res.append((bench(f), "narrow", sp))
        for cfg in WIDE_CFGS:
            mt, wm, wn = cfg // 1000, cfg // 100 % 10, cfg // 10 % 10
            if n % (wn * 32): continue
            for sp in (1, 2, 4, 8):
                if chunks % sp: continue
                bm = wm * mt * 16
                ws, cnt = w3._workspace(x.device, sp * m * n if sp > 1 else 0, (n // (wn * 32)) * -(-m // bm))
                f = lambda i: mod.w3_gemm_wide(xq, xs, ps[i % copies].planes, ps[i % copies].scales, ps[i % copies].row_scale, p0.lut, cfg, sp, ws, cnt)
                try:
                    y = f(0)
                    if ref is not None and ((y.float() - ref).norm() / ref.norm()).item() > 1e-2:
                        print("MISMATCH", n, k, m, cfg, sp); continue
                    res.append((bench(f), "wide", cfg, sp))
                except Exception as ex:
                    pass
        bf = bench(lambda i: x @ Wb[i % len(Wb)].T)
        res.sort()
        best = res[0]
        default = bench(lambda i: w3.w3_linear(xq, ps[i % copies], xs))
        print(f"N={n:6d} K={k:6d} M={m:4d} best={best[0]:7.1f}us {best[1:]}  default={default:7.1f}us  bf16={bf:7.1f}us  "
              f"w3 GB/s={nbytes/best[0]/1e3:6.0f}  nvfp4-equiv@bf16bw={bf*0.5625/2:6.1f}us", flush=True)
        out[f"{n}x{k}x{m}"] = best
json.dump(out, open(sys.argv[3], "w"), indent=0)
