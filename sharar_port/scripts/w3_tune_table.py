"""Sweep the port's W3 kernels (narrow splits, wide configs x splits) on the local GPU with L2-cold weights and write
the dispatch table w3.py reads for this arch (`<arch>_tuning.json`: "NxK" -> {row bucket: choice}).
usage: w3_tune_table.py OUT.json [shapes NxK,...] [buckets m,...]"""
import json, os, sys, torch
sys.path.insert(0, os.environ.get("PYTHONPATH_FORK", "/data/ccchang/sglang-relax"))
from sglang.srt.layers.quantization.sharar_w3 import w3

WIDE_CFGS = [2144, 2244, 4124, 4142, 4143, 4144, 4243, 8144, 8183]  # w3_gemm.cu W3_WIDE instances
SHAPES = [tuple(map(int, s.split("x"))) for s in (sys.argv[2] if len(sys.argv) > 2 else ",".join(w3._TUNED_SHAPES)).split(",")]
BUCKETS = [int(m) for m in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4,8,16,24,32,48,64,96,128").split(",")]
L2 = torch.cuda.get_device_properties(0).L2_cache_size
mod = w3._module()


def bench(fn, iters=50):
    for i in range(3):
        fn(i)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


table = {}
for n, k in SHAPES:
    torch.manual_seed(0)
    p0 = w3.pack(torch.randn(n, k, device="cuda") * 0.02)
    nbytes = p0.planes.numel() * 4 + p0.scales.numel()
    copies = int(2 * L2 // nbytes) + 1  # cycle through twice the L2
    ps = [p0] + [w3.W3Weight(*(t.clone() for t in (p0.planes, p0.scales, p0.row_scale)), p0.out_features, p0.in_features,
                             p0.codebook, p0.lut) for _ in range(copies - 1)]
    chunks = k // w3.CHUNK
    row = {}
    for m in BUCKETS:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        xq, xs = w3.quant_groups(x)
        ref = (xq.float() * xs.repeat_interleave(32, dim=1)) @ w3.dequantize(p0).T
        res = []

        def ok(y):
            return ((y.float() - ref).norm() / ref.norm()).item() < 1e-2

        if m <= w3.NARROW_ROWS:
            for sp in (1, 2, 4, 8, 16):
                if chunks % sp:
                    continue
                ws, cnt = w3._workspace(x.device, sp * m * n if sp > 1 else 0, n // 128)
                f = lambda i, sp=sp, ws=ws, cnt=cnt: mod.w3_gemm(xq, xs, ps[i % copies].planes, ps[i % copies].scales,
                                                                 ps[i % copies].row_scale, p0.lut, sp, ws, cnt)
                if ok(f(0)):
                    res.append((bench(f), ["narrow", sp]))
        for cfg in WIDE_CFGS:
            mt, wm, wn = cfg // 1000, cfg // 100 % 10, cfg // 10 % 10
            if n % (wn * 32):
                continue
            for sp in (1, 2, 4, 8):
                if chunks % sp:
                    continue
                ws, cnt = w3._workspace(x.device, sp * m * n if sp > 1 else 0, (n // (wn * 32)) * -(-m // (wm * mt * 16)))
                f = lambda i, cfg=cfg, sp=sp, ws=ws, cnt=cnt: mod.w3_gemm_wide(
                    xq, xs, ps[i % copies].planes, ps[i % copies].scales, ps[i % copies].row_scale, p0.lut, cfg, sp, ws, cnt)
                if ok(f(0)):
                    res.append((bench(f), ["wide", cfg, sp]))
        res.sort(key=lambda r: r[0])
        old = bench(lambda i: w3.w3_linear(xq, ps[i % copies], xs))  # the current dispatch (table or fallback)
        row[str(m)] = res[0][1]
        print(f"{n}x{k} M={m:3d} best {res[0][0]:7.1f}us {res[0][1]}  current dispatch {old:7.1f}us", flush=True)
    table[f"{n}x{k}"] = row
json.dump(table, open(sys.argv[1], "w"), indent=0, sort_keys=True)
print("wrote", sys.argv[1])
