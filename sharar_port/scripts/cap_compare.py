"""Exact vs 3-bit per cap over three seeds (GPQA): seed accuracies, mean gap, capped counts, and an item-level
paired bootstrap of (mean correct over seeds, 3-bit) - (same, exact)."""
import glob, json, random, statistics, sys
BASE, CAND = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("fnx", "fnw")
R = "/data/ccchang/sharar_port/runs"
def load(tag):
    out = {}
    for f in sorted(glob.glob(f"{R}/{tag}/gpqa/s*.json")):
        x = json.load(open(f))
        for r in (x["items"] if isinstance(x, dict) else x):
            out[r["id"]] = r
    return out
print(f"| cap | {BASE} seeds | {BASE} mean | {CAND} seeds | {CAND} mean | gap [95% CI] | capped {BASE} / {CAND} (mean) | uncapped acc {BASE} / {CAND} |")
print("|---|---|---:|---|---:|---:|---:|---:|")
for cap in (16, 32, 64):
    E = [load(f"{BASE}-g{cap}k-s{s}") for s in range(3)]
    W = [load(f"{CAND}-g{cap}k-s{s}") for s in range(3)]
    ids = sorted(set.intersection(*(set(r) for r in E + W)))
    acc = lambda runs: [sum(r[i]["correct"] for i in ids) / len(ids) for r in runs]
    cap_n = lambda runs: statistics.mean(sum(bool(r[i].get("capped")) for i in ids) for r in runs)
    def unc(runs):
        pairs = [(r[i]["correct"]) for r in runs for i in ids if not r[i].get("capped")]
        return sum(pairs) / len(pairs)
    d = [statistics.mean(w[i]["correct"] for w in W) - statistics.mean(e[i]["correct"] for e in E) for i in ids]
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(4000))
    ea, wa = acc(E), acc(W)
    print(f"| {cap}K | {' / '.join(f'{a:.1%}' for a in ea)} | {statistics.mean(ea):.1%} | {' / '.join(f'{a:.1%}' for a in wa)} | "
          f"{statistics.mean(wa):.1%} | {statistics.mean(d):+.1%} [{boots[100]:+.1%}, {boots[3899]:+.1%}] | {cap_n(E):.0f} / {cap_n(W):.0f} | "
          f"{unc(E):.1%} / {unc(W):.1%} |")
