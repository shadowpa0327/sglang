"""Seed-to-seed noise of one arm vs the gap to another: accuracy, capped counts, paired deltas, length variation (GPQA)."""
import glob, itertools, json, math, statistics, sys
R = "/data/ccchang/sharar_port/runs"
def load(tag, task="gpqa"):
    out = {}
    for f in sorted(glob.glob(f"{R}/{tag}/{task}/s*.json")):
        x = json.load(open(f))
        for r in (x["items"] if isinstance(x, dict) else x):
            out[r["id"]] = r
    return out
def mcnemar(b, w):  # exact two-sided binomial p on the discordant pairs
    n = b + w
    if n == 0: return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
tags = sys.argv[1:]
runs = {t: load(t) for t in tags}
ids = sorted(set.intersection(*(set(r) for r in runs.values())))
print(f"items in all runs: {len(ids)}")
print("| run | accuracy | capped | median tokens | acc. when not capped |")
print("|---|---:|---:|---:|---:|")
for t, r in runs.items():
    acc = sum(r[i]["correct"] for i in ids) / len(ids)
    cap = sum(bool(r[i].get("capped")) for i in ids)
    unc = [i for i in ids if not r[i].get("capped")]
    print(f"| {t} | {acc:.1%} | {cap} | {statistics.median(r[i]['completion_tokens'] for i in ids):.0f} | {sum(r[i]['correct'] for i in unc)/len(unc):.1%} |")
print("\n| pair | delta | better/worse | p (McNemar) | items capped in exactly one |")
print("|---|---:|---:|---:|---:|")
for a, b in itertools.combinations(tags, 2):
    A, B = runs[a], runs[b]
    bt = sum(B[i]["correct"] and not A[i]["correct"] for i in ids); wr = sum(A[i]["correct"] and not B[i]["correct"] for i in ids)
    one = sum(bool(A[i].get("capped")) != bool(B[i].get("capped")) for i in ids)
    print(f"| {b} vs {a} | {(bt - wr) / len(ids):+.1%} | {bt}/{wr} | {mcnemar(bt, wr):.3f} | {one} |")
# per-item length variation across the runs
lens = {i: [runs[t][i]["completion_tokens"] for t in tags] for i in ids}
cv = [statistics.pstdev(v) / statistics.mean(v) for v in lens.values() if statistics.mean(v) > 0]
print(f"\nper-item token count across runs: median coefficient of variation {statistics.median(cv):.2f}; "
      f"items capped in some but not all runs: {sum(len({bool(runs[t][i].get('capped')) for t in tags}) > 1 for i in ids)}")
