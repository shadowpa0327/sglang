"""Calibration samples for re-fitting Flash-Next's 3-bit copies from the model's own thinking traces (SpecForge
perfect-blend regen, T=1.0): 70% from traces with >= 8K characters of reasoning, 30% at random, rendered with the
chat template (thinking on, the template's xhigh), cut at 4,096 tokens, about 1.2M tokens in all; items sharing a
13-gram with GPQA Diamond, LiveCodeBench or IFEval prompts are dropped. Writes sharar's {"samples": [{"input_ids"}]}."""
import glob, json, random, re, sys
from transformers import AutoTokenizer
SRC = "/data/ccchang/specforge_cache/dataset/perfectblend_train_regen_qwen38_think.jsonl"
TOK = "/data/ccchang/hf_home/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
DS = "/data/ccchang/sharar_port/sharar_results/datasets"
OUT, BUDGET, MAXLEN, SEED = sys.argv[1], 1_200_000, 4096, 20261026
words = lambda s: re.findall(r"[a-z0-9]+", s.lower())
def grams(s, n=13):
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
bench = set()
for l in open(f"{DS}/hard/gpqa_diamond.jsonl"):
    r = json.loads(l); bench |= grams(r["Question"] + " " + r["Correct Answer"])
for l in open(f"{DS}/hard/ifeval.jsonl"):
    bench |= grams(json.loads(l)["prompt"])
for l in open(f"{DS}/lcb/test6.jsonl"):
    bench |= grams(json.loads(l)["question_content"])
print("benchmark 13-grams", len(bench), flush=True)
rows = [json.loads(l) for l in open(SRC)]
reason_len = lambda d: sum(len(m.get("reasoning_content") or "") for m in d["conversations"])
long_ = [d for d in rows if reason_len(d) >= 8000]
rng = random.Random(SEED)
rng.shuffle(long_); rng.shuffle(rows)
pool = []
li = ri = 0
while li < len(long_) or ri < len(rows):  # 7 long : 3 random
    for _ in range(7):
        if li < len(long_): pool.append(long_[li]); li += 1
    for _ in range(3):
        if ri < len(rows): pool.append(rows[ri]); ri += 1
tok = AutoTokenizer.from_pretrained(TOK)
seen, samples, total, dropped = set(), [], 0, 0
for d in pool:
    if d["id"] in seen: continue
    seen.add(d["id"])
    text = " ".join((m.get("content") or "") + " " + (m.get("reasoning_content") or "") for m in d["conversations"])
    if grams(text) & bench:
        dropped += 1; continue
    msgs = [{"role": m["role"], "content": m["content"] or "", **({"reasoning_content": m["reasoning_content"]} if m.get("reasoning_content") else {})}
            for m in d["conversations"]]
    text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True)
    ids = tok(text, add_special_tokens=False)["input_ids"][:MAXLEN]
    if len(ids) < 64: continue
    samples.append({"id": d["id"], "input_ids": ids}); total += len(ids)
    if total >= BUDGET: break
json.dump({"source": SRC, "seed": SEED, "max_tokens": MAXLEN, "samples": samples}, open(OUT, "w"))
print(f"{len(samples)} samples, {total} tokens, {dropped} dropped for benchmark overlap, "
      f"{sum(len(s['input_ids']) == MAXLEN for s in samples)} cut at {MAXLEN} -> {OUT}", flush=True)
