"""Tokens per verify round for relaxed-acceptance arms on one running server, picked per request through custom_params
(accept_top_k / accept_soft), so every arm shares the batches. GPQA-Diamond prompts, thinking on, T=1.0 / top-p 0.95 /
top-k 20, arms interleaved. With argmax MTP drafts (default server) and sampled drafts (--speculative-use-rejection-sampling,
optionally SGLANG_DRAFT_TEMP_SCALE / SGLANG_DRAFT_TOPK) this compares both sides of the relaxed rule.
usage: python mtp_accept_arms.py --port 30000 --gpqa gpqa_diamond.jsonl --tokenizer <target> --out arms.jsonl
       [--arms exact,s0,s0.1,s0.25,s0.5,forced] [--n 40] [--max-new 3072] [--concurrency 16]
The server's SGLANG_DFLASH_RELAXED_TOP_K / _MARGIN set the plausible set (sharar fast with MTP: 3, 2.0); "forced" is that
rule with s = inf, "exact" turns it off for the request."""
import argparse, collections, concurrent.futures as cf, json, random

import requests
from transformers import AutoTokenizer

ARMS = {"exact": {"accept_top_k": 1}, "s0": {"accept_soft": 0.0}, "s0.1": {"accept_soft": 0.1},
        "s0.25": {"accept_soft": 0.25}, "s0.5": {"accept_soft": 0.5}, "forced": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gpqa", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    arms = a.arms.split(",")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.gpqa)][: a.n]

    def prompt(r):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        random.Random(r["Question"]).shuffle(opts)
        q = r["Question"].strip() + "\n\n" + "\n".join(f"{c}) {o.strip()}" for c, o in zip("ABCD", opts))
        q += "\n\nAnswer with the letter in \\boxed{}."
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)

    texts = [prompt(r) for r in rows]
    jobs = [(arm, i) for i in range(len(rows)) for arm in arms]
    random.Random(0).shuffle(jobs)

    def run(job):
        arm, i = job
        sp = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_new_tokens": a.max_new, "custom_params": ARMS[arm]}
        m = requests.post(f"http://127.0.0.1:{a.port}/generate", json={"text": texts[i], "sampling_params": sp},
                          timeout=3600).json()["meta_info"]
        return {"arm": arm, "i": i, "completion_tokens": m["completion_tokens"], "spec_verify_ct": m["spec_verify_ct"]}

    by = collections.defaultdict(list)
    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
        for res in ex.map(run, jobs):
            f.write(json.dumps(res) + "\n")
            by[res["arm"]].append(res)
    for arm in arms:
        rs = by[arm]
        per = [r["completion_tokens"] / r["spec_verify_ct"] for r in rs]
        m = sum(per) / len(per)
        se = (sum((x - m) ** 2 for x in per) / max(1, len(per) - 1) / len(per)) ** 0.5
        agg = sum(r["completion_tokens"] for r in rs) / sum(r["spec_verify_ct"] for r in rs)
        print(f"{arm:7s} n={len(rs):3d}  tokens/round {agg:.3f}  (per-request {m:.3f} ± {se:.3f})")


if __name__ == "__main__":
    main()
