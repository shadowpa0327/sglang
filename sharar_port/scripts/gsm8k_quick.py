"""Paired GSM8K check: greedy, thinking off, first N test items; prints accuracy and writes per-item results."""
import glob, json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
url, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
rows = pq.read_table(glob.glob("/data/ccchang/hf_home/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")[0]).to_pylist()[:n]
def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None
def ask(r):
    body = {"model": "x", "messages": [{"role": "user", "content": r["question"] + "\nPlease reason step by step, and put your final answer after '####'."}],
            "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    resp = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    text = resp["choices"][0]["message"]["content"] or ""
    gold = r["answer"].split("####")[-1].strip().replace(",", "")
    pred = num(text.split("####")[-1]) if "####" in text else num(text)
    try: ok = pred is not None and float(pred) == float(gold)
    except ValueError: ok = False
    return {"ok": ok, "pred": pred, "gold": gold, "tokens": resp["usage"]["completion_tokens"], "text": text}
t = time.time()
with ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, rows))
json.dump(res, open(out, "w"))
tok = sum(r["tokens"] for r in res)
print(f"{out}: acc {sum(r['ok'] for r in res)}/{n} = {sum(r['ok'] for r in res)/n:.3f}  {tok/(time.time()-t):.0f} tok/s aggregate")
