"""Prefill every calibration sample once through the capture server (all-position logprobs, so the lm_head sees every
row), then trigger the save."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, inputs, cdir = sys.argv[1], sys.argv[2], sys.argv[3]
samples = json.load(open(inputs))["samples"]
def go(s):
    body = {"input_ids": s["input_ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 0}
    urllib.request.urlopen(urllib.request.Request(url + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600).read()
t = time.time()
with ThreadPoolExecutor(4) as ex:
    for i, _ in enumerate(ex.map(go, samples)):
        if (i + 1) % 25 == 0: print(f"{i + 1}/{len(samples)} samples ({time.time() - t:.0f}s)", flush=True)
open(f"{cdir}/SAVE", "w").write("save\n")
urllib.request.urlopen(urllib.request.Request(url + "/generate", json.dumps({"input_ids": samples[0]["input_ids"][:8], "sampling_params": {"max_new_tokens": 1}}).encode(), {"Content-Type": "application/json"})).read()
print("save triggered", flush=True)
