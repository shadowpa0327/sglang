import json, sys, time, urllib.request
url = sys.argv[1]; think = len(sys.argv) > 2 and sys.argv[2] == "think"
q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer with just the number at the end."
body = {"model": "x", "messages": [{"role": "user", "content": q}], "max_tokens": 2048 if think else 512, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": think}}
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})))
dt = time.time() - t
m = r["choices"][0]["message"]; n = r["usage"]["completion_tokens"]
print(repr((m.get("content") or "")[-300:]), "| reasoning chars", len(m.get("reasoning_content") or ""), f"| {n} tok {n/dt:.1f} tok/s")
