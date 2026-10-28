"""Probe a Nemotron-3.5-Super server against NVIDIA's eval settings (NeMo Gym nemotron_recipes/super-3.5): thinking on via
chat_template_kwargs, the nemotron_3 reasoning parser splitting the thinking into reasoning_content, temperature 1.0 /
top-p 0.95 with top-k off (-1), and an output cap of 248K (253,952 tokens) inside the 262,144 context.
usage: nemotron_probe.py URL"""
import json, sys, urllib.error, urllib.request

url = sys.argv[1]
Q = ("A rectangle has perimeter 30 and area 56. What is the length of its diagonal? "
     "Please reason step by step, and put your final answer within \\boxed{}.")


def chat(max_tokens, think=True, **extra):
    body = {"model": "x", "messages": [{"role": "user", "content": Q}], "max_tokens": max_tokens, "temperature": 1.0,
            "top_p": 0.95, "seed": 0, "chat_template_kwargs": {"enable_thinking": think}, **extra}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=1200)), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:300]}"


def show(tag, resp, err):
    if err:
        print(f"{tag}: ERROR {err}")
        return
    m = resp["choices"][0]["message"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    content = m.get("content") or ""
    print(f"{tag}: finish={resp['choices'][0]['finish_reason']} completion_tokens={resp['usage']['completion_tokens']} "
          f"reasoning_chars={len(reasoning)} content_chars={len(content)} think_tags_in_content={'<think>' in content or '</think>' in content}")
    print(f"   content tail: {content[-120:]!r}")


show("thinking on, cap 248K, top_k -1", *chat(253952, top_k=-1))
show("thinking on, cap 248K, top_k 20", *chat(253952, top_k=20))
show("thinking off, cap 1024", *chat(1024, think=False, top_k=-1))
show("cap 262144 (prompt + cap > context)", *chat(262144, top_k=-1))
