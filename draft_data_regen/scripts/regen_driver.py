"""Regenerate the assistant turns of a prompt file with the target model, at scale.

    python draft_data_regen/scripts/regen_driver.py <input.jsonl> <output.jsonl> <host:port> [<host:port> ...] \
        --model <target path or HF id> [--thinking --budget 8192 --effort-mix low:0.4,medium:0.3,xhigh:0.3]

Builds on SpecForge's scripts/regenerate_train_data.py (helpers vendored in regen_common.py; same request format and row schema: one chat
call per user turn, answers replace the source's assistant turns) and adds what long
regeneration runs on shared servers need:

- Resume by input line. Every finished row carries "src_line"; rerunning the same command
  skips lines already in <output>, <output>_error.jsonl or <output>_skipped.jsonl. Rows are
  flushed one by one, so a killed driver loses only the requests in flight.
- Several servers, least-loaded first, health-probed every 10 s. Connection failures,
  timeouts, 5xx answers and aborted requests (finish_reason "abort" when a server restarts)
  are retried on whichever server is healthy, up to 30 times. Only other 4xx answers and
  invalid inputs are final (they go to the _error / _skipped files).
- Multi-turn rows keep the turns answered before a later turn outgrew the context.
- --thinking saves each answer's reasoning_content (the server needs a --reasoning-parser).
  An answer that does not finish, or leaks think markers, ends the conversation at the
  previous turn (the row is skipped if it was the first turn).
- --budget N (thinking only) gives each turn N - prompt tokens of answer, so every turn
  fits an N-token training row. An answer cut at that limit is kept with "truncated": true
  and ends the conversation; build_regen_training_rows.py renders it without
  closing tags so the draft never learns a false end of turn. A turn with under 256
  tokens of room ends the conversation.
- --effort-mix gives each conversation one reasoning effort (passed to the chat template
  as --effort-kwarg) drawn from the given shares, seeded by input line so resumes agree;
  the output row records it as "reasoning_effort".

Sampling defaults (temperature 1.0, top-p 0.95, top-k 20) are what the Qwen3.8-Flash-Next
data used. A draft should see the target's real serving distribution, so set them to the
sampling the target will be served with (its model card's recommendation).
"""

import argparse
import concurrent.futures as cf
import copy
import functools
import json
import random
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai import OpenAI  # noqa: E402

import regen_common as regen  # noqa: E402

# A long thinking answer at ~10 tok/s per request outlives the client's 600 s default,
# and the client's own retries would silently resubmit it.
regen.OpenAI = functools.partial(OpenAI, timeout=3 * 3600, max_retries=0)

MAX_ATTEMPTS = 30
CONTEXT_ERRORS = ("maximum context length", "longer than the model's context length")
MIN_ANSWER_TOKENS = 256

_tokenizer = None
_tokenizer_lock = threading.Lock()


def prompt_tokens(args, messages, effort=None):
    """Tokens of the generation prompt the server builds for these messages (earlier answers
    without their reasoning, as build_query_kwargs sends them)."""
    global _tokenizer
    history = [
        {k: v for k, v in m.items() if k not in ("reasoning_content", "truncated")}
        for m in messages
    ]
    with _tokenizer_lock:
        if _tokenizer is None:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True
            )
        kwargs = {args.effort_kwarg: effort} if effort else {}
        ids = _tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            enable_thinking=args.reasoning == "save",
            tokenize=True,
            return_dict=False,
            **kwargs,
        )
    return len(ids)


def done_lines(*paths):
    seen = set()
    for path in paths:
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    seen.add(json.loads(line)["src_line"])
    return seen


class Servers:
    def __init__(self, addrs):
        self.addrs = addrs
        self.healthy = {a: True for a in addrs}
        self.inflight = {a: 0 for a in addrs}
        self.lock = threading.Lock()
        threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self):
        while True:
            for a in self.addrs:
                try:
                    ok = (
                        requests.get(f"http://{a}/health", timeout=10).status_code
                        == 200
                    )
                except requests.RequestException:
                    ok = False
                self.healthy[a] = ok
            time.sleep(10)

    def pick(self):
        """Least-in-flight healthy server, so a faster one gets more work."""
        while True:
            with self.lock:
                live = [a for a in self.addrs if self.healthy[a]]
                if live:
                    a = min(live, key=self.inflight.get)
                    self.inflight[a] += 1
                    return a
            time.sleep(10)

    def release(self, a):
        with self.lock:
            self.inflight[a] -= 1


def call_keep_turns(args, server_address, data):
    """regen.call_sglang, except that a turn that no longer fits ends the conversation there
    instead of failing the whole row, plus the thinking budget and effort handling above.
    """
    client = regen.OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    messages = data["conversations"]
    effort = data.get("reasoning_effort")
    out = []
    for message in messages:
        if message["role"] == "system":
            out.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            out.append(message)
            max_tokens = None
            if args.budget is not None:
                max_tokens = min(
                    args.max_tokens, args.budget - prompt_tokens(args, out, effort)
                )
                if max_tokens < MIN_ANSWER_TOKENS:
                    out.pop()
                    if any(m["role"] == "assistant" for m in out):
                        break
                    return regen.set_skipped(
                        data,
                        f"Prompt leaves under {MIN_ANSWER_TOKENS} tokens of the budget",
                    )
            try:
                query = regen.build_query_kwargs(args, out, max_tokens)
                if effort:
                    query.setdefault("extra_body", {}).setdefault(
                        "chat_template_kwargs", {}
                    )[args.effort_kwarg] = effort
                resp = client.chat.completions.create(**query)
            except Exception as e:
                if any(s in str(e) for s in CONTEXT_ERRORS) and any(
                    m["role"] == "assistant" for m in out
                ):
                    out.pop()
                    break
                data["status"] = "error"
                data["error"] = str(e)
                return data
            msg = resp.choices[0].message
            text = msg.content
            finish = resp.choices[0].finish_reason
            if args.reasoning == "save":
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning is None:
                    reasoning = (getattr(msg, "model_extra", None) or {}).get(
                        "reasoning_content"
                    )
                if finish not in ("stop", "length"):
                    # e.g. "abort" when the server shuts down mid-request: retry, never skip
                    data["status"] = "error"
                    data["error"] = f"finish_reason={finish}"
                    return data
                if (
                    args.budget is not None
                    and finish == "length"
                    and isinstance(reasoning, str)
                    and reasoning.strip()
                    and not regen.has_think_marker(reasoning)
                    and not regen.has_think_marker(text or "")
                ):
                    out.append(
                        {
                            "role": "assistant",
                            "content": text or "",
                            "reasoning_content": reasoning,
                            "truncated": True,
                        }
                    )
                    break
                if (
                    finish != "stop"
                    or not isinstance(text, str)
                    or not text.strip()
                    or not isinstance(reasoning, str)
                    or not reasoning.strip()
                    or regen.has_think_marker(text)
                    or regen.has_think_marker(reasoning)
                ):
                    out.pop()
                    if any(m["role"] == "assistant" for m in out):
                        break
                    return regen.set_skipped(
                        data,
                        "Thinking answer did not finish within max_tokens or is malformed",
                    )
                out.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "reasoning_content": reasoning,
                    }
                )
                continue
            if finish not in ("stop", "length"):
                data["status"] = "error"
                data["error"] = f"finish_reason={finish}"
                return data
            if (
                finish != "stop"
                or not isinstance(text, str)
                or not text.strip()
                or regen.has_think_marker(text)
            ):
                out.pop()
                if any(m["role"] == "assistant" for m in out):
                    break
                return regen.set_skipped(
                    data,
                    "Answer did not finish within max_tokens, is empty or contains a think marker",
                )
            out.append({"role": "assistant", "content": text})
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = out
    data["status"] = "success"
    return data


def effort_for(args, idx):
    """Reasoning effort of input line idx under --effort-mix (deterministic, so resumes agree)."""
    if not args.effort_mix:
        return None
    u = random.Random(f"effort-{idx}").random()
    acc = 0.0
    for name, share in args.effort_mix:
        acc += share
        if u < acc:
            return name
    return args.effort_mix[-1][0]


def work(args, servers, idx, row):
    out = None
    effort = effort_for(args, idx)
    if effort:
        row["reasoning_effort"] = effort
    for attempt in range(MAX_ATTEMPTS):
        addr = servers.pick()
        try:
            out = call_keep_turns(args, addr, copy.deepcopy(row))
        finally:
            servers.release(addr)
        status, err = out.get("status"), str(out.get("error", ""))
        if status in ("success", "skipped") or "Error code: 4" in err:
            break
        time.sleep(min(60, 5 * (attempt + 1)))
    out["src_line"] = idx
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("servers", nargs="+", help="host:port of each SGLang server")
    ap.add_argument(
        "--model",
        required=True,
        help="target model path or HF id (request model name and tokenizer for budgets)",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="thinking on: save reasoning_content (the server needs --reasoning-parser)",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=4096, help="new tokens per answer"
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=None,
        help="thinking only: prompt + answer tokens per turn; answers cut at it are kept as "
        "truncated (e.g. 8192 for an 8K training stage)",
    )
    ap.add_argument(
        "--effort-mix",
        default=None,
        help="thinking only: reasoning-effort shares per conversation, e.g. low:0.4,medium:0.3,xhigh:0.3",
    )
    ap.add_argument(
        "--effort-kwarg",
        default="reasoning_effort",
        help="chat-template kwarg that takes the effort (default: reasoning_effort)",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="requests in flight per server; match the server's --max-running-requests",
    )
    ap.add_argument(
        "--max-line",
        type=int,
        default=None,
        help="only process input lines below this index",
    )
    a = ap.parse_args()
    if (a.budget or a.effort_mix) and not a.thinking:
        ap.error("--budget and --effort-mix need --thinking")
    args = argparse.Namespace(
        model=a.model,
        reasoning="save" if a.thinking else "disable",
        is_gpt_oss=False,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        repetition_penalty=None,
        max_tokens=a.max_tokens,
        budget=a.budget,
        effort_kwarg=a.effort_kwarg,
        effort_mix=(
            [(k, float(v)) for k, v in (x.split(":") for x in a.effort_mix.split(","))]
            if a.effort_mix
            else None
        ),
    )
    err_path = a.output.replace(".jsonl", "_error.jsonl")
    skip_path = a.output.replace(".jsonl", "_skipped.jsonl")
    seen = done_lines(a.output, err_path, skip_path)
    total = sum(1 for _ in open(a.input, encoding="utf-8"))
    if a.max_line is not None:
        total = min(total, a.max_line)
        seen = {s for s in seen if s < total}
    print(f"{total} input rows, {len(seen)} already done", flush=True)

    servers = Servers(a.servers)
    outs = {
        "success": open(a.output, "a", encoding="utf-8"),
        "error": open(err_path, "a", encoding="utf-8"),
        "skipped": open(skip_path, "a", encoding="utf-8"),
    }
    counts = dict.fromkeys(outs, 0)
    t0 = last = time.time()
    cap = a.concurrency * len(a.servers)
    with cf.ThreadPoolExecutor(cap) as pool, open(a.input, encoding="utf-8") as f:
        inflight = set()

        def drain(block):
            nonlocal inflight, last
            done, inflight = cf.wait(
                inflight, return_when=cf.FIRST_COMPLETED if block else cf.ALL_COMPLETED
            )
            for fut in done:
                rec = fut.result()
                kind = rec.get("status") if rec.get("status") in outs else "error"
                if kind == "success":
                    rec = {
                        "id": rec.get("id"),
                        "conversations": rec["conversations"],
                        "src_line": rec["src_line"],
                        **({"source": rec["source"]} if rec.get("source") else {}),
                        **(
                            {"reasoning_effort": rec["reasoning_effort"]}
                            if rec.get("reasoning_effort")
                            else {}
                        ),
                    }
                outs[kind].write(json.dumps(rec, ensure_ascii=False) + "\n")
                outs[kind].flush()
                counts[kind] += 1
            if time.time() - last > 60:
                last = time.time()
                n = sum(counts.values())
                rate = n / (last - t0)
                left = total - len(seen) - n
                print(
                    f"[{time.strftime('%H:%M:%S')}] {counts} total_done={len(seen) + n}/{total} "
                    f"{rate:.2f}/s eta={left / max(rate, 1e-9) / 3600:.1f}h "
                    f"healthy={sum(servers.healthy.values())}/{len(a.servers)} inflight={servers.inflight}",
                    flush=True,
                )

        for idx, line in enumerate(f):
            if a.max_line is not None and idx >= a.max_line:
                break
            if idx in seen:
                continue
            row = json.loads(line)
            bad = regen.validate_regen_input(row)
            if bad is not None:
                rec = regen.set_skipped(row, bad)
                rec["src_line"] = idx
                outs["skipped"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["skipped"] += 1
                continue
            while len(inflight) >= cap:
                drain(block=True)
            inflight.add(pool.submit(work, args, servers, idx, row))
        while inflight:
            drain(block=False)
    print(f"done: {counts}", flush=True)


if __name__ == "__main__":
    main()
