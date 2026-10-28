"""Turn regenerated conversations into preformatted, last-turn-loss training rows.

    python sharar_port/draft_data_regen/scripts/build_regen_training_rows.py --tokenizer <target> --out-dir <dir> \
        [--budget 8192] [--decontaminate-against ./eval_datasets] \
        [--verify-chat-template qwen3.5] <regen.jsonl> [<regen.jsonl> ...]

Input: regen_driver.py output (rows with "conversations", optional
"reasoning_effort", "source" and per-answer "reasoning_content" / "truncated").

Each assistant turn becomes one row whose text is rendered with the target's own chat
template exactly as the server built that turn's generation prompt: the row's reasoning
effort (passed as --effort-kwarg), earlier answers without their reasoning (the driver
sends them that way), then the turn's own answer with its reasoning. Train these rows with

    data.is_preformatted: true
    data.train_only_last_turn: true
    data.chat_template: <SpecForge template matching the target>

so only that last answer carries loss and every generated token is trained once, with the
history the target saw. An answer cut at the regeneration budget ("truncated") is rendered
without closing tags: cut inside the reasoning it ends there, cut inside the answer it has
no end-of-turn token. How the template opens and closes reasoning is read off a probe
message, so any template that renders reasoning_content works.

Dropped: conversations whose user/system turns share a 13-word sequence with any string in
the --decontaminate-against files; answers stuck in a repetition loop (zlib ratio of the
last 4,000 characters < --loop-zlib; later turns of that conversation go too); rows longer
than --budget (+ 8 tokens for the end-of-turn the server does not count); turns whose
reasoning the template does not render.

--verify-chat-template NAME (needs SpecForge importable) parses a sample of the output with
SpecForge's preprocessing for that template and checks that the loss covers exactly the last
answer (from the end of its generation prompt to the end of the row). Run it once per new
target/template.

Output: <out-dir>/train.jsonl ({"id", "text", "source", "effort", "truncated"}, shuffled)
and <out-dir>/stats.json.
"""

import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

R_PROBE, C_PROBE = "REASONINGPROBE7351", "ANSWERPROBE7351"
END_SLACK = 8
tok = None
ARGS = None
DECON = None


def init():
    global tok
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ARGS.tokenizer, trust_remote_code=True)


def looping(text):
    tail = text[-4000:].encode()
    return len(text) >= 2000 and len(zlib.compress(tail)) / len(tail) < ARGS.loop_zlib


def template(messages, gen, kw):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=gen,
        enable_thinking=ARGS.thinking,
        **kw,
    )


def truncated_text(hist, prefix, reasoning, content, kw):
    """Prefix + how the template opens the reasoning + the reasoning [+ how it closes the
    reasoning + the partial answer], read off a probe answer rendered after the same history.
    """
    probe = template(
        hist
        + [{"role": "assistant", "content": C_PROBE, "reasoning_content": R_PROBE}],
        False,
        kw,
    )
    rest = probe[len(prefix) :] if probe.startswith(prefix) else None
    if rest is None or R_PROBE not in rest or C_PROBE not in rest:
        return None
    i_r, i_c = rest.index(R_PROBE), rest.index(C_PROBE)
    head, sep = rest[:i_r], rest[i_r + len(R_PROBE) : i_c]
    return prefix + head + reasoning + (sep + content if content else "")


def render(args):
    """-> (rows, drop reasons) for one regenerated conversation."""
    line, fallback_source = args
    row = json.loads(line)
    msgs = row["conversations"]
    source = row.get("source") or fallback_source
    if DECON is not None:
        why = DECON.hit([m for m in msgs if m["role"] != "assistant"])
        if why:
            return [], [f"eval overlap: {why}"]
    effort = row.get("reasoning_effort")
    kw = {ARGS.effort_kwarg: effort} if effort else {}
    out, drops, hist = [], [], []
    for m in msgs:
        if m["role"] != "assistant":
            hist.append({"role": m["role"], "content": m.get("content") or ""})
            continue
        reasoning, content = (m.get("reasoning_content") or "").strip(), (
            m.get("content") or ""
        ).strip()
        if looping(reasoning + "\n" + content):
            drops.append("repetition loop")
            break
        prefix = template(hist, True, kw)
        if m.get("truncated"):
            text = truncated_text(hist, prefix, reasoning, content, kw)
            if text is None:
                drops.append("template does not render reasoning_content")
                break
        else:
            turn = {"role": "assistant", "content": content}
            if reasoning:
                turn["reasoning_content"] = reasoning
            text = template(hist + [turn], False, kw)
            if not text.startswith(prefix):
                drops.append("rendered answer does not extend its generation prompt")
                break
            if reasoning and reasoning[:64] not in text[len(prefix) :]:
                drops.append("template does not render reasoning_content")
                break
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        n_prompt = len(tok(prefix, add_special_tokens=False)["input_ids"])
        if ARGS.budget and n_tok > ARGS.budget + END_SLACK:
            drops.append("over budget")
            break
        k = sum(1 for h in hist if h["role"] == "assistant")
        rid = str(row.get("id"))
        rid = rid if rid.startswith(f"{source}-") else f"{source}-{rid}"
        out.append(
            {
                "id": f"{rid}-t{k}",
                "text": text,
                "source": source,
                "effort": effort or "default",
                "truncated": bool(m.get("truncated")),
                "_tokens": n_tok,
                "_loss_tokens": n_tok - n_prompt,
                "_prefix_chars": len(prefix),
            }
        )
        hist.append({"role": "assistant", "content": content})
        if m.get("truncated"):
            break
    return out, drops


def verify(rows, template_name, n):
    """Parse a sample with SpecForge's preprocessing and compare its loss span with the
    rendered generation-prompt boundary."""
    try:
        from specforge.data.preprocessing import preprocess_conversations
        from specforge.data.template import TEMPLATE_REGISTRY
    except ImportError as e:
        raise SystemExit(
            "--verify-chat-template needs SpecForge importable (pip install -e <SpecForge> "
            f"or PYTHONPATH=<SpecForge>): {e}"
        )

    tmpl = TEMPLATE_REGISTRY.get(template_name)
    bad = []
    for r in rows[:n]:
        res = preprocess_conversations(
            tok,
            [r["text"]],
            tmpl,
            max_length=10**7,
            is_preformatted=True,
            train_only_last_turn=True,
        )
        mask = res["loss_mask"][0][0].tolist()
        n_prefix = len(
            tok(r["text"][: r["_prefix_chars"]], add_special_tokens=False)["input_ids"]
        )
        first = mask.index(1) if 1 in mask else -1
        span_ok = first >= 0 and all(mask[first:]) and not any(mask[:first])
        if not (
            span_ok
            and abs(first - n_prefix) <= 1
            and abs(sum(mask) - r["_loss_tokens"]) <= 2
        ):
            bad.append((r["id"], first, n_prefix, sum(mask), r["_loss_tokens"]))
    print(
        f"verify {template_name}: {len(bad)} of {min(n, len(rows))} sampled rows have a loss span that does "
        f"not match the last answer"
        + (
            f"; e.g. (id, loss start, prompt tokens, loss tokens, expected) {bad[:3]}"
            if bad
            else " (OK)"
        ),
        flush=True,
    )
    return len(bad)


def main():
    global ARGS, DECON
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="target model path or HF id (its chat template renders rows)",
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--budget",
        type=int,
        default=None,
        help="drop rows longer than this (the regeneration budget)",
    )
    ap.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="rows were regenerated with thinking off (render with enable_thinking=False)",
    )
    ap.add_argument("--effort-kwarg", default="reasoning_effort")
    ap.add_argument("--decontaminate-against", nargs="*", default=[], metavar="PATH")
    ap.add_argument("--loop-zlib", type=float, default=0.10)
    ap.add_argument(
        "--num-proc", type=int, default=max(1, min(96, (os.cpu_count() or 2) - 2))
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--verify-chat-template",
        default=None,
        metavar="NAME",
        help="SpecForge chat_template the run will use; checks the loss span on a sample",
    )
    ap.add_argument("--verify-samples", type=int, default=300)
    ARGS = ap.parse_args()
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    if ARGS.decontaminate_against:
        from download_prompt_sources import Decontaminator

        # Built before the pool forks, so workers share the sets and the str-hash seed.
        DECON = Decontaminator(ARGS.decontaminate_against)

    rows, stats, check = [], {}, []
    with mp.get_context("fork").Pool(ARGS.num_proc, initializer=init) as pool:
        for path in ARGS.inputs:
            source = Path(path).name.split("_")[0]
            # Split on "\n" only: splitlines() also breaks on U+2028 inside JSON strings.
            lines = [
                line for line in open(path, encoding="utf-8").read().split("\n") if line
            ]
            s = {
                "conversations": len(lines),
                "rows": 0,
                "tokens": 0,
                "loss_tokens": 0,
                "effort": collections.Counter(),
                "truncated": 0,
                "drops": collections.Counter(),
            }
            for out, drops in pool.imap(
                render, [(line, source) for line in lines], chunksize=64
            ):
                s["drops"].update(drops)
                for r in out:
                    s["rows"] += 1
                    s["tokens"] += r["_tokens"]
                    s["loss_tokens"] += r["_loss_tokens"]
                    s["effort"][r["effort"]] += 1
                    s["truncated"] += r["truncated"]
                    if len(check) < ARGS.verify_samples * 4:
                        check.append(dict(r))
                    for k in ("_tokens", "_loss_tokens", "_prefix_chars"):
                        r.pop(k)
                    rows.append(json.dumps(r, ensure_ascii=False))
            stats[Path(path).name] = s
            print(Path(path).name, json.dumps(s), flush=True)
    random.Random(ARGS.seed).shuffle(rows)
    with open(ARGS.out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")
    stats["total"] = {
        k: sum(s[k] for s in stats.values())
        for k in ("conversations", "rows", "tokens", "loss_tokens")
    }
    json.dump(stats, open(ARGS.out_dir / "stats.json", "w"), indent=1)
    print(json.dumps(stats["total"]), flush=True)
    if ARGS.verify_chat_template:
        init()
        random.Random(ARGS.seed).shuffle(check)
        # truncated rows first: they are the ones a template mismatch breaks
        check.sort(key=lambda r: not r["truncated"])
        sample = (
            check[: ARGS.verify_samples // 3]
            + check[-(ARGS.verify_samples - ARGS.verify_samples // 3) :]
        )
        if verify(sample, ARGS.verify_chat_template, ARGS.verify_samples):
            sys.exit(1)


if __name__ == "__main__":
    main()
