"""Turn Open-SWE-Traces agent sessions into preformatted training rows.

    python draft_data_regen/scripts/download_prompt_sources.py --swe-traces qwen38_flash_next --output-dir ./cache/dataset/prompts
    python draft_data_regen/scripts/build_swe_trace_rows.py \
        --data-dir ./cache/dataset/prompts/open_swe_traces --model qwen38_flash_next \
        --tokenizer <target> --out-dir ./cache/dataset/agent_32k --max-tokens 32768 \
        [--verify-chat-template qwen3.5]

nvidia/Open-SWE-Traces sessions are not regenerated: their tool results follow from the
original model's actions. They are on-policy only when the trajectory model (--model) is
the draft's target, so use the matching model directory (qwen38_flash_next for
Qwen3.8-Flash-Next, qwen38_27b for Qwen3.8-27B, ...).

Each session (system, user, assistant turns with tool calls, tool results; tool
definitions) is rendered once with the target's chat template, every turn's reasoning
kept, so one row trains every assistant turn: train it with

    data.is_preformatted: true
    data.train_only_last_turn: false
    data.max_length: <--max-tokens>

Sessions recorded at a lower reasoning effort sit in thinking-<effort>/ subdirectories (the
top-level files are the template's default, xhigh). Their system prompts do not carry the
effort, so each session is rendered with its effort passed as --effort-kwarg, exactly as the
model saw it; --efforts selects which to keep.

A session longer than --max-tokens is cut at the last end of turn that fits (the end-of-turn
marker is read off the template). Sessions on repositories whose name hashes into the
held-out bucket (1 in --heldout-mod, about 2%) go only to eval.jsonl, so agent evals never
see a training repository. Output: <out-dir>/train.jsonl, eval.jsonl
({"id", "text", "source", "harness", "effort", "repo", "language", "tokens"}, shuffled),
stats.json.
"""

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARGS = None
tok = None
END = None
COLUMNS = ["trajectory_id", "repo", "language", "messages", "tools"]


def init():
    global tok, END
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ARGS.tokenizer, trust_remote_code=True)
    probe = tok.apply_chat_template(
        [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "ANSWERPROBE7351"},
        ],
        tokenize=False,
    )
    END = probe[probe.rindex("ANSWERPROBE7351") + len("ANSWERPROBE7351") :]
    assert END.strip(), f"cannot find the template's end-of-turn marker in {probe!r}"


def held_out(repo):
    return int(hashlib.sha1(repo.encode()).hexdigest(), 16) % ARGS.heldout_mod == 0


def file_effort(path):
    """Effort a parquet file was recorded at: thinking-<effort>/ subdirectory, else default."""
    parent = Path(path).parent.name
    return (
        parent.removeprefix("thinking-")
        if parent.startswith("thinking-")
        else "default"
    )


def render(row):
    tools = [json.loads(t) if isinstance(t, str) else t for t in (row["tools"] or [])]
    kw = {} if row["_effort"] == "default" else {ARGS.effort_kwarg: row["_effort"]}
    msgs = []
    for m in row["messages"]:
        m = {k: v for k, v in m.items() if v is not None}
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    pass
        msgs.append(m)
    try:
        text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, **kw)
    except Exception as e:  # noqa: BLE001
        return None, f"render: {type(e).__name__}"
    head = text[: ARGS.max_tokens * 8]
    enc = tok(head, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n > ARGS.max_tokens:
        limit = enc["offset_mapping"][ARGS.max_tokens - 1][1]
        cut = text.rfind(END, 0, limit - len(END) + 1)
        if cut < 0:
            return None, "no turn end within limit"
        text = text[: cut + len(END)]
        n = len(tok(text, add_special_tokens=False)["input_ids"])
    first = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), None)
    if first is None:
        return None, "no assistant turn"
    prompt = tok.apply_chat_template(
        msgs[:first],
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        **kw,
    )
    if len(text) <= len(prompt):
        return None, "no assistant turn within limit"
    return {
        "id": row["trajectory_id"],
        "text": text,
        "source": "open-swe-traces",
        "harness": row["_harness"],
        "effort": row["_effort"],
        "repo": row["repo"],
        "language": row["language"],
        "tokens": n,
    }, None


def verify(rows, template_name, n):
    """Loss share per row under SpecForge's preprocessing (all assistant turns)."""
    try:
        from specforge.data.preprocessing import preprocess_conversations
        from specforge.data.template import TEMPLATE_REGISTRY
    except ImportError as e:
        raise SystemExit(f"--verify-chat-template needs SpecForge importable: {e}")
    tmpl = TEMPLATE_REGISTRY.get(template_name)
    shares = []
    for r in rows[:n]:
        res = preprocess_conversations(
            tok,
            [r["text"]],
            tmpl,
            max_length=10**7,
            is_preformatted=True,
            train_only_last_turn=False,
        )
        mask = res["loss_mask"][0][0]
        shares.append(float(mask.float().mean()))
    zero = sum(s == 0 for s in shares)
    shares.sort()
    print(
        f"verify {template_name}: {len(shares)} rows, loss share p10/p50/p90 "
        f"{shares[len(shares) // 10]:.2f}/{shares[len(shares) // 2]:.2f}/{shares[len(shares) * 9 // 10]:.2f}, "
        f"rows without loss {zero}"
        + (" (OK)" if zero == 0 else " (FAIL: assistant turns not found)"),
        flush=True,
    )
    return zero


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="local Open-SWE-Traces copy (its data/<harness>/<model>/... parquet files)",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="trajectory model directory, e.g. qwen38_flash_next",
    )
    ap.add_argument(
        "--harnesses",
        nargs="*",
        default=None,
        help="default: every harness that has sessions of --model",
    )
    ap.add_argument(
        "--efforts",
        nargs="*",
        default=None,
        help="keep only these efforts: default (top-level files, xhigh), low, medium "
        "(default: all)",
    )
    ap.add_argument(
        "--effort-kwarg",
        default="reasoning_effort",
        help="chat-template kwarg that takes the effort",
    )
    ap.add_argument("--tokenizer", required=True, help="target model path or HF id")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument(
        "--per-harness",
        type=int,
        default=None,
        help="training sessions per harness (default: all)",
    )
    ap.add_argument("--eval-per-harness", type=int, default=100)
    ap.add_argument("--heldout-mod", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument(
        "--num-proc", type=int, default=max(1, min(48, (os.cpu_count() or 2) - 2))
    )
    ap.add_argument("--verify-chat-template", default=None, metavar="NAME")
    ap.add_argument("--verify-samples", type=int, default=50)
    ARGS = ap.parse_args()
    ARGS.out_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    base = ARGS.data_dir / "data"
    harnesses = ARGS.harnesses or sorted(
        p.parent.name for p in base.glob(f"*/{ARGS.model}")
    )
    if not harnesses:
        raise SystemExit(f"no sessions of {ARGS.model} under {base}")
    rng = random.Random(ARGS.seed)
    train, evals, stats = (
        [],
        [],
        {"model": ARGS.model, "max_tokens": ARGS.max_tokens, "harnesses": {}},
    )
    for harness in harnesses:
        files = sorted(
            f
            for f in glob.glob(
                f"{base}/{harness}/{ARGS.model}/**/*.parquet", recursive=True
            )
            if ARGS.efforts is None or file_effort(f) in ARGS.efforts
        )
        rows, efforts = [], {}
        for f in files:
            for r in pq.read_table(f, columns=COLUMNS).to_pylist():
                r["_harness"] = harness
                r["_effort"] = file_effort(f)
                efforts[r["_effort"]] = efforts.get(r["_effort"], 0) + 1
                rows.append(r)
        rng.shuffle(rows)
        tr = [r for r in rows if not held_out(r["repo"])]
        tr = tr[: ARGS.per_harness] if ARGS.per_harness else tr
        ev = [r for r in rows if held_out(r["repo"])][: ARGS.eval_per_harness]
        stats["harnesses"][harness] = {
            "sessions": len(rows),
            "files": len(files),
            "efforts": efforts,
            "train": len(tr),
            "eval": len(ev),
        }
        print(
            f"{harness}: {len(rows)} sessions in {len(files)} files {efforts}; "
            f"train {len(tr)}, eval {len(ev)}",
            flush=True,
        )
        train += tr
        evals += ev
    with mp.get_context("fork").Pool(ARGS.num_proc, initializer=init) as pool:
        for name, rows in (("train", train), ("eval", evals)):
            kept, errs = [], {}
            for out, err in pool.imap_unordered(render, rows, chunksize=4):
                if out:
                    kept.append(out)
                else:
                    errs[err] = errs.get(err, 0) + 1
            rng.shuffle(kept)
            with open(ARGS.out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for o in kept:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            toks = sorted(o["tokens"] for o in kept)
            stats[name] = {
                "rows": len(kept),
                "dropped": errs,
                "tokens_total": sum(toks),
                "tokens_p50": toks[len(toks) // 2] if toks else 0,
                "at_cap_share": round(
                    sum(t > ARGS.max_tokens * 0.9 for t in toks) / max(1, len(toks)), 3
                ),
            }
            print(name, json.dumps(stats[name]), flush=True)
            if name == "train":
                sample = kept[: ARGS.verify_samples]
    json.dump(stats, open(ARGS.out_dir / "stats.json", "w"), indent=1)
    if ARGS.verify_chat_template:
        init()
        if verify(sample, ARGS.verify_chat_template, ARGS.verify_samples):
            sys.exit(1)


if __name__ == "__main__":
    main()
