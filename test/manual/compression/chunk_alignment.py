"""Chunk/block alignment under batched chunked prefill.

Several prompts of different lengths prefill together and share one chunk
budget, so a request can be admitted with a leftover budget and start its
chunks mid-block. Every whole block of every prompt must still be stored (no
`block_skipped_no_state`), and a second submission of the same prompts must
restore exactly the whole-block prefix of each one without storing anything.
Run from the SGLang checkout with PYTHONPATH=python .venv/bin/python ... .
Exits non-zero when a block was skipped, an expected block is missing, or a
restore does not cover the expected prefix.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path


def build_prompt(tokenizer, index, length):
    # Distinct from the first token on, so prompts never share a block.
    text = "".join(
        f"Record {index}-{i}: entry {(index + 1) * 7919 * (i + 1) % 100003} "
        f"is filed under section {i % 17}. "
        for i in range(length // 4 + 32)
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) >= length, (index, len(ids), length)
    return ids[:length]


def main():
    import sglang as sgl

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plugin", default="identity")
    parser.add_argument(
        "--lengths",
        default="3000,2500,4100,2200",
        help="Comma-separated prompt lengths in tokens",
    )
    parser.add_argument("--block-pages", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--hybrid", action="store_true", help="Expect state bytes")
    parser.add_argument(
        "--cuda-graph", action="store_true", help="Keep CUDA graphs enabled"
    )
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    block_tokens = args.block_pages * args.page_size
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "events.jsonl"
    if metrics.exists():
        raise ValueError("Use a fresh output directory for each run")
    engine = sgl.Engine(
        model_path=args.model,
        dtype="bfloat16",
        mem_fraction_static=0.65,
        max_total_tokens=32768,
        context_length=8192,
        max_running_requests=4,
        chunked_prefill_size=args.chunked_prefill_size,
        page_size=args.page_size,
        disable_cuda_graph=not args.cuda_graph,
        disable_overlap_schedule=False,
        attention_backend="triton",
        log_level="warning",
        enable_unified_cache_external_linker=True,
        unified_cache_external_linker_backend="compression",
        prefix_compression_config=json.dumps(
            {
                "plugin": args.plugin,
                "block_pages": args.block_pages,
                "metrics_path": str(metrics),
                "key_space": "auto",
            }
        ),
    )
    failures = []
    try:

        def control(action):
            for attempt in range(50):
                try:
                    engine.collective_rpc("prefix_compression_control", action=action)
                    return
                except AssertionError as error:
                    if "idle engine" not in str(error):
                        raise
                    time.sleep(0.1)
            raise RuntimeError("Engine did not become idle")

        tokenizer = engine.tokenizer_manager.tokenizer
        prompts = [build_prompt(tokenizer, i, n) for i, n in enumerate(lengths)]
        assert [len(p) for p in prompts] == lengths
        sampling = {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}

        control("compressed")
        # Pass 1: all prompts in one batch; their prefill stores the blocks.
        first = engine.generate(input_ids=prompts, sampling_params=sampling)
        # Pass 2: the same prompts must restore whole-block prefixes only.
        second = engine.generate(input_ids=prompts, sampling_params=sampling)
        time.sleep(1.0)  # let the scheduler flush the last finish events
        events = [json.loads(line) for line in metrics.read_text().splitlines()]

        def by_rid(name, rid):
            return [e for e in events if e["event"] == name and e.get("rid") == rid]

        report = []
        for i, (n, a, b) in enumerate(zip(lengths, first, second)):
            rid_a = a["meta_info"]["id"]
            rid_b = b["meta_info"]["id"]
            expected = n // block_tokens
            stored = by_rid("compress", rid_a)
            skipped = by_rid("block_skipped_no_state", rid_a)
            restores = by_rid("private_restore", rid_b)
            checkpoint_tokens = (
                math.lcm(block_tokens, args.chunked_prefill_size)
                if args.hybrid
                else block_tokens
            )
            expected_restore = (n - 1) // checkpoint_tokens * checkpoint_tokens
            matched = restores[0]["matched_tokens"] if restores else 0
            entry = {
                "prompt": i,
                "tokens": n,
                "expected_blocks": expected,
                "stored_blocks": len(stored),
                "skipped_blocks": len(skipped),
                "skipped_indices": sorted(e["block"] for e in skipped),
                "restore_matched_tokens": matched,
                "expected_restore_tokens": expected_restore,
                "second_pass_compress_events": len(by_rid("compress", rid_b)),
                "second_pass_skipped": len(by_rid("block_skipped_no_state", rid_b)),
                "second_pass_cached_tokens": b["meta_info"]["cached_tokens"],
            }
            report.append(entry)
            if skipped:
                failures.append(f"prompt {i}: {len(skipped)} block(s) skipped")
            if len(stored) != expected:
                failures.append(
                    f"prompt {i}: stored {len(stored)} of {expected} block(s)"
                )
            if args.hybrid and expected_restore and not any(
                e["state_bytes"] > 0
                and (e["block"] + 1) * block_tokens == expected_restore
                for e in stored
            ):
                failures.append(
                    f"prompt {i}: no recurrent checkpoint at {expected_restore}"
                )
            if len(restores) != 1 or matched != expected_restore:
                failures.append(
                    f"prompt {i}: restore matched {matched}, "
                    f"expected {expected_restore}"
                )
            if entry["second_pass_compress_events"]:
                failures.append(f"prompt {i}: second pass stored new blocks")
        result = {
            "model": args.model,
            "lengths": lengths,
            "block_tokens": block_tokens,
            "chunked_prefill_size": args.chunked_prefill_size,
            "per_prompt": report,
            "failures": failures,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        engine.shutdown()
    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("OK: every block stored, none skipped, every prefix restored")


if __name__ == "__main__":
    main()
