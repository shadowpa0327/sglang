# Reproduce the Qwen3.8-Flash-Next corpus for another target

This page rebuilds, for a new target model, the same training corpus we built
for Qwen3.8-Flash-Next. Only the answers change: the new target writes them.
Everything else stays the same: the same 466,969 prompts in the same order, each
conversation's reasoning effort, sampling, the 8K token budget, the filters and
the training settings. Follow the steps in order, and check each one against
its expected numbers before moving on.

To regenerate for **Qwen3.8-Flash-Next itself**, use the middle column of the
table below as is, and the exact server command in step 2. The answers are
sampled at temperature 1.0, so a rerun matches our corpus in its prompts,
efforts, settings and statistics, but not token for token. The agent rows in
step 5 are recorded sessions, so they come out identical to ours. To give
someone our exact chat answers, share the regenerated files themselves.

Background for each choice is in the [runbook](README.md)
and the [prompt source catalog](PROMPT_SOURCES.md). You do not need them to run
this recipe.

## What is pinned

The lock `draft_data_regen/locks/qwen38-flash-next-think-stage1/` holds:

- `lock.json`: the dataset repository and revision of each source, the
  settings, and the reference counts.
- `prompts.jsonl.gz`: one `{"source", "id", "reasoning_effort"}` line per
  prompt, in the order we regenerated them.

| Source | Prompts | Effort low / medium / xhigh |
| --- | --- | --- |
| PerfectBlend (`mlabonne/open-perfectblend`) | 344,928 | 135,672 / 103,115 / 106,141 |
| OpenCodeInstruct (`nvidia/OpenCodeInstruct`) | 120,954 | 48,081 / 36,836 / 36,037 |
| UltraChat (`HuggingFaceH4/ultrachat_200k`) | 1,089 | 0 / 0 / 1,089 |

Prompts that overlap the eval sets have already been removed from the lock.
The eval sets are DeepSpec's `eval_datasets` (commit 005e03b) plus the full
HumanEval, MBPP and LiveCodeBench prompts, checked with a 13-word overlap test.
Two PerfectBlend entries have no prompt text in the dataset and are skipped, so
466,969 prompts are regenerated. A rebuilt prompt file matches, prompt for
prompt, what Qwen3.8-Flash-Next was sent; we checked all 466,969.

## Values to set for your target

| Value | Qwen3.8-Flash-Next | Yours |
| --- | --- | --- |
| `TARGET`: Hugging Face id or local path | `nvidia/Qwen3.8-Flash-Next-NVFP4` | the checkpoint you will serve |
| GPUs per server (`--tp-size`) | 2 (H100) | smallest that fits with room for KV cache |
| `--reasoning-parser` in SGLang | `qwen3` | the parser for your model family |
| Effort levels | `reasoning_effort` = `low` / `medium` / `xhigh` | rename with `--effort-map`, e.g. `xhigh=high`; set `--effort-kwarg` if the template calls it something else |
| SpecForge `chat_template` | `qwen3.5` | the template whose loss check prints `(OK)` in step 4 |
| Open-SWE-Traces model directory | `qwen38_flash_next` | `download_prompt_sources.py --list`; none means skip step 5 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 | keep these, unless your model card recommends other thinking-mode sampling |

To find the effort levels, read the target's chat template. Qwen3.8 rejects
unknown levels with "Unexpected reasoning effort". A template without effort
control ignores the setting; in that case leave it as is.

```bash
export TARGET=<hf id or path>
export WORK=/big/disk/regen_<target>    # ~10 GB for chat; the agent part adds ~50 GB
export HF_HOME=/big/disk/hf_home         # ~35 GB of dataset downloads and caches
export LOCK=draft_data_regen/locks/qwen38-flash-next-think-stage1
```

## 1. Rebuild the prompt files

```bash
python draft_data_regen/scripts/download_prompt_sources.py --lock $LOCK --output-dir $WORK/prompts
```

Expect `perfectblend: 344926 of 344928`, `opencodeinstruct: 120954 of 120954`
and `ultrachat: 1089 of 1089`; the two missing PerfectBlend rows have no
prompt text. This takes about 5 minutes once the datasets are downloaded.
OpenCodeInstruct is the largest download at 24 GB. Any other count means the
dataset revision changed; stop and report it.

## 2. Serve the target

Serve it with the quantization and numerics it will have in production, with
no speculative decoding, using one server per GPU group:

```bash
python -m sglang.launch_server --model-path $TARGET --tp-size <N> --port 31000 \
    --reasoning-parser <parser> --context-length 32768 \
    --max-running-requests 256 --cuda-graph-max-bs-decode 256 --mem-fraction-static 0.88
```

Qwen3.8-Flash-Next on H100 needs SGLang 0.5.21 or later, the first release
with its `qwen4_exp` architecture. This is the exact command we ran, one server
per GPU pair (we ran three):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
python -m sglang.launch_server --model-path nvidia/Qwen3.8-Flash-Next-NVFP4 --tp-size 2 \
    --dtype bfloat16 --attention-backend fa3 --enforce-disable-flashinfer-allreduce-fusion \
    --mamba-ssm-dtype bfloat16 --disable-radix-cache --context-length 32768 \
    --mem-fraction-static 0.88 --max-running-requests 256 --cuda-graph-max-bs-decode 256 \
    --reasoning-parser qwen3 --host 127.0.0.1 --port 31000
```

On H100 its NVFP4 experts run through the Marlin W4A16 fallback. Keep the PLE
embedding table in host memory, which is the default; `--no-ple-offload-embedding`
runs out of memory at TP2. Older SGLang spells the graph flag
`--cuda-graph-max-bs`. Measure `gen throughput` in the server log
for about 5 minutes per setting. For Qwen3.8-Flash-Next, raising
`--max-running-requests` from 96 to 256 lifted it from 3,136 to 4,306 tokens
per second. Keep the driver's `--concurrency` equal to that setting.

## 3. Regenerate

Run one driver per source, listing every server. Each prompt row already
carries its effort, so do not pass `--effort-mix`.

```bash
for s in perfectblend opencodeinstruct ultrachat; do
  python draft_data_regen/scripts/regen_driver.py $WORK/prompts/${s}_prompts.jsonl $WORK/regen/${s}.jsonl \
      127.0.0.1:31000 [127.0.0.1:32000 ...] --model $TARGET \
      --thinking --budget 8192 --max-tokens 16384 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --concurrency 256 \
      [--effort-map xhigh=high] >> $WORK/regen_$s.log 2>&1
done
```

- `--budget 8192` gives each turn 8,192 tokens minus its prompt. An answer cut
  at that limit is kept and marked `truncated`.
- `--max-tokens` must be at least the budget; the default of 4096 would cut
  answers short.
- Earlier answers go back to the server without their reasoning.
- If the run stops, rerun the same command; it resumes from the output files.
- Progress and the time remaining are printed once a minute.

Expect `_error.jsonl` to be empty and under 0.5% of rows in `_skipped.jsonl`.
Read a few rows, and check that the `reasoning_content` fields hold real
reasoning. Qwen3.8-Flash-Next generated about 1.0B tokens. That took about 25
hours on three TP2 H100 servers, each generating 3,000 to 4,300 tokens per
second.

## 4. Build training rows

```bash
python draft_data_regen/scripts/build_regen_training_rows.py --tokenizer $TARGET --out-dir $WORK/stage1_8k \
    --budget 8192 --verify-chat-template <template> [--effort-kwarg <kwarg>] \
    $WORK/regen/perfectblend.jsonl $WORK/regen/opencodeinstruct.jsonl $WORK/regen/ultrachat.jsonl
```

Each assistant turn becomes one row, rendered as the server saw it. Rows stuck
in a repetition loop and rows over the budget are dropped. The verify line
must end in `(OK)`; if it does not, try another SpecForge template or add one.
For reference, Qwen3.8-Flash-Next gave 556,511 rows, 1.29B tokens, 1.01B loss
tokens and 10% truncated answers. Another target lands near these numbers, not
exactly on them; its counts depend on how long it writes.

## 5. Agent sessions (only if the target has them)

`nvidia/Open-SWE-Traces` sessions are used as recorded, and only for the model
that wrote them. We trained on the same three harnesses as below:

```bash
python draft_data_regen/scripts/download_prompt_sources.py --swe-traces <model dir> --output-dir $WORK/prompts \
    --revisions-from $LOCK/lock.json
python draft_data_regen/scripts/build_swe_trace_rows.py --data-dir $WORK/prompts/open_swe_traces \
    --model <model dir> --harnesses openhands sweagent minisweagent \
    --tokenizer $TARGET --out-dir $WORK/agent_32k --max-tokens 32768 \
    --verify-chat-template <template>
```

- `--revisions-from $LOCK/lock.json` pins Open-SWE-Traces to the version we
  used (`1db6741`, 10.8 GB). For Qwen3.8-Flash-Next it holds xhigh-effort
  sessions from these three harnesses only.
- Keep the harnesses in this order. It decides which held-out sessions become
  the eval set.
- For Qwen3.8-Flash-Next this gives our rows exactly: 95,596 training rows
  (2.90B tokens) and the same 300 eval sessions. We checked this on a
  1,200-row sample.
- Without the pin you get the current dataset. Since 2026-10-27 it holds
  489,109 Qwen3.8-Flash-Next sessions: Claude Code and OpenCode were added, and
  every harness also has low- and medium-effort sessions in `thinking-low/`
  and `thinking-medium/`. The builder renders each session at its own effort,
  and `--efforts default` keeps only the xhigh ones. All of them come to about
  15B tokens at 32K, or about 55 GB of rows.
- For another target, drop `--revisions-from` to get its latest sessions.

## 6. Train

Train the chat rows (step 4) as stage 1:

```yaml
data:
  is_preformatted: true
  train_only_last_turn: true     # each row trains only its final answer
  chat_template: <template>
  max_length: 8192
```

Our stage 1 was a DSpark draft with these settings:

- training block 15, serving block 7;
- 2 epochs, global batch 512;
- LR 6e-4 on a cosine schedule with 4% warmup;
- `loss_decay_gamma` 7.

Our SpecForge config for this stage is
`examples/configs/online/disaggregated/managed-local/qwen3.8-flash-next-nvfp4-dspark-h100-1server-dp2-think8k-stage1.yaml`.
It is in our SpecForge working tree and not upstream yet. It also needs the
draft config `configs/qwen3.8-flash-next-dspark-b15.json` and the SGLang 0.5.21
capture patch `patches/sglang/v0.5.21/`, both from the same tree.

Agent rows (step 5) are a separate stage with `train_only_last_turn: false`
and `max_length: 32768`. SpecForge sets that flag once per run, so the two
kinds of rows cannot share a run.

## Do not change

- The lock: its prompt set, its order and the per-prompt efforts. To add data,
  make a new lock instead of editing this one.
- The 8K budget per turn, and keeping truncated answers.
- One row per assistant turn, history without reasoning, and
  `train_only_last_turn: true` for chat rows.
- The loop filter (`--loop-zlib 0.10`) and the budget slack of 8 tokens.

## Where the reference run differed

- The first 3,818 PerfectBlend conversations and all 1,089 UltraChat ones ran
  before the budget existed. They used the template's default effort (`xhigh`,
  which is what the lock records) and up to 16K new tokens. The build drops
  turns over 8K (789 across the reference corpus). A run from the lock puts
  the 8K budget on these too, so their long answers are kept as truncated.
- Eval-overlap prompts were regenerated and then dropped at build time. The
  lock leaves them out up front.
- The early part of the run, before 2026-10-20 23:08 UTC, kept the SSM state
  in fp32 and ran 96 to 192 requests per server. The step 2 command, with a
  bf16 state, is how the model is served.

## Target without thinking

Leave out `--thinking`, `--budget` and `--effort-map`, and pass
`--max-tokens 4096`. Build with `--no-thinking`; that build reuses the same
prompts.
