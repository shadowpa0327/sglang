# Prompt Sources for Draft-Model Training Data

A speculative-decoding draft learns to predict its target model, so its training
answers should be written by that target. Public datasets therefore supply only
the **prompts** (system and user turns); the target regenerates every assistant
turn (see the [runbook](README.md)).
Agent sessions are the exception: they are used as recorded, and only when the
target wrote them. This page lists the sources, what each contributes, the
complete corpus they form, and the blends built from them.

All sources download with one script, which converts chat prompts to the
regeneration input format (sampling, deduplicating, decontaminating and pinning
the dataset revision) and fetches raw agent sessions:

```bash
python draft_data_regen/scripts/download_prompt_sources.py --list
```

## The complete corpus

The corpus behind the Qwen3.8-Flash-Next drafts has four parts. Three are chat
prompts the target answers; the fourth is agent sessions the target already
wrote.

| Part | Source | Complete size | Done so far | Trained as |
| --- | --- | --- | --- | --- |
| General chat | PerfectBlend | 1.42M prompts | 348,841 thinking and 401,379 non-thinking regenerations | 8K rows, last-turn loss |
| Multi-turn chat | UltraChat | 208K conversations, 3.2 user turns on average | 918 thinking and 78,308 non-thinking regenerations | 8K rows, last-turn loss |
| Code | OpenCodeInstruct | 5.0M prompts (250K sampled) | 120,865 thinking regenerations | 8K rows, last-turn loss |
| Coding agent | Open-SWE-Traces v2.0, Qwen3.8-Flash-Next sessions | 489,109 sessions: 5 harnesses at 3 reasoning efforts (168,824 xhigh, 151,689 medium, 168,596 low) | 97,292 xhigh sessions in 3 harnesses: 95,596 training rows (2.90B tokens at 32K) and 300 held-out eval sessions | 32K-64K rows, every assistant turn |

What completing it costs, at the throughput of three TP2 H100 servers (about
45-50M generated tokens per hour, thinking on, 8K per turn):

| Remaining work | Generated tokens | Time |
| --- | --- | --- |
| UltraChat, all 208K conversations (about 4K tokens each) | about 0.85B | about 18 h |
| OpenCodeInstruct, the other ~126K of the 250K sample | about 0.35B | about 7 h |
| PerfectBlend, the other ~1M prompts (optional; the blend stopped at 1B tokens by design) | about 2B | about 40 h |
| Open-SWE-Traces sessions not used yet: Claude Code and OpenCode at xhigh (71,532), and every low- and medium-effort session (320,285) | none: download 37.1 GB, convert on CPU (about 15B tokens, 55 GB of rows at 32K) | about 5 h |

Coverage gaps worth closing for a general-purpose draft (see the DaoCloud blend
below): multilingual (`aya`) and instruction following (`nemotron-if`,
`nemotron-structured`).

### Recipe

```bash
P=./cache/dataset/prompts; R=./cache/dataset/regen; TARGET=nvidia/Qwen3.8-Flash-Next-NVFP4
# 1. chat prompts, decontaminated against the evals you report
python draft_data_regen/scripts/download_prompt_sources.py \
    --sources perfectblend ultrachat opencodeinstruct \
    --sample opencodeinstruct=250000 \
    --decontaminate-against ./eval_datasets --output-dir $P
# 2. agent sessions written by the target
python draft_data_regen/scripts/download_prompt_sources.py --swe-traces qwen38_flash_next --output-dir $P
# 3. regenerate each chat source (servers on :31000 and :32000; one driver per file)
for s in perfectblend ultrachat opencodeinstruct; do
  python draft_data_regen/scripts/regen_driver.py $P/${s}_prompts.jsonl $R/${s}_regen.jsonl \
      127.0.0.1:31000 127.0.0.1:32000 --model $TARGET \
      --thinking --budget 8192 --max-tokens 16384 \
      --effort-mix low:0.4,medium:0.3,xhigh:0.3 --concurrency 128 > logs/regen_$s.log 2>&1 &
done
# 4. training rows: 8K chat (last-turn loss) and 32K agent sessions (all turns)
python draft_data_regen/scripts/build_regen_training_rows.py --tokenizer $TARGET --out-dir ./cache/dataset/chat_8k \
    --budget 8192 --decontaminate-against ./eval_datasets --verify-chat-template qwen3.5 $R/*_regen.jsonl
python draft_data_regen/scripts/build_swe_trace_rows.py --data-dir $P/open_swe_traces --model qwen38_flash_next \
    --tokenizer $TARGET --out-dir ./cache/dataset/agent_32k --max-tokens 32768 --verify-chat-template qwen3.5
```

The chat rows train with `data.train_only_last_turn: true` and the agent rows
with `false`. SpecForge applies the flag per run, so train them as separate
stages (chat at 8K first, then a long-context stage with the agent rows), as
Inkling-Small-DSpark does.

## Catalog

| Key | Hugging Face dataset | Rows | License | What it adds |
| --- | --- | --- | --- | --- |
| `perfectblend` | [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend) | 1.42M | Apache-2.0 | The general backbone: chat, math, code and instruction following in the PerfectBlend mix; 14% multi-turn |
| `ultrachat` | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft`) | 208K | MIT | Multi-turn assistant chat: 3.2 user turns on average, 92% with 2-4 |
| `opencodeinstruct` | [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 5.0M | CC-BY-4.0 | Single-turn coding tasks (mostly Python); sample it |
| `aya` | [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) | 202K | Apache-2.0 | Human-written prompts in 65 languages |
| `nemotron-if` | [`nvidia/Nemotron-SFT-Instruction-Following-Chat-v3`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) (`instruction_following`) | 249K | CC-BY-4.0 / ODC-By | Verifiable format and length constraints; about 60% multi-turn, a third with system prompts |
| `nemotron-chat` | same dataset, `chat` split | 637K | CC-BY-4.0 / ODC-By | Multi-turn open chat. Rows seeded from lmsys-chat-1m and WildChat (about 55%) ship **without** their prompts; the script drops them unless you restore them with the dataset's `prepare_chat_prompts.py` (needs lmsys access) |
| `nemotron-structured` | [`nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2) | 54K | CC-BY-4.0 | JSON/XML/YAML/TOML output to a schema; the tool-calling subset is skipped |
| `magicoder` | [`ise-uiuc/Magicoder-Evol-Instruct-110K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | 111K | Apache-2.0 | Evol-Instruct coding prompts; about 1.3% overlap HumanEval at 13 words |
| `codealpaca` | [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 20K | CC-BY-4.0 | Short coding instructions |
| `daocloud-prompts` | [`DaoCloud/Qwen3.8-27B-Drafter-SFT`](https://huggingface.co/datasets/DaoCloud/Qwen3.8-27B-Drafter-SFT) (`prompts` config) | 368K | Upstream terms | A ready-made, filtered and decontaminated drafter mix of PerfectBlend, OpenCodeInstruct, aya and two Nemotron instruction-following sets |
| `--swe-traces MODEL` | [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) | 169K sessions per recent model | CC-BY-4.0 | Coding-agent sessions with tools, used as recorded; see [Open-SWE-Traces](#open-swe-traces) |

Prompts remain subject to their upstream licenses; regenerated answers are
subject to the target model's license.

## What the download script does to each source

- **Prompt turns only matter.** Rows keep the source's system and user turns.
  Assistant turns of multi-turn sources are kept only so the regenerator's
  input check passes; the regenerator replaces every one of them. Sources that
  ship user turns only (`daocloud-prompts`) get a placeholder between them.
- **Normalization.** Consecutive same-role turns are merged, empty system
  prompts removed, trailing assistant turns dropped.
- **Drops.** Rows that are invalid after normalization, rows with tool
  definitions (the regenerators do not pass tools), exact duplicate prompts
  within a source, and, with `--decontaminate-against`, prompts that share a
  13-word sequence with any string in the eval files (or equal a short eval
  string of 4-12 words after lowercasing).
- **Order.** Each output file is shuffled with a fixed seed, so regenerating
  any prefix of it regenerates a random sample, and a run stopped early is
  still representative.
- **Reproducibility.** `manifest.json` records each source's dataset revision,
  counts and drop reasons; `--revisions-from manifest.json` downloads the same
  revisions again.

Expected decontamination losses against the DeepSpec eval suite (AIME 24/25,
Alpaca, Arena-Hard v2, GSM8K, HumanEval, LBPP, LiveCodeBench, MATH-500, MBPP,
MT-Bench, SWE-bench): about 1.3% of PerfectBlend (half of it LeetCode-style
problems that overlap LiveCodeBench), 1.0-1.5% of OpenCodeInstruct, and none
in a 1,000-conversation sample of UltraChat.

## UltraChat

UltraChat is the corpus's multi-turn chat: 207,865 conversations, of which 32%
have two user turns, 33% three, 27% four and 8% five or more. Its follow-up
questions were written against the original answers, so a regenerated answer
can make a follow-up slightly off; that is accepted practice, and the draft
still learns the target's answers given realistic multi-turn histories.

- **How it regenerates.** The driver answers one user turn at a time and sends
  earlier answers back without their reasoning, as a chat client does. Under
  `--budget 8192` each turn gets 8,192 tokens minus its prompt, and the prompt
  grows with every answer, so late turns of long conversations get less room;
  a turn left with under 256 tokens ends the conversation. In our partial
  thinking run, 2.8 of 3.2 turns per conversation were answered and 2.2
  training rows per conversation survived the 8K row limit.
- **Cost.** About 4K generated tokens per conversation with thinking on, so
  all of UltraChat is about 0.85B tokens, or about 18 hours on three TP2 H100
  servers. Use the prefix cache (SGLang's default) so each turn reuses its
  history.
- **Rows.** The builder writes one row per answer, each with the history the
  target saw, and trains only that answer.
- **Status.** The thinking run stopped at 918 conversations when its GPUs were
  reassigned; the non-thinking regeneration reached 78,308 (4K answer cap).

## Blends

### Qwen3.8-Flash-Next thinking DSpark, stage 1 (8K)

Regenerated with thinking on, temperature 1.0, top-p 0.95, top-k 20; reasoning
effort low / medium / xhigh drawn per conversation at 40 / 30 / 30%; each turn
limited to 8,192 tokens of prompt plus answer, answers cut there kept as
truncated rows.

| Source | Conversations | Training rows (one per turn) | Tokens | Cut at 8K |
| --- | --- | --- | --- | --- |
| PerfectBlend | 348,841 | 433,868 | 911M | 10.4% |
| OpenCodeInstruct (250K sampled) | 120,865 | 120,665 | 371M | 18.2% |
| UltraChat | 918 | 1,978 | 6M | 13% |
| **Total** | **470,624** | **556,511** | **1.29B (1.01B trained)** | |

Generation took about a day on three TP2 H100 servers (about 19.5K
conversations, or 45-50M generated tokens, per hour at the tuned settings).

Its exact prompts, in run order and with each conversation's effort, are pinned
in `draft_data_regen/locks/qwen38-flash-next-think-stage1/` (eval-overlap prompts
already removed). `download_prompt_sources.py --lock` rebuilds them, and the
[reproduction recipe](REPRODUCE.md) takes another target through the
same corpus.

### Qwen3.8-Flash-Next agent DSpark (32K)

All OpenHands, SWE-agent and mini-SWE-agent sessions written by
Qwen3.8-Flash-Next, rendered with its template and cut at 32K tokens: 95,596
training rows, 2.90B tokens (median 32K; 81% reach the cap), trained on every
assistant turn; 300 sessions on held-out repositories for evaluation.

### Earlier non-thinking blends

PerfectBlend (401,379) and UltraChat (78,308) regenerated with thinking off and
a 4K answer cap, combined with the agent rows into a 557,669-row DFlash2 blend.
A draft trained only on non-thinking answers loses about a third of its
acceptance on thinking-mode output, which is why the thinking corpus above
replaced them.

### Reference blends

- **DaoCloud Qwen3.8-27B drafter**: 367,535 conversations, 450,401 rows, 1.95B
  tokens; thinking on with effort low / medium / xhigh at 40 / 30 / 30%.

  | Source | Conversations |
  | --- | --- |
  | `mlabonne/open-perfectblend` | 209,120 |
  | `nvidia/OpenCodeInstruct` | 59,827 |
  | `CohereLabs/aya_dataset` | 58,835 |
  | `nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2` | 19,993 |
  | `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | 19,760 |

- **RadixArk Inkling-Small-DSpark-Preview**: stage 1 at 8K on 444K
  self-regenerated PerfectBlend conversations with reasoning (temperature 1.0,
  top-p 0.95); stage 2 at 64K on 250K chat conversations (192K new plus 58K
  replayed from stage 1) and agent trajectories.

## Open-SWE-Traces

[`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
(CC-BY-4.0) holds complete software-engineering agent sessions (system, user,
assistant and tool turns, with tool definitions) on Scale-SWE and
SWE-rebench-V2 tasks.

| Config | Trajectory model (`--swe-traces` name) | Sessions | Size |
| --- | --- | --- | --- |
| `v2.0` (default) | Qwen3.8-Flash-Next (`qwen38_flash_next`) | 489,109 (168,824 xhigh, 151,689 medium, 168,596 low) | 37.1 GB |
| `v1.2` | Qwen3.8-27B (`qwen38_27b`) | 235,396 | 30.4 GB |
| `v1.1` | Qwen3.6-27B (`qwen36_27b`), DeepSeek-V4-Flash-0731 (`deepseek_v4_flash_0731`) | 212,931, 50,295 | 13.7 GB, 3.7 GB |
| `v1.0` | MiniMax-M2.5 (`minimax_m25`), Qwen3.5-122B (`qwen35_122b`) | 90,422, 60,797 | 7.3 GB, 5.7 GB |

Qwen3.8-Flash-Next sessions by harness and reasoning effort (2026-10-27
revision `9b073ef`):

| Harness (directory) | xhigh | medium | low | In our 32K agent run |
| --- | --- | --- | --- | --- |
| Claude Code (`claudecode`) | 34,934 | 30,356 | 34,075 | no (released after the run) |
| OpenCode (`opencode`) | 36,598 | 31,490 | 33,071 | no (released after the run) |
| OpenHands (`openhands`) | 26,027 | 25,898 | 30,367 | xhigh only |
| SWE-agent (`sweagent`) | 34,688 | 33,849 | 35,045 | xhigh only |
| mini-SWE-agent (`minisweagent`) | 36,577 | 30,096 | 36,038 | xhigh only |

The xhigh sessions are the top-level files of each task folder (`scale-swe/`,
`swe-rebench-v2/`), and the others sit in `thinking-low/` and
`thinking-medium/` below them. The card's tables count only xhigh. A session's
system prompt does not record its effort, but the template renders effort into
it ("Reasoning effort is set to low..."; nothing for medium), so
`build_swe_trace_rows.py` renders each session at the effort of its folder.
Rendering a low-effort session without its effort would train the draft on a
prompt the model never saw. `--efforts default low medium` picks which to keep.
The low and medium sessions match the effort mix of the chat data.

Our agent run used revision `1db6741` (2026-10-20). For Qwen3.8-Flash-Next it
holds only the xhigh OpenHands, SWE-agent and mini-SWE-agent sessions. The lock
in `draft_data_regen/locks/qwen38-flash-next-think-stage1/lock.json` pins it, so
`--revisions-from` that file reproduces the run (see the
[reproduction recipe](REPRODUCE.md)).

**Use them as recorded, and only for their own model.** The tool results were
produced by the original model's actions, so replacing its turns would
desynchronize the trajectory; the sessions are on-policy exactly when the
trajectory model is your target. For another target, skip them or run fresh
agent rollouts with the target in sandboxed harnesses.

**Converting.** `draft_data_regen/scripts/build_swe_trace_rows.py` renders each session once with
the target's template (tools included, every turn's reasoning kept), cuts it at
the last end of turn within `--max-tokens`, and keeps sessions on 1 in 50
repositories (by name hash) for evaluation only:

```bash
python draft_data_regen/scripts/download_prompt_sources.py --swe-traces qwen38_flash_next --output-dir $P
python draft_data_regen/scripts/build_swe_trace_rows.py --data-dir $P/open_swe_traces --model qwen38_flash_next \
    --tokenizer $TARGET --out-dir ./cache/dataset/agent_32k --max-tokens 32768 \
    --verify-chat-template qwen3.5
```

Train the rows with `data.is_preformatted: true`, `data.train_only_last_turn:
false` and `data.max_length` equal to `--max-tokens`. Sessions are long: at
32K, 81% of rows are cut at the cap (median 32K tokens); at 64K, 56% (median
62K); uncut, the held-out sessions have a median of 74K tokens.
