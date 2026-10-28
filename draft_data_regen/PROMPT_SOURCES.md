# Prompt Sources for Draft-Model Training Data

A speculative-decoding draft learns to predict its target model, so its training
answers should be written by that target. Public datasets therefore supply only
the **prompts** (system and user turns); the target regenerates every assistant
turn (see the [runbook](README.md)).
This page lists the prompt sources we use, what each contributes, and the blends
they went into.

All sources below download with one script, which converts them to the
regeneration input format, samples, deduplicates, decontaminates and pins the
dataset revision:

```bash
python draft_data_regen/scripts/download_prompt_sources.py --list
python draft_data_regen/scripts/download_prompt_sources.py \
    --sources perfectblend opencodeinstruct aya nemotron-if \
    --sample opencodeinstruct=250000 aya=60000 \
    --decontaminate-against ./eval_datasets \
    --output-dir ./cache/dataset/prompts
```

## Catalog

| Key | Hugging Face dataset | Rows | License | What it adds |
| --- | --- | --- | --- | --- |
| `perfectblend` | [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend) | 1.42M | Apache-2.0 | The general backbone: chat, math, code and instruction following in the PerfectBlend mix; 14% multi-turn |
| `ultrachat` | [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`train_sft`) | 208K | MIT | Long multi-turn assistant chat |
| `opencodeinstruct` | [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 5.0M | CC-BY-4.0 | Single-turn coding tasks (mostly Python); sample it |
| `aya` | [`CohereLabs/aya_dataset`](https://huggingface.co/datasets/CohereLabs/aya_dataset) | 202K | Apache-2.0 | Human-written prompts in 65 languages |
| `nemotron-if` | [`nvidia/Nemotron-SFT-Instruction-Following-Chat-v3`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) (`instruction_following`) | 249K | CC-BY-4.0 / ODC-By | Verifiable format and length constraints; about 60% multi-turn, a third with system prompts |
| `nemotron-chat` | same dataset, `chat` split | 637K | CC-BY-4.0 / ODC-By | Multi-turn open chat. Rows seeded from lmsys-chat-1m and WildChat (about 55%) ship **without** their prompts; the script drops them unless you restore them with the dataset's `prepare_chat_prompts.py` (needs lmsys access) |
| `nemotron-structured` | [`nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2) | 54K | CC-BY-4.0 | JSON/XML/YAML/TOML output to a schema; the tool-calling subset is skipped |
| `magicoder` | [`ise-uiuc/Magicoder-Evol-Instruct-110K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | 111K | Apache-2.0 | Evol-Instruct coding prompts; about 1.3% overlap HumanEval at 13 words |
| `codealpaca` | [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | 20K | CC-BY-4.0 | Short coding instructions |
| `daocloud-prompts` | [`DaoCloud/Qwen3.8-27B-Drafter-SFT`](https://huggingface.co/datasets/DaoCloud/Qwen3.8-27B-Drafter-SFT) (`prompts` config) | 368K | Upstream terms | A ready-made, filtered and decontaminated drafter mix of PerfectBlend, OpenCodeInstruct, aya and two Nemotron instruction-following sets |

Agent trajectories are a separate case: see [Agent traces](#agent-traces).

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
problems that overlap LiveCodeBench) and 1.0-1.5% of OpenCodeInstruct.

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
conversations, or 45-50M generated tokens, per hour at the tuned settings). Gaps relative to the DaoCloud blend: no multilingual (`aya`) and no
instruction-following (`nemotron-if`, `nemotron-structured`) prompts.

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

## Agent traces

[`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
(CC-BY-4.0) holds complete software-engineering agent sessions (system, user,
assistant and tool turns, with tool definitions) on Scale-SWE and
SWE-rebench-V2 tasks, across five harnesses: OpenHands, SWE-agent,
mini-swe-agent, Claude Code and OpenCode.

| Config | Trajectory model | Size |
| --- | --- | --- |
| `v2.0` (default) | `qwen38_flash_next` (xhigh, medium, low effort) | 37.1 GB |
| `v1.2` | `qwen38_27b` (xhigh) | 30.4 GB |
| `v1.1` | `qwen36_27b`, `deepseek_v4_flash_0731` | 13.7 GB, 3.7 GB |
| `v1.0` | `minimax_m25`, `qwen35_122b` | 7.3 GB, 5.7 GB |

These sessions are not regenerated: the tool results were produced by the
original model's actions, so replacing its turns would desynchronize the
trajectory. Use them as they are, and only when the trajectory model is your
target (they are then on-policy). For another target, either skip them or run
fresh agent rollouts with the target in sandboxed harnesses. Download raw
sessions with:

```bash
python draft_data_regen/scripts/download_prompt_sources.py --swe-traces qwen38_flash_next \
    --output-dir ./cache/dataset/prompts
```
