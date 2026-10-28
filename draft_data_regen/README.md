# Regenerating Draft Training Data: Runbook for Coding Agents

This runbook is written for a coding agent (or a person) asked to produce
on-policy training data for a speculative-decoding draft of a **new target
model**: download prompts, have the target answer them at scale, and turn the
answers into training rows. It distills a run that produced 1.01B generated
tokens for Qwen3.8-Flash-Next; every rule below comes from something that went
wrong or mattered there.

Pipeline and tools:

| Step | Tool | Output |
| --- | --- | --- |
| 1. Learn the target | the snippets below | facts that set every later flag |
| 2. Download prompts | `draft_data_regen/scripts/download_prompt_sources.py` | `<source>_prompts.jsonl`, `manifest.json` |
| 3. Serve the target | `python -m sglang.launch_server` | one or more OpenAI-compatible servers |
| 4. Regenerate | `draft_data_regen/scripts/regen_driver.py` | `<name>_regen.jsonl` (+ `_error`, `_skipped`) |
| 5. Build training rows | `draft_data_regen/scripts/build_regen_training_rows.py` | `train.jsonl`, `stats.json` |
| 6. Train | `specforge train` with the data fields below | draft checkpoints |

The prompt sources themselves are described in
[PROMPT_SOURCES.md](PROMPT_SOURCES.md).

## In this directory

| File | What it does |
| --- | --- |
| `scripts/download_prompt_sources.py` | downloads prompt sources from Hugging Face as regeneration input; samples, deduplicates, decontaminates, pins revisions |
| `scripts/regen_driver.py` | resumable multi-server regeneration with thinking budgets and reasoning-effort mixes |
| `scripts/build_regen_training_rows.py` | one preformatted last-turn-loss training row per answer, rendered with the target's template |
| `scripts/regen_common.py` | request and validation helpers vendored from SpecForge's regenerator |

Requirements: `openai`, `requests`, `transformers`, `datasets` and
`huggingface_hub`. SpecForge is needed only for training and for
`--verify-chat-template`. Any SGLang build serves the target; this branch's
build includes the speculative-capture patch used for training. The same tools
live in SpecForge as `scripts/{download_prompt_sources,regen_driver,build_regen_training_rows}.py`
with these documents under `docs/sections/basic_usage/`.

## Ground rules

- **Ask before taking GPUs.** Check `nvidia-smi` and running processes first;
  shared hosts often have other jobs. Use only GPUs the user assigned.
- **Keep data off small disks.** Prompt files, regenerations and HF caches run
  to tens of GB. Set `HF_HOME` to a large volume and write outputs there.
- **Do not upload or publish** datasets or checkpoints without the user's
  explicit approval.
- **Never edit a running script.** Bash reads scripts incrementally; copy it,
  or stop it first.
- **Kill by PID or process group, not `pkill -f <pattern>`.** The pattern
  also matches your own shell's command line, which then kills the tool call.
  Find PIDs with `ps -eo pid,pgid,args | awk '/regen_driver/ && !/awk/'`.
- **Long jobs go in the background** (`setsid nohup ... &`), with logs on disk;
  check progress from the logs instead of holding a terminal.

## 1. Learn the target

Answer these before choosing flags. Read the model card, then confirm against
the chat template itself:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
t = tok.chat_template
for key in ("enable_thinking", "reasoning_effort", "preserve_thinking", "reasoning_content", "tools"):
    print(key, key in t)
msgs = [{"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "Q2"}]
print(repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)))
```

| Question | Why it matters |
| --- | --- |
| Does it think, and how is thinking toggled (`enable_thinking`, a system prompt, a separate model)? | Decides `--thinking` and which serving mode you train for. Drafts trained on non-thinking answers lose about a third of their acceptance on thinking output. |
| Is there an effort control (kwarg name, values, default)? | Set `--effort-mix` and `--effort-kwarg`. Mixing efforts matches how users serve the target. |
| Does the template render earlier turns' reasoning (`preserve_thinking`, "last query index" logic)? | The driver sends earlier answers **without** reasoning, so the server renders them that way; the builder renders history the same way. Look at what the probe above prints for `R1`. |
| Which SGLang `--reasoning-parser` splits its output? | Without it, `reasoning_content` is empty and every thinking answer is skipped. |
| Context length, recommended sampling, tool format? | Sets `--context-length`, `--temperature/--top-p/--top-k`. Tools are not supported by the driver. |
| Which SpecForge `chat_template` matches it? | Needed by training and by `--verify-chat-template`. |

## 2. Choose the mix and download prompts

Size by **generated tokens**, not rows. Thinking answers average about 2K tokens
at an 8K budget (about 1.9K for general chat, 2.7K for code), so 1B tokens is
about 470K conversations. A first-stage blend that has worked:

| Share | Source | Notes |
| --- | --- | --- |
| 55-75% | `perfectblend` | backbone; multi-turn rows yield several training rows |
| 15-25% | `opencodeinstruct` | sample 250K; code answers are long and truncate more |
| 5-15% | `aya` | multilingual; add if the target serves non-English users |
| 5-10% | `nemotron-if`, `nemotron-structured` | instruction following, structured output |

Or start from `daocloud-prompts`, which is already filtered and decontaminated.
Always decontaminate against the evals you will report:

```bash
python draft_data_regen/scripts/download_prompt_sources.py \
    --sources perfectblend opencodeinstruct aya nemotron-if nemotron-structured \
    --sample opencodeinstruct=250000 aya=60000 \
    --decontaminate-against /path/to/eval_datasets \
    --output-dir /data/prompts
```

Check `manifest.json`: drop rates above a few percent for one eval set mean
real overlap (PerfectBlend loses about 0.6% to LiveCodeBench alone). Keep the
manifest with the data; it pins the dataset revisions.

## 3. Serve the target

Start from one server per smallest GPU group that fits the weights (TP), and
add replicas (more servers) for throughput; the driver balances across them.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path $TARGET --tp-size 2 --host 127.0.0.1 --port 31000 \
    --context-length 32768 --mem-fraction-static 0.88 \
    --max-running-requests 256 --cuda-graph-max-bs 256 \
    --reasoning-parser <parser> \
    > logs/regen_server_31000.log 2>&1 &
curl --fail http://127.0.0.1:31000/health
```

Tune before the long run. Generation throughput is bound by how many requests
decode at once, so the main lever is concurrency: raise `--max-running-requests`
(and the CUDA-graph batch size with it) until memory or the KV cache stops you.
Measure each candidate for about 5 minutes under the real driver load, reading
the server log's `gen throughput (token/s)` and `#running-req` lines. Keep the
prefix cache on (the default) for multi-turn data.

Measured on Qwen3.8-Flash-Next NVFP4, TP2 on H100 (hybrid linear-attention
model):

| Change | Generated tok/s per server |
| --- | --- |
| 96 running requests (baseline) | 3,136 |
| 192 running requests | 3,726 |
| + `--mamba-ssm-dtype bfloat16`, 256 requests, mem fraction 0.88 | **4,306** |
| FlashInfer or CuTe-DSL linear-attention decode kernels | 1,900-2,240 (worse) |
| Alternative MoE runner, multi-step decode | no gain |

Hybrid (Mamba/GDN) models cap concurrency by state slots; dense models by KV
cache. Other per-model knobs (e.g. where large embedding tables live) can make
the server fail to start; when a config fails, read the server log before
retrying.

## 4. Regenerate

```bash
python draft_data_regen/scripts/regen_driver.py /data/prompts/perfectblend_prompts.jsonl \
    /data/regen/perfectblend_regen.jsonl \
    127.0.0.1:31000 127.0.0.1:32000 \
    --model $TARGET --thinking --budget 8192 --max-tokens 16384 \
    --effort-mix low:0.4,medium:0.3,xhigh:0.3 --effort-kwarg reasoning_effort \
    --temperature 1.0 --top-p 0.95 --top-k 20 --concurrency 256 \
    >> logs/regen_perfectblend.log 2>&1 &
```

- `--budget` should equal the first training stage's `data.max_length`. Each
  turn gets `budget - prompt tokens`; answers cut there are kept as
  `"truncated": true` (at 8K we saw 6-9% at low and medium effort, 20% at
  xhigh, and 40% for xhigh code). Do not drop them: long-reasoning prefixes are
  exactly what the draft must learn.
- `--concurrency` per server should equal the server's
  `--max-running-requests`. For several prompt files, run one driver per file
  and split servers between them (or lower `--concurrency` when they share).
- **Resume** by rerunning the same command. To retry `_error`/`_skipped` rows,
  move those files aside first.
- **Server crashes**: requests aborted by a restart come back as
  `finish_reason="abort"` and are retried, never skipped. For multi-day runs,
  keep servers alive with a watchdog loop that relaunches a dead port from a
  saved command line.
- **Progress**: the log prints counts, rate and ETA every minute. Report
  progress as rows kept and estimated generated tokens against the target.
- **Stopping at a target**: count lines in the outputs (`wc -l`) and stop the
  drivers and servers by process group when reached; the outputs stay valid.

## 5. Check the data

Before building rows, sample and look:

- `_skipped`/`_error` counts should be tiny (under 0.1% here). A large count
  usually means a missing `--reasoning-parser`, a context too short for
  `--budget`, or a template kwarg the server rejects.
- Read 20 rows across efforts: reasoning present and plausible, answers not
  cut mid-sentence unless `truncated`, no `<think>` markers inside content.
- Truncation rate by effort and source (low effort should truncate least).

## 6. Build training rows

```bash
python draft_data_regen/scripts/build_regen_training_rows.py \
    --tokenizer $TARGET --out-dir /data/train/stage1_8k --budget 8192 \
    --decontaminate-against /path/to/eval_datasets \
    --verify-chat-template qwen3.5 \
    /data/regen/perfectblend_regen.jsonl /data/regen/opencodeinstruct_regen.jsonl
```

The builder writes one preformatted row per assistant turn, rendered with the
target's own template exactly as the server saw that turn (effort included,
earlier answers without reasoning, truncated answers without closing tags), and
drops repetition loops and over-budget rows. `--verify-chat-template` must
report `0 of N ... (OK)`: it parses a sample with SpecForge's preprocessing and
checks the loss covers exactly the last answer. If it fails, the SpecForge
template's assistant header does not match the target's generation prompt; fix
the template choice before training.

## 7. Train

```yaml
data:
  train_data_path: /data/train/stage1_8k/train.jsonl
  is_preformatted: true
  train_only_last_turn: true
  chat_template: qwen3.5        # the template --verify-chat-template accepted
  max_length: 8192              # = --budget
```

The trainer tokenizes on first launch and caches by `data.cache_key`; the
cache key does not record `train_only_last_turn`, so use a new key when that
changes.

## Pitfalls that cost time

- **JSONL with `str.splitlines()`** splits rows on U+2028 inside strings and
  corrupts them. Split on `"\n"`.
- **Templates that render history reasoning by default** (`preserve_thinking`):
  rendering whole regenerated conversations with all reasoning does not match
  what the server saw. Use the per-turn rows from the builder.
- **Budget accounting** must use the target's tokenizer and the same template
  kwargs as the request; the driver does this, so keep `--model` pointing at the
  served checkpoint.
- **Multiprocessing with Python's string `hash()`**: workers must be forked
  from the process that built hashed lookup tables (the scripts force `fork`).
- **Disaggregated training** with a large global batch needs the producer's
  in-flight high watermark at least the global batch size, and a failed launch
  leaves a control directory that must be removed before relaunching.

## Reference run

Qwen3.8-Flash-Next (NVFP4), thinking on, 8K budget, effort 40/30/30, three TP2
H100 servers at about 4.2K tok/s each: 470,624 conversations (PerfectBlend
348,841, OpenCodeInstruct 120,865, UltraChat 918) and 1.01B generated tokens in
about a day; 556,511 training rows after building. Truncated at 8K: 10.4% of
PerfectBlend and 18.2% of OpenCodeInstruct conversations. Drops: 1.3% of
PerfectBlend to decontamination, under 0.01% to repetition loops.
