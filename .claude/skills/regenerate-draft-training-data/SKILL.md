---
name: regenerate-draft-training-data
description: Produce on-policy training data for a speculative-decoding draft (EAGLE3, DFlash, DFlash2, DSpark) by having the target model regenerate answers to public prompts and adding agent sessions the target wrote. Use when asked to regenerate, re-generate or replay training data for a target model, build the complete training corpus, download prompt sources (PerfectBlend, UltraChat, OpenCodeInstruct, aya, Nemotron instruction following) or Open-SWE-Traces agent sessions, run thinking-mode regeneration with an effort mix or token budget, or turn regenerated conversations or agent sessions into SpecForge training rows.
---

# Regenerate draft training data

Full runbook: `draft_data_regen/README.md`. Prompt
source catalog and reference blends: `draft_data_regen/PROMPT_SOURCES.md`.
Read the runbook before starting; this file is the checklist. The complete corpus
(PerfectBlend, UltraChat and OpenCodeInstruct regenerated; Open-SWE-Traces sessions
as recorded) and the commands that rebuild it are in the source catalog.

## Checklist

1. **Confirm resources.** Which GPUs, which disk, how many generated tokens
   (about 470K conversations per 1B tokens at an 8K budget). Check
   `nvidia-smi` and running processes; use only GPUs the user assigned.
2. **Learn the target** (runbook step 1): thinking toggle, effort kwarg and
   values, how history reasoning renders, SGLang `--reasoning-parser`,
   context length, recommended sampling, matching SpecForge `chat_template`.
3. **Download prompts** (chat: `perfectblend ultrachat opencodeinstruct`, plus
   `aya nemotron-if nemotron-structured` for coverage) with decontamination:
   `python draft_data_regen/scripts/download_prompt_sources.py --list`, then
   `--sources ... --sample ... --decontaminate-against <evals> --output-dir <big disk>`.
4. **Serve the target** with SGLang (one server per smallest TP group that
   fits, replicas for throughput). Tune `--max-running-requests` and the
   CUDA-graph batch size first; measure `gen throughput` in the server log for
   ~5 minutes per setting.
5. **Regenerate**: `draft_data_regen/scripts/regen_driver.py <prompts> <out> <servers...> --model <target>`
   with `--thinking --budget <stage max_length> --effort-mix ...` for thinking
   models. Run it in the background with a log; rerun the same command to resume.
6. **Check** `_skipped`/`_error` counts and read sample rows.
7. **Build rows**: `draft_data_regen/scripts/build_regen_training_rows.py --tokenizer <target>
   --out-dir <dir> --budget <same> --verify-chat-template <name> <regen files>`.
   The verify line must say `(OK)`.
8. **Agent sessions**, only if the target wrote some in `nvidia/Open-SWE-Traces`:
   `draft_data_regen/scripts/download_prompt_sources.py --swe-traces <model>`, then
   `draft_data_regen/scripts/build_swe_trace_rows.py --data-dir <dir>/open_swe_traces --model <model>
   --tokenizer <target> --out-dir <dir> --max-tokens 32768 --verify-chat-template <name>`.
9. **Hand off** the training fields: chat rows `is_preformatted: true`,
   `train_only_last_turn: true`, `chat_template: <verified>`,
   `max_length: <budget>`; agent rows the same with `train_only_last_turn: false`
   and `max_length: <--max-tokens>`, as a separate stage.

## Rules

- Never upload datasets or checkpoints without explicit approval.
- Never edit a running script; never `pkill -f` (it matches your own shell);
  stop jobs by PID or process group.
- Keep caches and outputs on a large volume (`HF_HOME` too).
- Agent trajectories (`--swe-traces`) are used as is, only when the trajectory
  model is the target; the driver does not regenerate tool-using sessions.
- Report progress as rows kept and estimated generated tokens against the
  target, with an ETA from the driver's log.
