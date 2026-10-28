# Regenerating Draft Training Data: Checklist

This is the short version of the [runbook](README.md), for a
coding agent or a person producing on-policy training data for a
speculative-decoding draft (EAGLE3, DFlash, DFlash2, DSpark). Read the runbook
before starting; use this page to keep track. The
[prompt source catalog](PROMPT_SOURCES.md) describes the sources, the complete
corpus and the reference blends.

**The same corpus for another target** (the Qwen3.8-Flash-Next prompts,
efforts and settings): follow the [reproduction recipe](REPRODUCE.md)
step by step. It rebuilds the prompts from the lock in
`draft_data_regen/locks/qwen38-flash-next-think-stage1/`; do not sample new ones.

## Checklist

1. **Confirm resources.** Find out which GPUs and which disk to use, and how
   many tokens to generate. At an 8K budget, 1B generated tokens is about 470K
   conversations. Check `nvidia-smi` and running processes, and use only the
   GPUs the user assigned.
2. **Learn the target** (runbook step 1). Find its thinking toggle, the
   effort kwarg and its levels, how reasoning in earlier turns renders, the
   SGLang `--reasoning-parser`, the context length, the recommended sampling
   and the matching SpecForge `chat_template`.
3. **Download prompts**, decontaminated. For chat use `perfectblend ultrachat
   opencodeinstruct`; add `aya nemotron-if nemotron-structured` for coverage.
   Start with `python draft_data_regen/scripts/download_prompt_sources.py --list`, then run it
   with `--sources ... --sample ... --decontaminate-against <evals> --output-dir <big disk>`.
   For the same corpus, run `--lock <lock dir>` instead.
4. **Serve the target** with SGLang. Use one server per smallest TP group that
   fits, and add replicas for throughput. Tune `--max-running-requests` and
   the CUDA-graph batch size first: read `gen throughput` in the server log for
   about 5 minutes per setting.
5. **Regenerate** with `draft_data_regen/scripts/regen_driver.py <prompts> <out> <servers...> --model <target>`.
   For a thinking model add `--thinking --budget <stage max_length> --max-tokens <at least the budget>`,
   plus `--effort-mix ...`. Leave out `--effort-mix` for lock-built prompts,
   which already carry their effort. Run it in the background with a log;
   rerun the same command to resume.
6. **Check** the `_skipped` and `_error` counts, and read some sample rows.
7. **Build rows** with `draft_data_regen/scripts/build_regen_training_rows.py --tokenizer <target>
   --out-dir <dir> --budget <same> --verify-chat-template <name> <regen files>`.
   The verify line must say `(OK)`.
8. **Add agent sessions**, but only if the target wrote some in
   `nvidia/Open-SWE-Traces`. Download them with
   `draft_data_regen/scripts/download_prompt_sources.py --swe-traces <model>`, then run
   `draft_data_regen/scripts/build_swe_trace_rows.py --data-dir <dir>/open_swe_traces --model <model>
   --tokenizer <target> --out-dir <dir> --max-tokens 32768 --verify-chat-template <name>`.
   Low- and medium-effort sessions (the `thinking-*/` folders) are rendered at
   their own effort; `--efforts default` keeps only the xhigh ones. To rebuild
   our exact agent rows, add `--revisions-from <lock dir>/lock.json` to the
   download and pass `--harnesses openhands sweagent minisweagent` in that
   order.
9. **Hand off** the training fields.
   - Chat rows: `is_preformatted: true`, `train_only_last_turn: true`,
     `chat_template: <verified>`, `max_length: <budget>`.
   - Agent rows: the same, but with `train_only_last_turn: false` and
     `max_length: <--max-tokens>`, as a separate stage.

## Rules

- Never upload datasets or checkpoints without explicit approval.
- Never download the DaoCloud prompts (`DaoCloud/Qwen3.8-27B-Drafter-SFT`).
  Build mixes from the original sources.
- Never edit a running script. Never use `pkill -f`, because the pattern also
  matches your own shell; stop jobs by PID or process group.
- Keep caches and outputs on a large volume, and set `HF_HOME` there too.
- Use agent sessions (`--swe-traces`) as recorded, and only when the model that
  wrote them is the target. The driver does not regenerate tool-using
  sessions.
- Report progress as rows kept and estimated generated tokens against the
  goal, with an ETA from the driver's log.
