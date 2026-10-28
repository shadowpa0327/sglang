# sharar's lossy decode techniques on SGLang (H100)

This branch runs sharar's `fast` techniques inside SGLang so their accuracy can be measured on H100: the calibrated
3-bit (W3) decode copies of the trunk and routed experts, Flash-Next's budgeted expert selection (expert sharing), and
relaxed DFlash2 verification. Targets: Qwen3.8-27B-NVFP4 (`RadixArk/...@319f741`) and Qwen3.8-Flash-Next-NVFP4
(`RadixArk/...@7b71922`), each with its incoai DFlash2 draft.

## Code

- `sglang/srt/layers/quantization/sharar_w3/`: sharar's W3 kernel (`w3_gemm.cu`, `w3.py`, an SM90 path that converts the
  activation fragment to FP16 once instead of ptxas's per-mma FP8 emulation, and an H100 dispatch table
  `h100_tuning.json`), `route.py` (`route_budget`), `w3_moe.py` (3-bit routed experts), vendored from sharar @0ecb67f,
  and `attach.py`, which loads sharar's W3 checkpoints, shards them like the SGLang linears they replace, and wraps
  those linears' quant methods.
- `sglang/srt/model_executor/model_runner.py` `_attach_sharar_w3`: attaches everything after weight load, before graph
  capture, and sets the decode/verify mode per forward.
- `sglang/srt/speculative/dflash_relaxed.py`: relaxed verification (top-k within a logit margin, greedy and sampled).

## Environment

| Variable | Effect |
|---|---|
| `SGLANG_SHARAR_W3=<dir>` | the sharar W3 checkpoint (`manifest.json`) to serve: decode and target-verify forwards, and any call of at most 128 rows (27B) or 64 rows (Flash-Next), read the 3-bit copies |
| `SGLANG_SHARAR_W3_REVISION=<rev>` | refuse a checkpoint calibrated for another target revision |
| `SGLANG_SHARAR_W3_TRUNK=0` | with `SGLANG_SHARAR_W3_EXPERTS=1`: the 3-bit routed experts only |
| `SGLANG_SHARAR_W3_EXPERTS=1` | Flash-Next: serve the checkpoint's 3-bit routed experts (the NVFP4 experts are released) |
| `SGLANG_SHARAR_MOE=keep:per_request:budget` | Flash-Next expert sharing in decode/verify forwards (sharar fast: `4:12:64`) |
| `SGLANG_DFLASH_RELAXED_TOP_K`, `SGLANG_DFLASH_RELAXED_MARGIN` | server-default relaxed rule (sharar's sampled rule: `2`, `1.0`) |
| `SGLANG_DFLASH_RELAXED_SOFT=<s>` | sampled requests: a plausible draft keeps its coin, scaled by e^-s, and is accepted when `coin * q < e^s * p` instead of always, so no token comes out more than e^s times as often as the target's law gives it (`0`: exact sampling; unset: the forced accept); per request `accept_soft` in `custom_params` |
| `SGLANG_DFLASH_RELAXED_TOP=0` | sampled requests: the target's own top token keeps the ordinary test and only near misses get the relaxed pass; per request `relax_top` |
| `SGLANG_SHARAR_W3_CHECK=1` | at attach, every 3-bit linear and expert block against the original weights on random inputs (fails above 0.6 relative error) |
| `SGLANG_SHARAR_CAPTURE=<dir>` | record GPTQ inputs in sharar's formats (trunk covariances, MoE input rows); saved when `<dir>/SAVE` appears |

Flash-Next needs `--disable-shared-experts-fusion` with the 3-bit experts. A server logs `sharar W3: decode coverage N/N`
once every attached 3-bit linear has served a decode/verify call.

## Experiments (`sharar_port/scripts`)

- `serve.sh`, `arm27.sh`, `armfn.sh`: one server per arm (exact, w3, relax, share, fast, ...).
- `eval_custom.sh`, `runq.py`: sharar-eval GPQA runs (thinking on, T=1.0, top-p 0.95, top-k 20) per cap and seed.
- `seed_noise.py`, `cap_compare.py`: seed-to-seed spread and paired arm comparisons per cap.
- `build_calib.py`, `capture_serve.sh`, `capture_client.py`, `fit_trunk.py`, `recal_pipeline.sh`: re-fit Flash-Next's
  3-bit copies on the model's own thinking traces with sharar's GPTQ.
- `check_w3_mapping.py`, `test_w3_dispatch.py`, `w3_tune.py`, `bench_w3_moe.py`: validation and kernel tuning.
- `test_relaxed_soft.py`: the relaxed rule's output law at one verify slot (exact, soft, forced) through the DFlash chain
  kernel and the MTP target-only kernel: `PYTHONPATH=<checkout> python sharar_port/scripts/test_relaxed_soft.py` (the
  kernel tests skip without a GPU).

## Draft training data (`sharar_port/draft_data_regen`)

Regenerating speculative-decoding draft training data with the target model itself: the Qwen3.8-Flash-Next thinking
corpus and its agent sessions, and the same corpus for another target.

- `README.md`: the runbook. `CHECKLIST.md`: its steps and rules on one page.
- `REPRODUCE.md`: the recipe that rebuilds the corpus from the prompt lock in `locks/`.
- `PROMPT_SOURCES.md`: the prompt sources, the complete corpus and reference blends.
- `scripts/`: prompt download, regeneration driver and training-row builders.

## Results so far (GPQA Diamond, 3 seeds per cap, paired against the same model's exact arm)

| Model, arm | 16K | 32K | 64K |
|---|---|---|---|
| 27B exact | 62.3% | 78.1% | 84.5% |
| 27B 3-bit trunk | −3.2 [−6.7, +0.3] | −5.7 [−9.1, −2.7] | −4.0 [−7.2, −1.0] |
| 27B relaxed 2:1.0 | −0.3 [−2.5, +1.9] | −1.0 [−3.2, +1.0] | −1.5 [−3.5, +0.5] |
| Flash-Next exact | 62.8% | 79.0% | 86.4% |
| Flash-Next 3-bit trunk + experts* | −5.9 [−9.6, −2.5] | −7.1 [−10.4, −3.9] | −5.7 [−8.9, −2.7] |

\* Before commit 15e1811, which serves the GDN `in_proj_qkvz` 3-bit copies that SGLang's fused BF16 qkvz+ba GEMM
bypassed (159 of 195 trunk matrices were served); the corrected run is pending.

## Differences from sharar on GB10

NVFP4 runs as Marlin W4A16 on H100; Flash-Next's exact trunk is BF16 (sharar: FP8); the drafter's 3-bit copies, NVFP4 KV
and adaptive draft depth are not ported (fixed blocks: 16 on the 27B, 8 on Flash-Next); at TP2 Flash-Next's shared-expert
down projection (K 320 a rank) uses its dequantized 3-bit weights with sharar's activation rounding.
