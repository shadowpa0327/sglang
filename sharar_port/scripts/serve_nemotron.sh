#!/usr/bin/env bash
# serve_nemotron.sh <gpus> <port> [extra sglang args...]: NVIDIA-Nemotron-3.5-Super (NVFP4, revision 6aae0d6, the one the
# sharar W3 copies are fitted against) on SGLang, TP over the given GPUs. Env: SPEC (none: no drafter, default; mtp: the
# checkpoint's MTP head, MTP_STEPS chained top-1 drafts), W3 (1: sharar's 3-bit trunk and routed experts,
# shadowpa0327/NVIDIA-Nemotron-3.5-Super-NVFP4-sharar-w3@2ca8b35; needs the Nemotron wiring, branch sharar-nemotron,
# PYTHONPATH_FORK=/ddn/scratch/cornell/cchang/sglang-nemotron until merged), MOE (expert sharing keep:per_request:budget),
# SSM (Mamba state dtype, float32) and SSM_SR=1 (FP16 with stochastic rounding, sharar fast's), CTX (262144, NVIDIA's max-model-len), MAXRUN, ENV_SH, HF_HUB. Numerics as NVIDIA's published serving (NeMo Gym
# nemotron_recipes/super-3.5): FP8 KV cache, FP32 Mamba SSM cache, the nemotron_3 reasoning parser (vLLM's nemotron_v3).
# --trust-remote-code: the checkpoint's own transformers config / processor / model
# classes (reviewed: no network, exec or subprocess; HF_HUB_OFFLINE keeps them from fetching anything).
ENV_SH=${ENV_SH:-$HOME/sglang/sharar_port/scripts/env_gb200.sh}
_fork=${PYTHONPATH_FORK:-}  # the env file sets PYTHONPATH_FORK=~/sglang: keep a caller's (the worktree)
[ -f "$ENV_SH" ] && source "$ENV_SH"
[ -n "$_fork" ] && PYTHONPATH_FORK=$_fork
gpus=$1; port=$2; shift 2
H=${HF_HUB:-/ddn/scratch/cornell/cchang/hf_home/hub}
TGT=$H/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/6aae0d6d0285e561bbb72efa10ba1425d41ebddc
export PYTHONPATH=${PYTHONPATH_FORK:-$HOME/sglang}
if [ "${W3:-0}" = 1 ]; then
  export SGLANG_SHARAR_W3=$H/models--shadowpa0327--NVIDIA-Nemotron-3.5-Super-NVFP4-sharar-w3/snapshots/2ca8b352b0fbff1620117f86c023552e761a7e32
  export SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_W3_REVISION=6aae0d6d0285e561bbb72efa10ba1425d41ebddc
fi
[ -n "${MOE:-}" ] && export SGLANG_SHARAR_MOE=$MOE
TP=$(( $(echo $gpus | tr -cd , | wc -c) + 1 ))
LILICORR=$H/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr/snapshots/292bf1efc06ba39567da7351f2ad7a522afa3959
if [ "${SPEC:-none}" = lilicorr ]; then  # NVIDIA's LiLiCorr drafter (block 8: up to 7 drafts), served through SGLang's DFLASH path
  # the drafter is ModelOpt W4A16_NVFP4 (fc and the LiLiCorr head packed); without an explicit draft quantization the
  # draft inherits the target's modelopt_mixed, builds an unquantized fc and refuses the packed weight
  SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "$LILICORR" --speculative-dflash-block-size ${BLOCK:-8}
             --speculative-draft-model-quantization modelopt_fp4)
elif [ "${SPEC:-none}" = mtp ]; then
  SPEC_ARGS=(--speculative-algorithm NEXTN --speculative-draft-model-path "$TGT" --speculative-num-steps ${MTP_STEPS:-3}
             --speculative-eagle-topk 1 --speculative-num-draft-tokens $(( ${MTP_STEPS:-3} + 1 )))
else
  SPEC_ARGS=()
fi
CUDA_VISIBLE_DEVICES=$gpus FLASHINFER_DISABLE_VERSION_CHECK=1 exec python -m sglang.launch_server \
  --model-path "$TGT" --tp-size $TP --served-model-name nemotron35-super --trust-remote-code \
  --attention-backend ${ATTN:-trtllm_mha} --host 127.0.0.1 --port "$port" \
  --context-length ${CTX:-262144} --disable-radix-cache --mem-fraction-static ${MEMFRAC:-0.85} \
  --reasoning-parser ${REASONING:-nemotron_3} --kv-cache-dtype ${KV:-fp8_e4m3} --mamba-ssm-dtype ${SSM:-float32} \
  $([ "${SSM_SR:-0}" = 1 ] && echo --enable-mamba-cache-stochastic-rounding) \
  --max-running-requests ${MAXRUN:-16} "${SPEC_ARGS[@]}" "$@"
