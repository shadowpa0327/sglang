#!/usr/bin/env bash
# serve_nemotron.sh <gpus> <port> [extra sglang args...]: NVIDIA-Nemotron-3.5-Super (NVFP4, revision 6aae0d6, the one the
# sharar W3 copies are fitted against) on SGLang, TP over the given GPUs. Env: SPEC (none: no drafter, default; mtp: the
# checkpoint's MTP head, MTP_STEPS chained top-1 drafts), W3 (1: sharar's 3-bit trunk and routed experts,
# shadowpa0327/NVIDIA-Nemotron-3.5-Super-NVFP4-sharar-w3@2ca8b35; needs the Nemotron wiring, branch sharar-nemotron,
# PYTHONPATH_FORK=/ddn/scratch/cornell/cchang/sglang-nemotron until merged), MOE (expert sharing keep:per_request:budget),
# CTX, MAXRUN, ENV_SH, HF_HUB.
ENV_SH=${ENV_SH:-$HOME/sglang/sharar_port/scripts/env_gb200.sh}
[ -f "$ENV_SH" ] && source "$ENV_SH"
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
if [ "${SPEC:-none}" = mtp ]; then
  SPEC_ARGS=(--speculative-algorithm NEXTN --speculative-draft-model-path "$TGT" --speculative-num-steps ${MTP_STEPS:-3}
             --speculative-eagle-topk 1 --speculative-num-draft-tokens $(( ${MTP_STEPS:-3} + 1 )))
else
  SPEC_ARGS=()
fi
CUDA_VISIBLE_DEVICES=$gpus FLASHINFER_DISABLE_VERSION_CHECK=1 exec python -m sglang.launch_server \
  --model-path "$TGT" --tp-size $TP --served-model-name nemotron35-super \
  --attention-backend ${ATTN:-trtllm_mha} --host 127.0.0.1 --port "$port" \
  --context-length ${CTX:-131072} --disable-radix-cache --mem-fraction-static ${MEMFRAC:-0.85} \
  --max-running-requests ${MAXRUN:-16} "${SPEC_ARGS[@]}" "$@"
