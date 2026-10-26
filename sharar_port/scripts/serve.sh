#!/usr/bin/env bash
# serve.sh <model: 27b|flashnext> <gpus> <port> <tag> [extra sglang args...]
# SGLang (relaxed-verify fork) + the pinned DFlash2 draft. Env: CTX, MAXRUN, BLOCK, PYTHONPATH_FORK.
source /data/ccchang/port_scripts/env.sh
export PYTHONPATH=${PYTHONPATH_FORK:-/data/ccchang/sglang-relax} SGLANG_QSA_RING_GROUPS=3
model=$1; gpus=$2; port=$3; tag=$4; shift 4
H=/data/ccchang/hf_home/hub
if [ "$model" = 27b ]; then
  TGT=$H/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
  DRAFT=$H/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4
  TP=${TP:-1}; BLOCK=${BLOCK:-16}
else
  TGT=${TGT_OVERRIDE:-$H/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594}
  DRAFT=$H/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6
  TP=${TP:-2}; BLOCK=${BLOCK:-8}
fi
R=/data/ccchang/sharar_port/results; mkdir -p $R
export SGLANG_DFLASH_RELAXED_STATS=$R/stats_${tag}_${port}.json  # one file a server: two servers of an arm raced on one
cd /data/ccchang/sharar_port
CUDA_VISIBLE_DEVICES=$gpus FLASHINFER_DISABLE_VERSION_CHECK=1 exec python -m sglang.launch_server \
  --model-path "$TGT" --tp-size $TP \
  --attention-backend fa3 --enforce-disable-flashinfer-allreduce-fusion \
  --host 127.0.0.1 --port "$port" --context-length ${CTX:-73728} --disable-radix-cache \
  --mem-fraction-static ${MEMFRAC:-0.85} --max-running-requests ${MAXRUN:-32} \
  --linear-attn-verify-backend triton --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT" \
  --speculative-dflash-block-size "$BLOCK" --speculative-draft-model-quantization unquant "$@"
