#!/usr/bin/env bash
# capture_serve.sh <gpus> <port> <capture dir>: Flash-Next exact (BF16 trunk, NVFP4 experts) at TP2 without a draft or CUDA
# graphs, recording the 3-bit fit's activations (SGLANG_SHARAR_CAPTURE, attach.py `_Capture`).
source /data/ccchang/port_scripts/env.sh
export PYTHONPATH=/data/ccchang/sglang-relax SGLANG_QSA_RING_GROUPS=3 SGLANG_SHARAR_CAPTURE=$3
TGT=/data/ccchang/hf_home/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
cd /data/ccchang/sharar_port
CUDA_VISIBLE_DEVICES=$1 FLASHINFER_DISABLE_VERSION_CHECK=1 exec python -m sglang.launch_server --model-path $TGT --tp-size 2 \
  --attention-backend fa3 --enforce-disable-flashinfer-allreduce-fusion --host 127.0.0.1 --port $2 \
  --context-length 8192 --disable-radix-cache --mem-fraction-static 0.70 --max-running-requests 4 \
  --linear-attn-verify-backend triton --mamba-ssm-dtype bfloat16 --disable-cuda-graph --disable-shared-experts-fusion
