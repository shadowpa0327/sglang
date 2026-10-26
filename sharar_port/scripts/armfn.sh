#!/usr/bin/env bash
# armfn.sh <arm: exact|fast|w3|share|relax> <gpus a,b> <port>: one Flash-Next eval arm (TP2, DFlash2 block 8).
arm=$1; gpus=$2; port=$3
W3=/data/ccchang/hf_home/hub/models--yemara--Qwen3.8-Flash-Next-NVFP4-sharar-w3/snapshots/694e2790772b36297b52bb1dffdc33281d766671
export SGLANG_SHARAR_W3_REVISION=7b719225242aacd3dbd3f9407468c2ee9a9d2594
# sharar fast on Flash-Next + DFlash2: 3-bit trunk and routed experts, budgeted experts keep 4 / 12 a request / 64 a call,
# relaxed top-2 within 1.0 logits (settings.py profile_defaults, proposer dflash2)
case $arm in
  exact) ;;
  fast)  export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  w3)    export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 ;;
  share) export SGLANG_SHARAR_MOE=4:12:64 ;;
  relaxshare) export SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;  # fast without the 3-bit weights
  w3recal)  export SGLANG_SHARAR_W3=/data/ccchang/sharar_port/recal/w3_think SGLANG_SHARAR_W3_EXPERTS=1 ;;  # re-fit on thinking traces
  w3trunk)   export SGLANG_SHARAR_W3=$W3 ;;                                                  # 3-bit trunk, NVFP4 experts
  w3experts) export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_W3_TRUNK=0 ;;  # 3-bit experts, BF16 trunk
  relax) export SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
CTX=${CTX:-40960} MAXRUN=${MAXRUN:-16} exec /data/ccchang/sharar_port/scripts/serve.sh flashnext $gpus $port fn-$arm \
  --served-model-name qwen38-flashnext --disable-shared-experts-fusion
