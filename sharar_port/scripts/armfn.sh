#!/usr/bin/env bash
# armfn.sh <arm: exact|fast|w3|share|relax> <gpus a,b> <port>: one Flash-Next eval arm (TP2, DFlash2 block 8).
arm=$1; gpus=$2; port=$3
PORT_ROOT=${PORT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
W3=${HF_HUB:-/data/ccchang/hf_home/hub}/models--yemara--Qwen3.8-Flash-Next-NVFP4-sharar-w3/snapshots/694e2790772b36297b52bb1dffdc33281d766671
W3QAD=${HF_HUB:-/data/ccchang/hf_home/hub}/models--yemara--Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad/snapshots/5c4c62a62853caebea8cedda7fbed952efa1effe
W3THINK=${HF_HUB:-/data/ccchang/hf_home/hub}/models--shadowpa0327--Qwen3.8-Flash-Next-NVFP4-sharar-w3-think/snapshots/508e2110ded141759d0d5bbcdd81a55ae8aa6080
export SGLANG_SHARAR_W3_REVISION=7b719225242aacd3dbd3f9407468c2ee9a9d2594
# sharar fast on Flash-Next + DFlash2: 3-bit trunk and routed experts, budgeted experts keep 4 / 12 a request / 64 a call,
# relaxed top-2 within 1.0 logits (settings.py profile_defaults, proposer dflash2)
case $arm in
  exact) ;;
  fast)  export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  w3)    export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 ;;
  share) export SGLANG_SHARAR_MOE=4:12:64 ;;
  relaxshare) export SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;  # fast without the 3-bit weights
  w3recal)  export SGLANG_SHARAR_W3=$PORT_ROOT/recal/w3_think SGLANG_SHARAR_W3_EXPERTS=1 ;;  # re-fit on thinking traces
  w3qad)    export SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 ;;  # QAD-trained trunk copies, the same routed experts
  w3think)  export SGLANG_SHARAR_W3=$W3THINK SGLANG_SHARAR_W3_EXPERTS=1 ;;  # GPTQ trunk + experts recalibrated on thinking traces
  fastqad)   export SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;  # fast with the QAD copies
  fastthink) export SGLANG_SHARAR_W3=$W3THINK SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;  # fast with the think copies
  # the fast arms with the MTP drafter: 3-bit copies + expert sharing, exact verification unless a K:M rule is set (eagle_utils relaxed patch)
  fastmtp)      export SPEC=mtp SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 ;;
  fastqadmtp)   export SPEC=mtp SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 ;;
  fastthinkmtp) export SPEC=mtp SGLANG_SHARAR_W3=$W3THINK SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 ;;
  fastqadmtp21) export SPEC=mtp SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;  # + relaxed 2:1 on the MTP chain
  fastqadmtp32) export SPEC=mtp SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_MOE=4:12:64 SGLANG_DFLASH_RELAXED_TOP_K=3 SGLANG_DFLASH_RELAXED_MARGIN=2.0 ;;  # + relaxed 3:2 on the MTP chain
  w3qadmtp)     export SPEC=mtp SGLANG_SHARAR_W3=$W3QAD SGLANG_SHARAR_W3_EXPERTS=1 ;;  # w3qad with the MTP drafter (no sharing)
  w3trunk)   export SGLANG_SHARAR_W3=$W3 ;;                                                  # 3-bit trunk, NVFP4 experts
  w3experts) export SGLANG_SHARAR_W3=$W3 SGLANG_SHARAR_W3_EXPERTS=1 SGLANG_SHARAR_W3_TRUNK=0 ;;  # 3-bit experts, BF16 trunk
  relax) export SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
CTX=${CTX:-40960} MAXRUN=${MAXRUN:-16} exec $PORT_ROOT/scripts/serve.sh flashnext $gpus $port fn-$arm \
  --served-model-name qwen38-flashnext --disable-shared-experts-fusion
