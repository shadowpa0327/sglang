#!/usr/bin/env bash
# arm27.sh <arm: exact|fast|w3|relax> <gpu> <port>: one 27B eval arm (sharar fast's pieces on SGLang, H100).
arm=$1; gpu=$2; port=$3
W3=/data/ccchang/hf_home/hub/models--yemara--Qwen3.8-27B-NVFP4-sharar-w3/snapshots/72997f2aadf229bdc39603d802df61b367078a91
export SGLANG_SHARAR_W3_REVISION=319f741cce68d7914884900c138a1fbb70a42f30
case $arm in
  exact) ;;
  fast)  export SGLANG_SHARAR_W3=$W3 SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  w3)    export SGLANG_SHARAR_W3=$W3 ;;
  relax) export SGLANG_DFLASH_RELAXED_TOP_K=2 SGLANG_DFLASH_RELAXED_MARGIN=1.0 ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
# sharar's 27B sampled rule is top-2 within 1.0 logits (settings.py ACCEPT_DEFAULT_SAMPLED_THRESHOLD); requests are sampled
CTX=${CTX:-40960} MAXRUN=${MAXRUN:-16} exec /data/ccchang/sharar_port/scripts/serve.sh 27b $gpu $port 27b-$arm --served-model-name qwen38-27b
