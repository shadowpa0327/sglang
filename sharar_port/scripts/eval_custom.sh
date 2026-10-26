#!/usr/bin/env bash
# eval_custom.sh <tag> <port> <model> <tasks> <max_tokens> <seed> <concurrency>: one sharar-eval run (thinking on, T=1.0).
tag=$1; port=$2; model=$3; tasks=$4; maxtok=$5; seed=$6; conc=$7
url=http://127.0.0.1:$port
export SHARAR_RESULTS=/data/ccchang/sharar_port/sharar_results HF_HOME=/data/ccchang/hf_home UV_CACHE_DIR=/data/ccchang/uv_cache
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $url/health)" = 200 ]; do
  pgrep -f "[s]glang.launch_server.*--port $port " >/dev/null || { echo "[$(date -Is)] server on $port died"; exit 1; }
  sleep 20
done
echo "[$(date -Is)] $tag eval starts"
cd /home/ccchang/sharar-eval
/data/ccchang/venvs/sf/bin/uv run sharar-eval run --url $url --plain --model $model --label $tag --tasks $tasks \
  --thinking --temperature 1.0 --seed $seed --max-tokens $maxtok --concurrency $conc \
  --engine "sglang-relax sharar-port@$(git -C /data/ccchang/sglang-relax rev-parse --short HEAD) $tag" \
  --out /data/ccchang/sharar_port/runs/$tag
echo "[$(date -Is)] EVAL_DONE $tag rc=$?"
