#!/usr/bin/env bash
# Quick check of one Flash-Next arm on the test partition (2 GPUs, < 2 h): smoke prompts, GSM8K 100 (greedy), GPQA's first
# LIMIT items at 248K (T=1.0, seed 0, the paired subset of the full runs); prints the acceptance length and relaxed counts.
# Usage: sbatch fn_quick.sbatch <arm>
#SBATCH -J fn-quick
#SBATCH -p beta_test
#SBATCH --qos=test
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH -t 2:00:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/fn_quick_%j.out
arm=${1:?arm}; port=30100; LIMIT=${LIMIT:-100}
PORT_ROOT=$HOME/sglang/sharar_port
export ENV_SH=$PORT_ROOT/scripts/env_gb200.sh PORT_ROOT CTX=$((256 * 1024)) MAXRUN=16
source $ENV_SH
log=$PORT_ROOT/logs/quick_${arm}_${SLURM_JOB_ID}.log tag=quick-$arm-$SLURM_JOB_ID
echo "[$(date -Is)] $arm: log $log"
SGLANG_SHARAR_W3_CHECK=0 setsid $PORT_ROOT/scripts/armfn.sh $arm 0,1 $port > $log 2>&1 &
pid=$!
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
  kill -0 $pid 2>/dev/null || { echo "server died"; grep -v "Ignore import" $log | tail -30; exit 1; }; sleep 15; done
echo "[$(date -Is)] ready"
python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port think | cut -c1-200
python -I $PORT_ROOT/scripts/gsm8k_quick.py http://127.0.0.1:$port 100 $PORT_ROOT/results/gsm8k_$tag.json
(cd /ddn/scratch/cornell/cchang/sharar-eval && SHARAR_RESULTS=/ddn/scratch/cornell/cchang/sharar_results VIRTUAL_ENV= ~/.local/bin/uv run --frozen \
  sharar-eval run --url http://127.0.0.1:$port --plain --model qwen38-flashnext --label $tag --tasks gpqa --limit $LIMIT --thinking \
  --temperature 1.0 --seed 0 --max-tokens $((248 * 1024)) --concurrency 16 --engine "sglang sharar-port gb200 $arm" \
  --out $PORT_ROOT/runs/_quick/$tag) 2>&1 | tail -2
echo "== acceptance"; grep -oE "accept len: [0-9.]+|accept_len[^,]*|MTP relaxed acceptance: .*" $log | tail -3
grep -E "speculative_algorithm|speculative_num_steps" $log | grep -oE "'speculative_(algorithm|num_steps|eagle_topk|num_draft_tokens)': [^,]*" | sort -u
kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
echo "[$(date -Is)] done"
