#!/usr/bin/env bash
# Quick check of Nemotron-3.5-Super on the test partition (2 GPUs, TP2): serve (SPEC=mtp for the MTP head), smoke prompts,
# GSM8K 100 greedy. Usage: sbatch [--export=ALL,SPEC=mtp] nemotron_quick.sbatch; 3-bit: --export=ALL,W3=1,PYTHONPATH_FORK=<worktree>
# (SGLANG_SHARAR_W3_CHECK=1: every 3-bit matrix and expert block against the served weights at load).
#SBATCH -J nemo-quick
#SBATCH -p beta_test
#SBATCH --qos=test
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH -t 1:30:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/nemotron_quick_%j.out
PORT_ROOT=$HOME/sglang/sharar_port; port=30100; tag=nemotron-${SPEC:-none}$([ "${W3:-0}" = 1 ] && echo -w3)-$SLURM_JOB_ID
source $PORT_ROOT/scripts/env_gb200.sh
log=$PORT_ROOT/logs/$tag.log
echo "[$(date -Is)] SPEC=${SPEC:-none} W3=${W3:-0} PYTHONPATH_FORK=${PYTHONPATH_FORK:-} log $log"
export SGLANG_SHARAR_W3_CHECK=${SGLANG_SHARAR_W3_CHECK:-1}
setsid $PORT_ROOT/scripts/serve_nemotron.sh 0,1 $port > $log 2>&1 &
pid=$!
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
  kill -0 $pid 2>/dev/null || { echo "server died"; grep -v "Ignore import" $log | tail -40; exit 1; }; sleep 15; done
echo "[$(date -Is)] ready"
python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port | cut -c1-300
python -I $PORT_ROOT/scripts/gsm8k_quick.py http://127.0.0.1:$port 100 $PORT_ROOT/results/gsm8k_$tag.json
grep -oE "accept len: [0-9.]+" $log | tail -2
grep -E "sharar" $log | cut -c1-300 | head -12
kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
echo "[$(date -Is)] done"
