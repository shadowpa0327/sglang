#!/usr/bin/env bash
# Up to four Nemotron-3.5-Super configurations side by side on one beta node (priority QoS, 4 GPUs), one GPU each (TP1:
# the W3 kernel needs N and K multiples of 128, and TP2 splits the Mamba in_proj's 128-row dt segment to 64 a rank), smoke
# prompts + GSM8K 100 greedy, then stopped. A configuration is space-separated env for serve_nemotron.sh, e.g. "PLAIN=1",
# "W3=1" or "SPEC=mtp W3=1 MOE=4:12:64"; "-" leaves that GPU idle. Code: this worktree (PYTHONPATH_FORK).
# Usage: sbatch nemotron_pair.sbatch "<config on GPU 0>" ["<GPU 1>" ["<GPU 2>" ["<GPU 3>"]]]
#SBATCH -J nemo-pair
#SBATCH -p beta
#SBATCH --qos=priority
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH -t 4:00:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/nemotron_pair_%j.out
WT=/ddn/scratch/cornell/cchang/sglang-nemotron
PORT_ROOT=$HOME/sglang/sharar_port  # logs and results only; the code comes from the worktree
source $PORT_ROOT/scripts/env_gb200.sh
export PYTHONPATH_FORK=$WT SGLANG_SHARAR_W3_CHECK=${SGLANG_SHARAR_W3_CHECK:-1}

run_cfg() {  # run_cfg "<env>" <gpus> <port>
  local cfg=$1 gpus=$2 port=$3 name pid
  name=nemotron-$(echo "$cfg" | sed 's/ /_/g; s/=/-/g; s/:/./g')-$SLURM_JOB_ID  # one log per configuration
  local log=$PORT_ROOT/logs/$name.log
  echo "[$(date -Is)] [$cfg] gpus $gpus port $port log $log"
  (export $cfg; exec setsid $WT/sharar_port/scripts/serve_nemotron.sh $gpus $port) > $log 2>&1 &
  pid=$!
  until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
    kill -0 $pid 2>/dev/null || { echo "[$cfg] server died"; grep -v "Ignore import" $log | tail -40 | sed "s/^/[$cfg] /"; return 1; }
    sleep 15; done
  echo "[$(date -Is)] [$cfg] ready"
  python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port | cut -c1-300 | sed "s/^/[$cfg] /"
  python -I $PORT_ROOT/scripts/gsm8k_quick.py http://127.0.0.1:$port 100 $PORT_ROOT/results/gsm8k_$name.json | sed "s/^/[$cfg] /"
  grep -oE "accept len: [0-9.]+" $log | tail -2 | sed "s/^/[$cfg] /"
  grep -E "sharar" $log | cut -c1-300 | head -12 | sed "s/^/[$cfg] /"
  kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
}

for g in 0 1 2 3; do
  cfg=${@:$((g + 1)):1}
  [ "${cfg:--}" = - ] || run_cfg "$cfg" $g $((30100 + 100 * g)) &
done
wait
echo "[$(date -Is)] done"
