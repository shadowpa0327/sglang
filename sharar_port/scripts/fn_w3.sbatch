#!/usr/bin/env bash
# Flash-Next W3 on GB200 (TP2): one server per arm on its own GPU pair, in parallel (beta's QoS needs 4 GPUs a job);
# smoke + GSM8K quick on each, then stop them. Usage: sbatch fn_w3.sbatch [arm_on_gpus_0,1 [arm_on_gpus_2,3]]
#SBATCH -J fn-w3
#SBATCH -p beta
#SBATCH --qos=standard
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH -t 4:00:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/fn_w3_%j.out
PORT_ROOT=$HOME/sglang/sharar_port
export ENV_SH=$PORT_ROOT/scripts/env_gb200.sh PORT_ROOT
source $ENV_SH
mkdir -p $PORT_ROOT/logs $PORT_ROOT/results
N=${GSM8K_N:-200}

run_arm() {  # run_arm <arm> <gpus> <port>
  local arm=$1 gpus=$2 port=$3 log=$PORT_ROOT/logs/fn_${1}_${SLURM_JOB_ID}.log pid
  echo "[$(date -Is)] $arm: gpus $gpus port $port log $log"
  SGLANG_SHARAR_W3_CHECK=$([ $arm = exact ] && echo 0 || echo 1) setsid $PORT_ROOT/scripts/armfn.sh $arm $gpus $port > $log 2>&1 &
  pid=$!
  until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
    kill -0 $pid 2>/dev/null || { echo "[$arm] server died"; tail -40 $log; return 1; }; sleep 15; done
  echo "[$(date -Is)] $arm: ready"
  python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port | sed "s/^/[$arm] /"
  python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port think | sed "s/^/[$arm] /"
  python -I $PORT_ROOT/scripts/gsm8k_quick.py http://127.0.0.1:$port $N $PORT_ROOT/results/gsm8k_fn_${arm}_${SLURM_JOB_ID}.json | sed "s/^/[$arm] /"
  grep -E "sharar|relative error" $log | cut -c1-400 | head -20 | sed "s/^/[$arm] /"
  kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
}

run_arm ${1:-w3} 0,1 30100 &
run_arm ${2:-exact} 2,3 30200 &
wait
echo "[$(date -Is)] done"
