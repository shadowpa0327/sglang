#!/usr/bin/env bash
# run_arm.sh <arm> <gpus> <port> <tag>: serve one Flash-Next arm, smoke + GSM8K quick, stop it (inside an allocation).
arm=$1; gpus=$2; port=$3; tag=$4
PORT_ROOT=$HOME/sglang/sharar_port
export ENV_SH=$PORT_ROOT/scripts/env_gb200.sh PORT_ROOT
source $ENV_SH
log=$PORT_ROOT/logs/fn_${arm}_${tag}.log
echo "[$(date -Is)] $arm: gpus $gpus port $port log $log"
SGLANG_SHARAR_W3_CHECK=$([ $arm = exact ] && echo 0 || echo 1) setsid $PORT_ROOT/scripts/armfn.sh $arm $gpus $port > $log 2>&1 &
pid=$!
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
  kill -0 $pid 2>/dev/null || { echo "[$arm] server died"; grep -v "Ignore import" $log | tail -30; exit 1; }; sleep 15; done
echo "[$(date -Is)] $arm: ready"
python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port | sed "s/^/[$arm] /"
python -I $PORT_ROOT/scripts/smoke.py http://127.0.0.1:$port think | sed "s/^/[$arm] /"
python -I $PORT_ROOT/scripts/gsm8k_quick.py http://127.0.0.1:$port ${GSM8K_N:-200} $PORT_ROOT/results/gsm8k_fn_${arm}_${tag}.json | sed "s/^/[$arm] /"
grep -E "sharar|relative error" $log | cut -c1-300 | head -20 | sed "s/^/[$arm] /"
kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
