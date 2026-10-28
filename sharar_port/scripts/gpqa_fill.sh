#!/usr/bin/env bash
# gpqa_fill.sh <arm> <gpus> <port> <ctx_k> "<caps k>" "<seeds>": serve one arm on idle GPUs inside a running job
# (srun --jobid=<id> --overlap) and run sharar-eval GPQA for each seed and cap, as fn_gpqa.sbatch does.
arm=$1; gpus=$2; port=$3; ctx=$4; caps=$5; seeds=$6
PORT_ROOT=$HOME/sglang/sharar_port
export ENV_SH=$PORT_ROOT/scripts/env_gb200.sh PORT_ROOT
source $ENV_SH
EVAL=/ddn/scratch/cornell/cchang/sharar-eval
export SHARAR_RESULTS=/ddn/scratch/cornell/cchang/sharar_results CTX=$((ctx * 1024)) MAXRUN=16
COMMIT=$(git -C $HOME/sglang rev-parse --short HEAD)$(git -C $HOME/sglang diff --quiet || echo -dirty)
log=$PORT_ROOT/logs/gpqa_${arm}_fill_${SLURM_JOB_ID}.log
echo "[$(date -Is)] $arm: gpus $gpus port $port ctx $CTX log $log"
setsid $PORT_ROOT/scripts/armfn.sh $arm $gpus $port > $log 2>&1 &
pid=$!
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
  kill -0 $pid 2>/dev/null || { echo "[$arm] server died"; grep -v "Ignore import" $log | tail -30; exit 1; }; sleep 15; done
echo "[$(date -Is)] $arm: ready"
grep -E "sharar MoE|sharar W3 attached" $log | cut -c1-200
for seed in $seeds; do
  for cap in $caps; do
    tag=fn-$arm-g${cap}k-s$seed
    echo "[$(date -Is)] $tag: start"
    (cd $EVAL && VIRTUAL_ENV= ~/.local/bin/uv run --frozen sharar-eval run --url http://127.0.0.1:$port --plain --model qwen38-flashnext \
      --label $tag --tasks gpqa --thinking --temperature 1.0 --seed $seed --max-tokens $((cap * 1024)) \
      --concurrency 16 --engine "sglang sharar-port@$COMMIT gb200 $arm" --out $PORT_ROOT/runs/$tag) > $PORT_ROOT/logs/eval_${tag}.log 2>&1
    echo "[$(date -Is)] $tag: rc=$? $(tail -1 $PORT_ROOT/logs/eval_${tag}.log | cut -c1-200)"
  done
done
grep -E "expert sharing|relaxed" $log | head -3 | cut -c1-200
kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
echo "[$(date -Is)] done"
