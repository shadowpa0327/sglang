#!/usr/bin/env bash
# GPQA Diamond (TASK=lcb_v6: LiveCodeBench v6) (sharar-eval, thinking on, T=1.0, top-p 0.95, top-k 20) on Flash-Next, GB200 TP2: one server per arm on
# its own GPU pair, in parallel; per arm every seed runs the caps in order (seed 0 at all caps first).
# Usage: sbatch fn_gpqa.sbatch [arm_on_gpus_0,1 [arm_on_gpus_2,3]]   Env: CAPS (k tokens), SEEDS, CONC, TASK,
# PAR_SEEDS=1 (every seed at once on the arm's server, for small tasks such as aime25). An arm may set its own task
# and seeds: arm:task:seeds[:par], seeds comma-separated (e.g. fastqad:lcb_v6:0,1  exact:aime25:0,1,2,3,4,5,6,7:par)
#SBATCH -J fn-gpqa
#SBATCH -p beta
#SBATCH --qos=priority
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH -t 1-00:00:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/fn_gpqa_%j.out
PORT_ROOT=$HOME/sglang/sharar_port
export ENV_SH=$PORT_ROOT/scripts/env_gb200.sh PORT_ROOT
source $ENV_SH
EVAL=/ddn/scratch/cornell/cchang/sharar-eval
export SHARAR_RESULTS=/ddn/scratch/cornell/cchang/sharar_results
CAPS=${CAPS:-248}; SEEDS=${SEEDS:-"0 1 2"}; CONC=${CONC:-16}; TASK=${TASK:-gpqa}  # gpqa or lcb_v6
MAXCAP=$(for c in $CAPS; do echo $c; done | sort -n | tail -1)
export CTX=$(( (MAXCAP + 8) * 1024 ))  # the longest cap plus room for the prompt
COMMIT=$(git -C $HOME/sglang rev-parse --short HEAD)$(git -C $HOME/sglang diff --quiet || echo -dirty)
mkdir -p $PORT_ROOT/logs $PORT_ROOT/results $PORT_ROOT/runs
FETCHED=$PORT_ROOT/logs/datasets_fetched_$SLURM_JOB_ID
(cd $EVAL && VIRTUAL_ENV= HF_HUB_OFFLINE=0 HF_TOKEN=$(cat $HF_HOME/token) ~/.local/bin/uv run --frozen sharar-eval datasets fetch > $PORT_ROOT/logs/datasets_fetch_$SLURM_JOB_ID.log 2>&1 \
  && touch $FETCHED || touch $FETCHED.failed; echo "[$(date -Is)] datasets fetch: $(ls $FETCHED* | xargs -n1 basename)") &

run_arm() {  # run_arm <arm[:task:seeds[:par]]> <gpus> <port>
  local spec=$1 gpus=$2 port=$3 pid seed cap tag
  local arm=${spec%%:*} TASK=$TASK SEEDS=$SEEDS PAR_SEEDS=${PAR_SEEDS:-0}
  if [[ $spec == *:* ]]; then
    IFS=: read -r _ TASK s par <<< "$spec"; SEEDS=${s//,/ }; [ "$par" = par ] && PAR_SEEDS=1
  fi
  local MAXRUN=$([ $PAR_SEEDS = 1 ] && echo $((CONC * $(echo $SEEDS | wc -w))) || echo $CONC)
  export MAXRUN
  local log=$PORT_ROOT/logs/${TASK}_${arm}_${SLURM_JOB_ID}.log
  echo "[$(date -Is)] $arm: task $TASK seeds $SEEDS par $PAR_SEEDS maxrun $MAXRUN"
  echo "[$(date -Is)] $arm: gpus $gpus port $port ctx $CTX log $log"
  setsid $PORT_ROOT/scripts/armfn.sh $arm $gpus $port > $log 2>&1 &
  pid=$!
  until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
    kill -0 $pid 2>/dev/null || { echo "[$arm] server died"; tail -40 $log; return 1; }; sleep 15; done
  echo "[$(date -Is)] $arm: ready"
  until [ -f $FETCHED ] || [ -f $FETCHED.failed ]; do sleep 30; done
  [ -f $FETCHED ] || { echo "[$arm] datasets fetch failed, no GPQA"; kill -TERM -- -$pid; return 1; }
  local seedpids=""
  for seed in $SEEDS; do
    { for cap in $CAPS; do
      tag=fn-$arm$([ $TASK = gpqa ] || echo -$TASK)-g${cap}k-s$seed
      echo "[$(date -Is)] $tag: start"
      (cd $EVAL && VIRTUAL_ENV= ~/.local/bin/uv run --frozen sharar-eval run --url http://127.0.0.1:$port --plain --model qwen38-flashnext \
        --label $tag --tasks $TASK --thinking --temperature 1.0 --seed $seed --max-tokens $((cap * 1024)) \
        --concurrency $CONC --engine "sglang sharar-port@$COMMIT gb200 $arm" --out $PORT_ROOT/runs/$tag) \
        > $PORT_ROOT/logs/eval_${tag}.log 2>&1
      echo "[$(date -Is)] $tag: rc=$? $(tail -3 $PORT_ROOT/logs/eval_${tag}.log | tr '\n' ' ' | cut -c1-300)"
    done; } &
    seedpids="$seedpids $!"
    [ $PAR_SEEDS = 1 ] || wait $!  # one seed at a time unless par
  done
  wait $seedpids  # the seed loops only, not the server
  grep -E "decode coverage" $log | head -2 | cut -c1-300 | sed "s/^/[$arm] /"
  kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
}

run_arm ${1:-w3} 0,1 30100 &
[ "${2:-exact}" = - ] || run_arm ${2:-exact} 2,3 30200 &  # "-": one arm only
wait
echo "[$(date -Is)] done"
