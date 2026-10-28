#!/usr/bin/env bash
# sharar-eval on NVIDIA-Nemotron-3.5-Super: up to four arms a beta node (priority QoS), one GPU each (TP1), each with its
# own task and seeds. Settings as NVIDIA's NeMo Gym recipes (nemotron_recipes/super-3.5): thinking on, temperature 1.0,
# top-p 0.95, no top-k (--top-k 0), output bounded by the 262,144 context (cap 248K = 253,952 tokens).
# Arm spec: arm:task:seeds[:par], seeds comma-separated (par: every seed at once on the arm's server). Arms: exact (the
# NVFP4 checkpoint), w3 (sharar's 3-bit trunk and routed experts), exactmtp / w3mtp (+ the MTP drafter, exact
# verification). "-" leaves a GPU idle. Runs: $RUNS/nt-<arm>[-<task>]-g<cap>k-s<seed> (GPQA has no task suffix).
# Usage: sbatch nemotron_eval.sbatch <spec GPU 0> [<GPU 1> [<GPU 2> [<GPU 3>]]]   Env: CAP (k, 248), CONC (16)
#SBATCH -J nemo-eval
#SBATCH -p beta
#SBATCH --qos=priority
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH -t 1-00:00:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/nemotron_eval_%j.out
WT=/ddn/scratch/cornell/cchang/sglang-nemotron
PORT_ROOT=$HOME/sglang/sharar_port  # logs only; the code comes from the worktree
source $PORT_ROOT/scripts/env_gb200.sh
export PYTHONPATH_FORK=$WT
EVAL=/ddn/scratch/cornell/cchang/sharar-eval
export SHARAR_RESULTS=/ddn/scratch/cornell/cchang/sharar_results
RUNS=${RUNS:-/ddn/scratch/cornell/cchang/sharar_port/runs_nemotron}
CAP=${CAP:-248}; CONC=${CONC:-16}
export CTX=262144  # NVIDIA's max-model-len
COMMIT=$(git -C $WT rev-parse --short HEAD)$(git -C $WT diff --quiet || echo -dirty)
mkdir -p $RUNS $PORT_ROOT/logs

arm_env() {  # the serve_nemotron.sh environment of an arm
  case $1 in
    exact) echo PLAIN=1 ;; w3) echo W3=1 ;; exactmtp) echo SPEC=mtp ;; w3mtp) echo "SPEC=mtp W3=1" ;;
    # sharar's off-the-shelf `fast` (launch/settings.py nemotron_defaults): both 3-bit copies, MTP drafts with relaxed
    # acceptance 3:2.0, FP16-SR Mamba state, expert sharing keep 12 / 48 a request / 128 a call, at most 4 requests (its
    # MAX_BATCH; CONC defaults to 4 for it), depth 5 (sharar's at three or four requests; its adaptive depth is not
    # ported), the 2-bit draft head (SHARAR_DRAFT_HEAD=w2: the draft steps' lm_head is a 2-bit copy whose top 16 the FP8
    # rows rescore; the MTP layer itself stays NVFP4, as in sharar)
    fast) echo "W3=1 SPEC=mtp MTP_STEPS=5 SGLANG_DFLASH_RELAXED_TOP_K=3 SGLANG_DFLASH_RELAXED_MARGIN=2.0 SSM=float16 SSM_SR=1 MOE=12:48:128 SGLANG_SHARAR_DRAFT_HEAD=w2" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

run_arm() {  # run_arm <arm:task:seeds[:par]> <gpu> <port>
  local spec=$1 gpu=$2 port=$3 arm task s par seeds env pid seed tag seedpids=""
  IFS=: read -r arm task s par <<< "$spec"; seeds=${s//,/ }
  env=$(arm_env $arm) || return 1
  local CONC=$CONC; case $arm in fast*) CONC=${FAST_CONC:-4} ;; esac  # sharar fast serves at most 4 requests
  local maxrun=$([ "$par" = par ] && echo $((CONC * $(echo $seeds | wc -w))) || echo $CONC)
  local log=$PORT_ROOT/logs/nemotron-${arm}-${task}-$SLURM_JOB_ID.log
  echo "[$(date -Is)] $arm: task $task seeds $seeds par ${par:-no} gpu $gpu port $port maxrun $maxrun log $log"
  (export $env MAXRUN=$maxrun; exec setsid $WT/sharar_port/scripts/serve_nemotron.sh $gpu $port) > $log 2>&1 &
  pid=$!
  until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; do
    kill -0 $pid 2>/dev/null || { echo "[$arm] server died"; grep -v "Ignore import" $log | tail -30; return 1; }; sleep 15; done
  echo "[$(date -Is)] $arm: ready"
  for seed in $seeds; do
    tag=nt-$arm$([ $task = gpqa ] || echo -$task)-g${CAP}k-s$seed
    { echo "[$(date -Is)] $tag: start"
      (cd $EVAL && VIRTUAL_ENV= ~/.local/bin/uv run --frozen sharar-eval run --url http://127.0.0.1:$port --plain \
        --model nemotron35-super --label $tag --tasks $task --thinking --temperature 1.0 --top-k 0 --seed $seed \
        --max-tokens $((CAP * 1024)) --concurrency $CONC --engine "sglang sharar-nemotron@$COMMIT gb200 $arm" \
        --out $RUNS/$tag) > $PORT_ROOT/logs/eval_${tag}.log 2>&1
      echo "[$(date -Is)] $tag: rc=$? $(tail -2 $PORT_ROOT/logs/eval_${tag}.log | tr '\n' ' ' | cut -c1-250)"; } &
    seedpids="$seedpids $!"
    [ "$par" = par ] || wait $!  # one seed at a time unless par
  done
  wait $seedpids  # the seed loops only, not the server
  grep -E "sharar W3 check|decode coverage" $log | cut -c1-200 | sed "s/^/[$arm] /"
  kill -TERM -- -$pid 2>/dev/null; wait $pid 2>/dev/null
}

for g in 0 1 2 3; do
  spec=${@:$((g + 1)):1}
  [ "${spec:--}" = - ] || run_arm "$spec" $g $((30100 + 100 * g)) &
done
wait
echo "[$(date -Is)] done"
