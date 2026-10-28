#!/usr/bin/env bash
# finish_after.sh <jobid> <label>: once the job's loop is running <label>, pause the loop (the running eval finishes;
# the next one never starts), stop that arm's server when the eval exits, and cancel the job when no eval is left in it.
job=$1; label=$2
out=$HOME/sglang/sharar_port/logs/fn_gpqa_$job.out
on() { srun --jobid=$job --overlap "$@"; }
until grep -q "$label: start" $out; do sleep 30; done
sleep 20
loop=$(on bash -c "ps -o ppid= -p \$(pgrep -u \$USER -f 'uv run --frozen sharar-eval run .*--label $label( |\$)' | head -1)" | tr -d ' ')
[ -n "$loop" ] || { echo "no eval for $label"; exit 1; }
on kill -STOP $loop
echo "[$(date -Is)] paused loop $loop during $label"
until ! on pgrep -u $USER -f "sharar-eval run .*--label $label( |\$)" >/dev/null; do sleep 60; done
on bash -c "srv=\$(ps -o pid=,cmd= --ppid $loop | awk '/launch_server/ {print \$1}'); [ -n \"\$srv\" ] && kill -TERM -- -\$srv"
echo "[$(date -Is)] $label done; server stopped"
until ! on pgrep -u $USER -f "sharar-eval run" >/dev/null; do sleep 60; done
echo "[$(date -Is)] no evals left in $job: cancelling"; scancel $job
