#!/usr/bin/env bash
# stop_arm.sh <label>: inside a running fn_gpqa job (srun --jobid=<id> --overlap), stop the arm whose eval has this label:
# pause its loop (so the job stays alive and no further runs start), stop the eval and the arm's server.
label=$1
e=$(pgrep -u $USER -f "uv run --frozen sharar-eval run .*--label $label( |$)" | head -1)
[ -n "$e" ] || { echo "no eval with label $label"; exit 1; }
loop=$(ps -o ppid= -p $e | tr -d ' ')
kill -STOP $loop
srv=$(ps -o pid=,cmd= --ppid $loop | awk '/launch_server/ {print $1}')
pkill -TERM -P $e; kill -TERM $e
[ -n "$srv" ] && kill -TERM -- -$srv
echo "[$(date -Is)] stopped $label: loop $loop paused, eval $e and server group ${srv:-none} terminated"
