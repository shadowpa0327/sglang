#!/usr/bin/env bash
# stop_after.sh <jobid> <label>...: wait until each label's run starts in the job's fn_gpqa output, stop that arm
# (stop_arm.sh); then, once no arm or fill run in the job is left working, cancel the job.
job=$1; shift
S=$HOME/sglang/sharar_port/scripts; out=$HOME/sglang/sharar_port/logs/fn_gpqa_$job.out
for label in "$@"; do
  until grep -q "$label: start" $out; do sleep 30; done
  sleep 20  # the eval process starts right after the line
  srun --jobid=$job --overlap $S/stop_arm.sh $label
done
until ! srun --jobid=$job --overlap pgrep -u $USER -f "sharar-eval run" >/dev/null; do sleep 60; done
echo "[$(date -Is)] no runs left in $job: cancelling"; scancel $job
