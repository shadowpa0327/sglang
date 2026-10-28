#!/usr/bin/env bash
# Smoke test of one Flash-Next arm on 2 GPUs: serve, smoke prompts, GSM8K 200; exits 0 only if GSM8K >= MIN_ACC.
# Usage: sbatch [--dependency ...] fn_smoke.sbatch <arm>   (later jobs: --dependency=afterok:<this job>)
#SBATCH -J fn-smoke
#SBATCH -p beta_test
#SBATCH --qos=test
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=72
#SBATCH -t 1:30:00
#SBATCH -o /mnt/home/cchang/sglang/sharar_port/logs/fn_smoke_%j.out
arm=${1:?arm}
$HOME/sglang/sharar_port/scripts/run_arm.sh $arm 0,1 30100 smoke_$SLURM_JOB_ID
acc=$(python3 -c "import json;r=json.load(open('$HOME/sglang/sharar_port/results/gsm8k_fn_${arm}_smoke_$SLURM_JOB_ID.json'));print(sum(x['ok'] for x in r)/len(r))" 2>/dev/null || echo 0)
echo "[$(date -Is)] $arm GSM8K $acc (min ${MIN_ACC:-0.85})"
python3 -c "import sys; sys.exit(0 if float('$acc') >= float('${MIN_ACC:-0.85}') else 1)"
