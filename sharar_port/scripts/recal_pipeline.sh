#!/usr/bin/env bash
# Re-fit Flash-Next's 3-bit trunk and experts on thinking traces: capture (GPUs 6,7) -> trunk GPTQ -> experts GPTQ.
set -e
cd /data/ccchang/sharar_port
C=recal/capture_think; T=recal/w3_think_trunk; O=recal/w3_think
mkdir -p $C
echo "[$(date -Is)] capture server"
setsid nohup scripts/capture_serve.sh 6,7 31700 /data/ccchang/sharar_port/$C > logs/capture_think.log 2>&1 < /dev/null &
until [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:31700/health)" = 200 ]; do
  pgrep -f "[s]glang.launch_server.*--port 31700 " >/dev/null || { echo "capture server died"; exit 1; }; sleep 15; done
source /data/ccchang/port_scripts/env.sh
python -I scripts/capture_client.py http://127.0.0.1:31700 recal/calib_think_1m2.json $C || true
until [ -f $C/SAVED ]; do sleep 10; done
for p in $(pgrep -f "[s]glang.launch_server.*--port 31700 "); do kill -TERM -- -$(ps -o pgid= -p $p | tr -d ' ') 2>/dev/null; done
echo "[$(date -Is)] captured: $(ls $C/cov_*.pt | wc -l) covariances, $(python3 -c "import json;print(json.load(open('$C/capture.json'))['rows'])") MoE rows"
sleep 20
export SHARAR_CUDA_ARCH=9.0a SHARAR_HF_CACHE=/data/ccchang/hf_home PYTHONPATH=/home/ccchang/sharar SHARAR_TARGET=flashnext
CUDA_VISIBLE_DEVICES=6 python scripts/fit_trunk.py $C $T recal/calib_think_1m2.json
echo "[$(date -Is)] trunk fitted"
CUDA_VISIBLE_DEVICES=6 python -m sharar.weights.flashnext_w3_experts fit --method gptq --activations $C --source $T --output $O
echo "[$(date -Is)] RECAL_DONE $O"
