# Environment for the sharar port on this GB200 cluster: the scratch venv, pip CUDA 13 toolkit for the W3 JIT, scratch HF cache.
# Usage: source env_gb200.sh (serve.sh: ENV_SH=.../env_gb200.sh)
_S=/ddn/scratch/cornell/cchang
source $_S/envs/sglang-sharar/bin/activate
export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=$_S/hf_home HF_HUB=$_S/hf_home/hub HF_HUB_OFFLINE=1
export TORCH_EXTENSIONS_DIR=$_S/torch_extensions TRITON_CACHE_DIR=$_S/triton_cache
export PYTHONPATH_FORK=$HOME/sglang
export XDG_CACHE_HOME=$_S/cache FLASHINFER_WORKSPACE_BASE=$_S/cache UV_CACHE_DIR=$_S/uv_cache
