#!/usr/bin/env bash
# Activate a selected Python/toolkit without any machine-specific absolute path.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ROUTERVC_PYTHON:-python}
env_root=$("$python_bin" -c 'import sys; print(sys.prefix)')
site_dir=$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PATH="$env_root/bin:$PATH"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export MAX_JOBS=${MAX_JOBS:-1} PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -z ${CUDA_HOME:-} && -x "$env_root/bin/nvcc" ]]; then export CUDA_HOME="$env_root"; fi
export LD_LIBRARY_PATH="$site_dir/nvidia/cu13/lib:$env_root/targets/x86_64-linux/lib:$env_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo_dir"
exec "$python_bin" "$@"
