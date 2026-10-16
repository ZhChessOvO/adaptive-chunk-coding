#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ -n ${TMUX:-} ]] || { echo 'Run compilation inside tmux.' >&2; exit 2; }
[[ -d "$repo_dir/third_party/cutlass/include" ]] || { echo 'Restore CUTLASS source from the migration archive first.' >&2; exit 2; }
export MAX_JOBS=${MAX_JOBS:-1}
cd "$repo_dir/src/cpp"
python -m pip install --no-build-isolation --no-deps .
cd "$repo_dir/src/layers/extensions/inference"
python -m pip install --no-build-isolation --no-deps .
cd "$repo_dir"
python -c 'import torch, MLCodec_extensions_cpp, inference_extensions_cuda; print(torch.__version__, torch.cuda.get_device_name(0))'
