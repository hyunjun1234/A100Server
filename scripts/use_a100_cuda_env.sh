#!/usr/bin/env bash

# Prefer pip-installed CUDA nvcc, then system CUDA 12.x.
NVCC_ROOT=$(python3 - <<'PY'
from pathlib import Path
try:
    import nvidia.cuda_nvcc
    path = Path(nvidia.cuda_nvcc.__file__).resolve()
    for candidate in [path.parent] + list(path.parents):
        if (candidate / "bin" / "nvcc").exists():
            print(candidate)
            raise SystemExit(0)
except Exception:
    pass
print("")
PY
)

if [ -n "$NVCC_ROOT" ] && [ -x "$NVCC_ROOT/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_ROOT"
elif [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d /usr/local/cuda-12.5 ]; then
  export CUDA_HOME=/usr/local/cuda-12.5
elif [ -d /usr/local/cuda-12 ]; then
  export CUDA_HOME=/usr/local/cuda-12
elif [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
