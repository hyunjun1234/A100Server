#!/usr/bin/env bash

# Prefer pip-installed CUDA 12.x nvcc if available.
NVCC_ROOT=$(python - <<'PY'
from pathlib import Path
try:
    import nvidia.cuda_nvcc
    p = Path(nvidia.cuda_nvcc.__file__).resolve()
    candidates = [p.parent] + list(p.parents)
    for c in candidates:
        if (c / "bin" / "nvcc").exists():
            print(c)
            raise SystemExit(0)
except Exception:
    pass
print("")
PY
)

if [ -n "$NVCC_ROOT" ] && [ -x "$NVCC_ROOT/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_ROOT"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
else
  # fallback: system CUDA 12.x
  if [ -d /usr/local/cuda-12.6 ]; then
    export CUDA_HOME=/usr/local/cuda-12.6
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  elif [ -d /usr/local/cuda-12.5 ]; then
    export CUDA_HOME=/usr/local/cuda-12.5
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  elif [ -d /usr/local/cuda-12 ]; then
    export CUDA_HOME=/usr/local/cuda-12
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_NAME="NVIDIA A100 80GB PCIe"
