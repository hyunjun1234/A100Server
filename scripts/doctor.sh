#!/usr/bin/env bash
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -f open_suite_config.env ] && source open_suite_config.env

mkdir -p results/telemetry

python3 - <<'PY'
import json
import platform
import shutil
import subprocess
from pathlib import Path

def command(args):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": -1, "error": repr(exc)}

report = {
    "python": platform.python_version(),
    "executables": {
        name: shutil.which(name)
        for name in [
            "python3",
            "git",
            "nvcc",
            "ncu",
            "nvidia-smi",
            "cmake",
            "ninja",
        ]
    },
    "nvidia_smi": command([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ]),
    "nvcc": command(["nvcc", "--version"]),
    "ncu": command(["ncu", "--version"]),
    "disk": command(["df", "-h", "."]),
}

try:
    import torch
    report["torch"] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
except Exception as exc:
    report["torch"] = {"error": repr(exc)}

Path("results/telemetry/doctor.json").write_text(
    json.dumps(report, indent=2)
)
print(json.dumps(report, indent=2))
PY

if command -v ncu >/dev/null 2>&1; then
  echo
  echo "Checking NCU performance-counter access..."
  ncu --query-metrics >/dev/null 2>results/telemetry/ncu_permission.err || true

  if grep -q "ERR_NVGPUCTRPERM" results/telemetry/ncu_permission.err; then
    echo "WARNING: NCU performance counters are not permitted."
  else
    echo "NCU permission check did not report ERR_NVGPUCTRPERM."
  fi
fi
