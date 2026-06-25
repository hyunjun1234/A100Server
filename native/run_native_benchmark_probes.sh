#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-probe}"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
mkdir -p results/native_benchmarks logs/native_benchmarks

probe() {
  python3 - <<'PY'
import csv,json,subprocess
from pathlib import Path
benchmarks=[
 ("kernelbench","third_party/KernelBench","ready","integrated"),
 ("tritonbench","third_party/TritonBench","ready_with_inputs","native"),
 ("robust_kbench","third_party/robust-kbench","python311_required","native"),
 ("cudabench","third_party/CUDABench","ready_with_config","native"),
 ("flashinfer_bench","third_party/flashinfer-bench","conditional_trace","native"),
 ("kernelgym","third_party/KernelGYM","heavy","native_distributed"),
 ("kernelbench_x","third_party/KernelBenchX","ready_with_config","native"),
 ("cuda_l2","third_party/CUDA-L2","ready_A100","native_official_artifact"),
 ("qimeng_gemm","third_party/QiMeng-GEMM","ready_A100","native_official_artifact"),
 ("qimeng_tensorop","third_party/QiMeng-TensorOp","unsupported_A100_public_artifact","C920V2"),
 ("qimeng_attention","third_party/QiMeng-Attention","ready_with_command","native"),
 ("qimeng_kernel","third_party/QiMeng-Kernel","checkpoint_dataset_required","native"),
 ("agent_kernel_arena","third_party/AgentKernelArena","unsupported_A100","AMD_ROCm"),
]
rows=[]
for name,destination,status,mode in benchmarks:
 path=Path(destination); commit=''
 if (path/'.git').exists():
  try: commit=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
  except Exception: pass
 rows.append({'benchmark':name,'repo_exists':path.exists(),'status':status,'mode':mode,'commit':commit,'readme_exists':any(path.glob('README*')) if path.exists() else False,'python_files':len(list(path.rglob('*.py'))) if path.exists() else 0,'note':'explicit skip on A100' if status.startswith('unsupported') else 'run via dedicated/configurable native runner'})
out=Path('results/native_benchmarks/capability.csv')
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
Path('results/native_benchmarks/capability.json').write_text(json.dumps(rows,indent=2))
print(out)
for row in rows: print(row)
PY
  native/run_qimeng_family.sh probe || true
  native/run_cuda_l2.sh probe || true
}

run_one() {
  local benchmark="$1" repo="$2" command="$3"
  if [ -z "$command" ]; then echo "[SKIP] $benchmark: native command is empty"; return; fi
  if [ ! -d "$repo" ]; then echo "[SKIP] $benchmark: repo not found: $repo"; return; fi
  python3 telemetry/run_command.py --system benchmark_native --benchmark "$benchmark" --task native_full --command "$command" --cwd "$repo" --output-dir "results/native_benchmarks/$benchmark" --proxy-port "${PROXY_PORT:-8100}" --bench-gpus "${BENCH_GPU:-0}" --timeout "${NATIVE_TIMEOUT:-14400}" || true
}

full() {
  run_one tritonbench third_party/TritonBench "${TRITONBENCH_CMD:-}"
  run_one robust_kbench third_party/robust-kbench "${ROBUST_KBENCH_CMD:-}"
  run_one cudabench third_party/CUDABench "${CUDABENCH_CMD:-}"
  run_one flashinfer_bench third_party/flashinfer-bench "${FLASHINFER_BENCH_CMD:-}"
  run_one kernelgym third_party/KernelGYM "${KERNELGYM_CMD:-}"
  run_one kernelbench_x third_party/KernelBenchX "${KERNELBENCHX_CMD:-}"
  native/run_qimeng_family.sh full || true
  native/run_cuda_l2.sh smoke || true
  echo "[SKIP] agent_kernel_arena: official benchmark is AMD/ROCm-native"
}
case "$MODE" in probe) probe;; full) probe; full;; *) echo 'Usage: run_native_benchmark_probes.sh [probe|full]'; exit 1;; esac
