#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
MODE="${1:-smoke}"
REPO="$ROOT/third_party/CUDA-L2"
CUTLASS="$REPO/cutlass"
OUT="$ROOT/results/native_benchmarks/cuda_l2"
mkdir -p "$OUT"

skip() {
  python3 - "$OUT/skip.json" "$1" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"benchmark":"cuda_l2","status":"skipped","reason":sys.argv[2],"epoch":time.time()}, indent=2))
PY
  echo "[SKIP] CUDA-L2: $1"
}

[ -d "$REPO" ] || { skip "repository not cloned"; exit 0; }
[ -x "$REPO/eval_one_file.sh" ] || { skip "eval_one_file.sh not found or not executable"; exit 0; }

if [ ! -d "$CUTLASS/.git" ]; then
  echo "Cloning CUTLASS v4.2.1..."
  git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS" || {
    skip "CUTLASS v4.2.1 clone failed"; exit 0;
  }
fi

export CUTLASS_DIR="$CUTLASS"
export TORCH_CUDA_ARCH_LIST="8.0"

if [ "$MODE" = "probe" ]; then
  {
    echo "CUTLASS_DIR=$CUTLASS_DIR"
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    git -C "$REPO" rev-parse HEAD || true
    git -C "$CUTLASS" rev-parse HEAD || true
  } > "$OUT/probe.txt"
  echo "CUDA-L2 probe -> $OUT/probe.txt"
  exit 0
fi

CONFIGS="${CUDA_L2_MNK_LIST:-64_4096_64}"
if [ "$MODE" = "full" ] && [ -n "${CUDA_L2_FULL_CONFIG_FILE:-}" ] && [ -f "$CUDA_L2_FULL_CONFIG_FILE" ]; then
  CONFIGS="$(tr '\n' ' ' < "$CUDA_L2_FULL_CONFIG_FILE")"
fi

for mnk in $CONFIGS; do
  [ -n "$mnk" ] || continue
  mode_args="--mode ${CUDA_L2_MODE:-offline}"
  if [ "${CUDA_L2_MODE:-offline}" = "server" ]; then
    mode_args="$mode_args --target_qps ${CUDA_L2_TARGET_QPS:-100}"
  fi
  command="./eval_one_file.sh --mnk $mnk --acc_precise ${CUDA_L2_ACC_PRECISE:-fp32} --device_type a100 --warmup_seconds ${CUDA_L2_WARMUP_SECONDS:-5} --benchmark_seconds ${CUDA_L2_BENCHMARK_SECONDS:-10} --base_dir ./results_open_suite --gpu_device_id 0 $mode_args"
  python3 telemetry/run_command.py \
    --system cuda_l2 \
    --benchmark cuda_l2 \
    --task "${mnk}_${CUDA_L2_ACC_PRECISE:-fp32}_${CUDA_L2_MODE:-offline}" \
    --command "$command" \
    --cwd "$REPO" \
    --output-dir "$OUT/${mnk}_${CUDA_L2_ACC_PRECISE:-fp32}_${CUDA_L2_MODE:-offline}" \
    --proxy-port "${PROXY_PORT:-8100}" \
    --bench-gpus "${BENCH_GPU:-0}" \
    --timeout "${CUDA_L2_TIMEOUT:-3600}" || true
  python3 scripts/extract_numeric_metrics.py \
    --input "$OUT/${mnk}_${CUDA_L2_ACC_PRECISE:-fp32}_${CUDA_L2_MODE:-offline}/stdout.txt" \
    --output "$OUT/${mnk}_${CUDA_L2_ACC_PRECISE:-fp32}_${CUDA_L2_MODE:-offline}/extracted_metrics.json" || true
done
