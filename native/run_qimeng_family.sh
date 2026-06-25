#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-probe}"
mkdir -p results/native_benchmarks/qimeng_tensorop

native/run_qimeng_gemm.sh "$MODE" || true
native/run_qimeng_attention.sh "$MODE" || true
native/run_qimeng_kernel.sh "$MODE" || true

python3 scripts/probe_repo.py --name qimeng_tensorop \
  --repo third_party/QiMeng-TensorOp \
  --output results/native_benchmarks/qimeng_tensorop/probe.json || true
python3 scripts/write_skip.py \
  --output results/native_benchmarks/qimeng_tensorop/skip.json \
  --name qimeng_tensorop \
  --reason 'The currently public QiMeng-TensorOp artifact targets the C920V2 RISC-V platform; no official A100 score is produced.'
