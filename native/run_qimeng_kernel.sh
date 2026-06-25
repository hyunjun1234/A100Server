#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
MODE="${1:-probe}"
REPO="$ROOT/third_party/QiMeng-Kernel"
OUT="$ROOT/results/native_benchmarks/qimeng_kernel"
mkdir -p "$OUT"
python3 scripts/probe_repo.py --name qimeng_kernel --repo "$REPO" --output "$OUT/probe.json" || true
if [ "$MODE" = "probe" ]; then exit 0; fi

if [ ! -x "$REPO/scripts/inference.sh" ] || [ ! -x "$REPO/scripts/eval_generations.sh" ]; then
  python3 scripts/write_skip.py --output "$OUT/skip.json" --name qimeng_kernel \
    --reason 'The current public checkout does not contain executable scripts/inference.sh and scripts/eval_generations.sh described by the README.'
  exit 0
fi
if [ -z "${QIMENG_KERNEL_CHECKPOINT:-}" ] || [ -z "${QIMENG_KERNEL_DATASET:-}" ]; then
  python3 scripts/write_skip.py --output "$OUT/skip.json" --name qimeng_kernel \
    --reason 'QIMENG_KERNEL_CHECKPOINT or QIMENG_KERNEL_DATASET is empty.'
  exit 0
fi

infer="bash scripts/inference.sh '${QIMENG_KERNEL_CHECKPOINT}' '${QIMENG_KERNEL_API_BASE}' '${QIMENG_KERNEL_API_KEY}' '${QIMENG_KERNEL_API_MODEL}' '${QIMENG_KERNEL_DATASET}'"
python3 telemetry/run_command.py \
  --system qimeng_kernel --benchmark qimeng_kernel --task inference \
  --command "$infer" --cwd "$REPO" --output-dir "$OUT/inference" \
  --proxy-port "${PROXY_PORT:-8100}" --bench-gpus "${BENCH_GPU:-0}" \
  --timeout "${NATIVE_TIMEOUT:-14400}" || true

eval_cmd="bash scripts/eval_generations.sh '${QIMENG_KERNEL_BENCHMARK}' '${QIMENG_KERNEL_RUN_DIR}' '${QIMENG_KERNEL_SUBSET}'"
python3 telemetry/run_command.py \
  --system qimeng_kernel --benchmark qimeng_kernel --task evaluation \
  --command "$eval_cmd" --cwd "$REPO" --output-dir "$OUT/evaluation" \
  --proxy-port "${PROXY_PORT:-8100}" --bench-gpus "${BENCH_GPU:-0}" \
  --timeout "${NATIVE_TIMEOUT:-14400}" || true
