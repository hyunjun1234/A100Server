#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
MODE="${1:-probe}"
REPO="$ROOT/third_party/QiMeng-Attention"
OUT="$ROOT/results/native_benchmarks/qimeng_attention"
mkdir -p "$OUT"

python3 scripts/probe_repo.py \
  --name qimeng_attention \
  --repo "$REPO" \
  --output "$OUT/probe.json" || true

if [ "$MODE" = "probe" ]; then exit 0; fi
if [ -z "${QIMENG_ATTENTION_CMD:-}" ]; then
  python3 scripts/write_skip.py --output "$OUT/skip.json" --name qimeng_attention \
    --reason 'QIMENG_ATTENTION_CMD is empty; inspect probe.json and set the exact official command for this commit.'
  exit 0
fi
python3 telemetry/run_command.py \
  --system qimeng_attention \
  --benchmark qimeng_attention \
  --task official_native \
  --command "$QIMENG_ATTENTION_CMD" \
  --cwd "$REPO" \
  --output-dir "$OUT/full" \
  --proxy-port "${PROXY_PORT:-8100}" \
  --bench-gpus "${BENCH_GPU:-0}" \
  --timeout "${QIMENG_ATTENTION_TIMEOUT:-7200}" || true
