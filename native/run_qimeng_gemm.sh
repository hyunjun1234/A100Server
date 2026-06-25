#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
MODE="${1:-smoke}"
REPO="$ROOT/third_party/QiMeng-GEMM"
OUT="$ROOT/results/native_benchmarks/qimeng_gemm"
mkdir -p "$OUT"

skip() {
  python3 - "$OUT/skip.json" "$1" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"benchmark":"qimeng_gemm","status":"skipped","reason":sys.argv[2],"epoch":time.time()}, indent=2))
PY
  echo "[SKIP] QiMeng-GEMM: $1"
}

[ -d "$REPO/code/CUDA" ] || { skip "official CUDA directory code/CUDA not found"; exit 0; }

if [ "$MODE" = "probe" ]; then
  find "$REPO" -maxdepth 3 -type f | sort > "$OUT/file_tree.txt"
  [ -f "$REPO/README.md" ] && cp "$REPO/README.md" "$OUT/README.md"
  echo "QiMeng-GEMM probe -> $OUT"
  exit 0
fi

python3 telemetry/run_command.py \
  --system qimeng_gemm \
  --benchmark qimeng_gemm \
  --task build \
  --command 'make clean >/dev/null 2>&1 || true; make' \
  --cwd "$REPO/code/CUDA" \
  --output-dir "$OUT/build" \
  --proxy-port "${PROXY_PORT:-8100}" \
  --bench-gpus "${BENCH_GPU:-0}" \
  --timeout "${QIMENG_GEMM_TIMEOUT:-1800}" || true

IFS=';' read -ra SHAPES <<< "${QIMENG_GEMM_SHAPES:-1024,1024,1024}"
for shape in "${SHAPES[@]}"; do
  IFS=',' read -r M N K <<< "$shape"
  [ -n "$M" ] && [ -n "$N" ] && [ -n "$K" ] || continue
  label="${M}_${N}_${K}"
  python3 telemetry/run_command.py \
    --system qimeng_gemm \
    --benchmark qimeng_gemm \
    --task "$label" \
    --command "./test $M $N $K" \
    --cwd "$REPO/code/CUDA" \
    --output-dir "$OUT/$label" \
    --proxy-port "${PROXY_PORT:-8100}" \
    --bench-gpus "${BENCH_GPU:-0}" \
    --timeout "${QIMENG_GEMM_TIMEOUT:-1800}" || true
  python3 scripts/extract_numeric_metrics.py \
    --input "$OUT/$label/stdout.txt" \
    --output "$OUT/$label/extracted_metrics.json" || true
done
