#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE="a100_open_kernel_results_${STAMP}"
ARCHIVE="${BUNDLE}.tar.gz"

rm -rf "$BUNDLE"
mkdir -p \
  "$BUNDLE/results" \
  "$BUNDLE/logs" \
  "$BUNDLE/systems" \
  "$BUNDLE/config" \
  "$BUNDLE/registry"

cp -a results/. "$BUNDLE/results/" 2>/dev/null || true

find logs -maxdepth 2 -type f \
  \( -name "*.out" -o -name "*.log" \) \
  -exec cp --parents {} "$BUNDLE/" \; 2>/dev/null || true

cp systems/*.env "$BUNDLE/systems/" 2>/dev/null || true
cp open_suite_config.env native_benchmarks.env benchmark_config.env \
  "$BUNDLE/config/" 2>/dev/null || true
cp registry/* "$BUNDLE/registry/" 2>/dev/null || true
cp METRICS.md "$BUNDLE/" 2>/dev/null || true

find runs -type f \
  \( -name "task_meta.json" \
     -o -name "summary.json" \
     -o -name "verdicts.json" \
     -o -name "verdict_*.json" \
     -o -name "events.jsonl" \
     -o -name "gpu.csv" \
     -o -name "generation_resource.txt" \
     -o -name "eval_resource.txt" \) \
  -exec cp --parents {} "$BUNDLE/" \; 2>/dev/null || true

tar -czf "$ARCHIVE" "$BUNDLE"
rm -rf "$BUNDLE"

echo "$ROOT/$ARCHIVE"
