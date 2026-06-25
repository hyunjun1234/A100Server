#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="a100_open_kernel_feedback_${STAMP}"
WORK="$ROOT/$NAME"
ARCHIVE="$ROOT/${NAME}.tar.gz"
rm -rf "$WORK"
mkdir -p "$WORK"/{environment,configs,repo_state,results,error_analysis,log_tails,failed_artifacts}

# Refresh summaries when possible.
python3 telemetry/aggregate.py --root "$ROOT" --task-list "${TASK_LIST:-kernelbench_subset50.txt}" > "$WORK/results/aggregate_stdout.txt" 2>&1 || true
native/run_method_probes.sh > "$WORK/results/method_probe_stdout.txt" 2>&1 || true
native/run_native_benchmark_probes.sh probe > "$WORK/results/native_probe_stdout.txt" 2>&1 || true
python3 scripts/summarize_errors.py --root "$ROOT" --output-dir "$WORK/error_analysis" || true

# Environment, no secret-bearing full env dump.
{
  date -Is
  uname -a
  python3 --version
  which python3 || true
  which nvcc || true
  nvcc --version || true
  which ncu || true
  ncu --version || true
  df -h .
  free -h || true
  ulimit -a || true
} > "$WORK/environment/system.txt" 2>&1
nvidia-smi -q > "$WORK/environment/nvidia_smi_q.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,uuid,driver_version,pci.bus_id,memory.total,power.limit,compute_cap --format=csv > "$WORK/environment/gpus.csv" 2>&1 || true
python3 -m pip freeze > "$WORK/environment/pip_freeze.txt" 2>&1 || true
ps auxww > "$WORK/environment/processes.txt" 2>&1 || true

# Config and code metadata.
cp -a open_suite_config.env native_benchmarks.env benchmark_config.env METRICS.md SOURCES_REVIEW.md "$WORK/configs/" 2>/dev/null || true
cp -a systems "$WORK/configs/" 2>/dev/null || true
cp -a registry "$WORK/configs/" 2>/dev/null || true
cp -a results/. "$WORK/results/" 2>/dev/null || true

python3 - "$ROOT" "$WORK/repo_state/repos.json" <<'PY'
import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); out=Path(sys.argv[2]); rows=[]
for repo in [root] + sorted((root/'third_party').glob('*')):
    if not (repo/'.git').exists(): continue
    def cmd(*args):
        try:return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=30).stdout.strip()
        except Exception:return ''
    rows.append({'path':str(repo.relative_to(root)) if repo!=root else '.', 'remote':cmd('remote','get-url','origin'), 'commit':cmd('rev-parse','HEAD'), 'branch':cmd('rev-parse','--abbrev-ref','HEAD'), 'status':cmd('status','--porcelain')[:10000], 'diff_stat':cmd('diff','--stat')[:10000]})
out.write_text(json.dumps(rows,indent=2))
PY

# Compact log tails, preserving relative path. Full giant logs are intentionally excluded.
find logs runs results -type f \( -name '*.out' -o -name '*.log' -o -name '*.txt' \) -print0 2>/dev/null | \
while IFS= read -r -d '' file; do
  rel="${file#./}"
  dest="$WORK/log_tails/${rel}.tail.txt"
  mkdir -p "$(dirname "$dest")"
  tail -n 500 "$file" > "$dest" 2>/dev/null || true
done

# High-value task files. Cap copied source/raw reply files at 512 KiB each.
find runs -type f \( -name 'task_meta.json' -o -name 'summary.json' -o -name 'verdicts.json' -o -name 'verdict_*.json' -o -name 'events.jsonl' -o -name 'generation_resource.txt' -o -name 'eval_resource.txt' -o -name 'gpu.csv' -o -name 'candidate_*.py' -o -name 'round*_kernel.py' -o -name '*raw_reply*.txt' \) -print0 2>/dev/null | \
while IFS= read -r -d '' file; do
  size=$(stat -c %s "$file" 2>/dev/null || echo 0)
  [ "$size" -le 524288 ] || continue
  dest="$WORK/failed_artifacts/$file"
  mkdir -p "$(dirname "$dest")"
  cp "$file" "$dest" 2>/dev/null || true
done

cat > "$WORK/README_SEND_TO_ASSISTANT.txt" <<EOF
Upload the single archive:
  $ARCHIVE

It contains:
- environment and GPU/CUDA/NCU information
- exact repository commits and dirty status
- configs and system env files
- result CSV/JSON files
- categorized errors and clusters
- log tails
- candidate/raw reply/verdict/task telemetry files (size-capped)

Model weights, caches and large build directories are excluded.
EOF

tar -czf "$ARCHIVE" "$NAME"
rm -rf "$WORK"
echo "$ARCHIVE"
