#!/usr/bin/env bash
# a100_no_uv_followup_fix.sh
#
# 사용법:
#   cd /home/jun/unified_bench
#   bash a100_no_uv_followup_fix.sh patch
#   bash a100_no_uv_followup_fix.sh rerun_autokernel
#   bash a100_no_uv_followup_fix.sh summary
#   bash a100_no_uv_followup_fix.sh pack
#
# 목적:
#   a100_autokernel_no_uv_fix.sh 이후 남은 문제 해결:
#   1. AutoKernel candidate가 `from kernels.cuda._compile import compile_cuda`를 쓰는데
#      final_eval에서 `kernels` 모듈을 못 찾아서 죽는 문제 해결
#      -> systems/autokernel.env에 AutoKernel repo를 PYTHONPATH로 export
#   2. speedup(gm)처럼 보이는 값이 사실상 산술 평균/이상치 영향을 받아
#      CUDA-L1 level2 688x처럼 과장되어 보이는 문제 확인용 corrected summary 생성
#
# 주의:
#   이 스크립트는 K-Search 생성 코드 품질 문제까지 자동 해결하지 않는다.
#   K-Search는 현재 missing external .cu/.cpp, 잘못된 load_inline, example usage 포함이 주된 문제다.

set -euo pipefail

ROOT="${ROOT:-/home/jun/unified_bench}"
MODE="${1:-patch}"

cd "$ROOT"

log() {
  echo
  echo "=========================================="
  echo "$*"
  echo "=========================================="
}

find_autokernel_repo() {
  if [ -d "$ROOT/third_party/autokernel" ]; then
    echo "$ROOT/third_party/autokernel"
  elif [ -d "$ROOT/third_party/AutoKernel" ]; then
    echo "$ROOT/third_party/AutoKernel"
  else
    echo "$ROOT/third_party/autokernel"
  fi
}

patch_autokernel_env() {
  log "Patching AutoKernel env with PYTHONPATH for kernels.cuda._compile"

  AK_REPO="$(find_autokernel_repo)"

  if [ ! -f drivers/autokernel_kb_program_adapter.py ]; then
    echo "ERROR: drivers/autokernel_kb_program_adapter.py not found."
    echo "Run: bash a100_autokernel_no_uv_fix.sh patch"
    exit 1
  fi

  cat > systems/autokernel.env <<EOF
# AutoKernel KernelBench program adapter.
# Uses official AutoKernel kernelbench/program_kb.md instructions with current local OpenAI-compatible model server.
# Avoids uv sync and does NOT call nonexistent autokernel.py.
#
# Important:
# Some AutoKernel-style candidates import:
#   from kernels.cuda._compile import compile_cuda
# Therefore the AutoKernel repo must be on PYTHONPATH during both generation and final_eval.
export AUTOKERNEL_REPO="$AK_REPO"
export PYTHONPATH="\$AUTOKERNEL_REPO:\${PYTHONPATH:-}"
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/autokernel_kb_program_adapter.py --task {TASK} --out {CAND_DIR} --rounds {ROUNDS} --seed {SEED} --temperature {TEMP}'
CANDIDATE_GLOB='candidate_*.py'
EOF

  echo "===== systems/autokernel.env ====="
  cat systems/autokernel.env

  echo
  echo "Checking AutoKernel kernels helper:"
  if [ -e "$AK_REPO/kernels/cuda/_compile.py" ]; then
    echo "OK: $AK_REPO/kernels/cuda/_compile.py"
  else
    echo "WARNING: $AK_REPO/kernels/cuda/_compile.py not found."
    echo "The PYTHONPATH patch may not fix candidates that import kernels.cuda._compile."
    echo "Run:"
    echo "  find third_party -path '*kernels/cuda/_compile.py' -print"
  fi
}

rerun_autokernel_subset50() {
  patch_autokernel_env

  log "Rerunning AutoKernel subset50 only"

  if [ -f use_a100_cuda126_env.sh ]; then
    # shellcheck disable=SC1091
    source use_a100_cuda126_env.sh
  fi

  rm -rf runs/autokernel_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2

  mkdir -p logs
  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_subset50.txt" \
  SYSTEMS="autokernel" \
  nohup ./run_all_with_server.sh 1 1 0.2 900 \
    > logs/a100_autokernel_subset50_pythonpath_fix.out 2>&1 &

  echo "Started AutoKernel subset50 rerun."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_autokernel_subset50_pythonpath_fix.out"
}

write_corrected_summary_tool() {
  log "Writing tools/collect_corrected_summary.py"
  mkdir -p tools

  cat > tools/collect_corrected_summary.py <<'PY'
#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict

src = Path("results/unified_per_task.csv")
if not src.exists():
    raise SystemExit("results/unified_per_task.csv not found. Run collect_results.py first.")

rows = list(csv.DictReader(src.open()))
groups = defaultdict(list)

for r in rows:
    groups[(r["system"], r["level"])].append(r)

out_rows = []
for (system, level), rs in sorted(groups.items()):
    n = len(rs)
    runnable_vals = [float(r.get("runnable_rate") or 0.0) for r in rs]
    correct_vals = [float(r.get("correct_rate") or 0.0) for r in rs]
    pass_vals = [float(r.get("pass@1") or 0.0) for r in rs]
    fast_vals = [float(r.get("fast_1") or 0.0) for r in rs]

    # Only correct tasks with positive best_score are valid for speedup statistics.
    speeds = []
    for r in rs:
        c = float(r.get("correct_rate") or 0.0)
        s = float(r.get("best_score") or 0.0)
        if c > 0 and s > 0:
            speeds.append(s)

    if speeds:
        log_sum = sum(math.log(max(s, 1e-12)) for s in speeds)
        gmean = math.exp(log_sum / len(speeds))
        mean = sum(speeds) / len(speeds)
        med = sorted(speeds)[len(speeds)//2] if len(speeds) % 2 else (
            sorted(speeds)[len(speeds)//2 - 1] + sorted(speeds)[len(speeds)//2]
        ) / 2
        mx = max(speeds)
    else:
        gmean = mean = med = mx = 0.0

    out_rows.append({
        "system": system,
        "level": level,
        "n_tasks": n,
        "runnable": round(sum(runnable_vals)/n, 3),
        "correct": round(sum(correct_vals)/n, 3),
        "pass@1": round(sum(pass_vals)/n, 3),
        "fast_1": round(sum(fast_vals)/n, 3),
        "n_correct_speed": len(speeds),
        "speedup_gmean_correct": round(gmean, 3),
        "speedup_mean_correct": round(mean, 3),
        "speedup_median_correct": round(med, 3),
        "speedup_max_correct": round(mx, 3),
    })

dst = Path("results/unified_summary_corrected.csv")
with dst.open("w", newline="") as f:
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

print("Wrote", dst)
for r in out_rows:
    print(r)
PY

  chmod +x tools/collect_corrected_summary.py
}

make_corrected_summary() {
  write_corrected_summary_tool

  log "Running collect_results.py and corrected summary"
  python collect_results.py 2>/dev/null || true
  python tools/collect_corrected_summary.py
}

pack_debug() {
  log "Packing debug zip"

  rm -rf debug_upload_a100_followup_fix
  mkdir -p debug_upload_a100_followup_fix

  cp logs/a100_autokernel_subset50_pythonpath_fix.out debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp logs/a100_official6_subset50_no_uv_autokernel.out debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp results/unified_summary.csv debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp results/unified_summary_corrected.csv debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp results/unified_per_task.csv debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp results/unified_table.tex debug_upload_a100_followup_fix/ 2>/dev/null || true
  cp results/evaluation_sheet.json debug_upload_a100_followup_fix/ 2>/dev/null || true

  mkdir -p debug_upload_a100_followup_fix/systems
  cp systems/*.env debug_upload_a100_followup_fix/systems/ 2>/dev/null || true

  mkdir -p debug_upload_a100_followup_fix/tools
  cp tools/collect_corrected_summary.py debug_upload_a100_followup_fix/tools/ 2>/dev/null || true

  find runs logs -type f | grep -E "autokernel|ksearch|cuda_l1|cudaforge|autotriton|drkernel|summary.json|verdict|candidate_.*py|round000_kernel.py|raw_reply|server" \
    | while read -r f; do
        mkdir -p "debug_upload_a100_followup_fix/$(dirname "$f")"
        cp "$f" "debug_upload_a100_followup_fix/$f" 2>/dev/null || true
      done

  tar -czf a100_followup_fix_debug.tar.gz debug_upload_a100_followup_fix

  echo
  echo "Upload this file:"
  echo "$ROOT/a100_followup_fix_debug.tar.gz"
}

case "$MODE" in
  patch)
    patch_autokernel_env
    write_corrected_summary_tool
    ;;
  rerun_autokernel)
    rerun_autokernel_subset50
    ;;
  summary)
    make_corrected_summary
    ;;
  pack)
    pack_debug
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage:"
    echo "  bash a100_no_uv_followup_fix.sh patch"
    echo "  bash a100_no_uv_followup_fix.sh rerun_autokernel"
    echo "  bash a100_no_uv_followup_fix.sh summary"
    echo "  bash a100_no_uv_followup_fix.sh pack"
    exit 1
    ;;
esac
