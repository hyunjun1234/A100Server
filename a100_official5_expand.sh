#!/usr/bin/env bash
# a100_official5_expand.sh
#
# A100 2장 서버용 official5 확장 실험 스크립트
#
# 대상 시스템:
#   cudaforge autokernel cuda_l1 autotriton drkernel
#
# 제외:
#   ksearch cuda_agent geak
#
# 사용법:
#   cd /home/jun/unified_bench
#   chmod +x a100_official5_expand.sh
#
#   bash a100_official5_expand.sh patch
#   bash a100_official5_expand.sh tiny5
#   bash a100_official5_expand.sh round1
#   bash a100_official5_expand.sh round3
#   bash a100_official5_expand.sh status
#   bash a100_official5_expand.sh summary
#   bash a100_official5_expand.sh pack
#
# 업로드할 최종 파일:
#   /home/jun/unified_bench/a100_official5_subset50_expand_debug.tar.gz

set -euo pipefail

ROOT="${ROOT:-/home/jun/unified_bench}"
MODE="${1:-help}"

OFFICIAL5_SYSTEMS="cudaforge autokernel cuda_l1 autotriton drkernel"

cd "$ROOT"

log() {
  echo
  echo "=========================================="
  echo "$*"
  echo "=========================================="
}

ensure_dirs() {
  mkdir -p models logs runs results systems drivers tools
}

source_cuda_env() {
  if [ -f "$ROOT/use_a100_cuda126_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/use_a100_cuda126_env.sh"
  else
    echo "WARNING: use_a100_cuda126_env.sh not found. Continuing without it."
  fi
}

write_cuda_env_if_missing() {
  if [ -f use_a100_cuda126_env.sh ]; then
    return 0
  fi

  log "Writing use_a100_cuda126_env.sh"

  cat > use_a100_cuda126_env.sh <<'EOF'
#!/usr/bin/env bash

NVCC_ROOT=$(python - <<'PY'
from pathlib import Path
try:
    import nvidia.cuda_nvcc
    p = Path(nvidia.cuda_nvcc.__file__).resolve()
    for c in [p.parent] + list(p.parents):
        if (c / "bin" / "nvcc").exists():
            print(c)
            raise SystemExit(0)
except Exception:
    pass
print("")
PY
)

if [ -n "$NVCC_ROOT" ] && [ -x "$NVCC_ROOT/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_ROOT"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
elif [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d /usr/local/cuda-12.5 ]; then
  export CUDA_HOME=/usr/local/cuda-12.5
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d /usr/local/cuda-12 ]; then
  export CUDA_HOME=/usr/local/cuda-12
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_NAME="NVIDIA A100 80GB PCIe"
EOF

  chmod +x use_a100_cuda126_env.sh
}

install_min_deps() {
  log "Checking minimal dependencies"

  python - <<'PY'
missing = []
for m in ["requests", "openai", "transformers", "torch", "triton", "datasets"]:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
print("missing:", missing)
PY

  python -m pip install -U datasets >/dev/null
}

patch_autokernel_env_check() {
  log "Checking AutoKernel no-uv adapter"

  if [ ! -f drivers/autokernel_kb_program_adapter.py ]; then
    echo "ERROR: drivers/autokernel_kb_program_adapter.py not found."
    echo "먼저 a100_autokernel_no_uv_fix.sh patch 를 실행하거나, 내가 준 no-uv fix sh를 적용해야 합니다."
    exit 1
  fi

  AK_REPO="$ROOT/third_party/autokernel"
  if [ ! -d "$AK_REPO" ] && [ -d "$ROOT/third_party/AutoKernel" ]; then
    AK_REPO="$ROOT/third_party/AutoKernel"
  fi

  cat > systems/autokernel.env <<EOF
# AutoKernel KernelBench program adapter.
# Uses official AutoKernel kernelbench/program_kb.md instructions with current local OpenAI-compatible model server.
# Avoids uv sync and does NOT call nonexistent autokernel.py.
export AUTOKERNEL_REPO="$AK_REPO"
export PYTHONPATH="\$AUTOKERNEL_REPO:\${PYTHONPATH:-}"
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/autokernel_kb_program_adapter.py --task {TASK} --out {CAND_DIR} --rounds {ROUNDS} --seed {SEED} --temperature {TEMP}'
CANDIDATE_GLOB='candidate_*.py'
EOF

  cat systems/autokernel.env
}

patch_cuda_l1_env() {
  log "Patching CUDA-L1 env for A100 official artifact"

  cat > systems/cuda_l1.env <<'EOF'
# CUDA-L1 official released artifact adapter for A100.
# Re-evaluates official A100 JSON artifact on local A100.
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/cuda_l1_artifact_adapter.py --task {TASK} --out {CAND_DIR} --gpu-json a100.json'
CANDIDATE_GLOB='candidate_*.py'
EOF

  cat systems/cuda_l1.env
}

patch_official_checkpoints() {
  log "Patching AutoTriton / DrKernel official checkpoint IDs"

  if [ -f systems/autotriton.env ]; then
    if grep -q '^SYSTEM_MODEL_HF_ID=' systems/autotriton.env; then
      sed -i 's|^SYSTEM_MODEL_HF_ID=.*|SYSTEM_MODEL_HF_ID="ai9stars/AutoTriton"|' systems/autotriton.env
    else
      echo 'SYSTEM_MODEL_HF_ID="ai9stars/AutoTriton"' >> systems/autotriton.env
    fi
  else
    cat > systems/autotriton.env <<'EOF'
SYSTEM_MODEL_ALIAS='autotriton8b'
SYSTEM_MODEL_HF_ID="ai9stars/AutoTriton"
SYSTEM_MODEL_LOCAL_DIR='models/autotriton8b'
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/trained_model.py --task {TASK} --cand_dir {CAND_DIR} --rounds {ROUNDS} --seed {SEED} --temperature {TEMP}'
CANDIDATE_GLOB='round*_kernel.py'
EOF
  fi

  if [ -f systems/drkernel.env ]; then
    if grep -q '^SYSTEM_MODEL_HF_ID=' systems/drkernel.env; then
      sed -i 's|^SYSTEM_MODEL_HF_ID=.*|SYSTEM_MODEL_HF_ID="hkust-nlp/drkernel-14b"|' systems/drkernel.env
    else
      echo 'SYSTEM_MODEL_HF_ID="hkust-nlp/drkernel-14b"' >> systems/drkernel.env
    fi
  else
    cat > systems/drkernel.env <<'EOF'
SYSTEM_MODEL_ALIAS='drkernel14b'
SYSTEM_MODEL_HF_ID="hkust-nlp/drkernel-14b"
SYSTEM_MODEL_LOCAL_DIR='models/drkernel14b'
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/trained_model.py --task {TASK} --cand_dir {CAND_DIR} --rounds {ROUNDS} --seed {SEED} --temperature {TEMP}'
CANDIDATE_GLOB='round*_kernel.py'
EOF
  fi

  grep SYSTEM_MODEL_HF_ID systems/autotriton.env systems/drkernel.env
}

write_corrected_summary_tool() {
  log "Writing tools/collect_corrected_summary.py"

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
    # ksearch/cuda_agent/geak 제외
    if r.get("system") in {"ksearch", "cuda_agent", "geak"}:
        continue
    groups[(r["system"], r["level"])].append(r)

out_rows = []
for (system, level), rs in sorted(groups.items()):
    n = len(rs)
    if n == 0:
        continue
    runnable_vals = [float(r.get("runnable_rate") or 0.0) for r in rs]
    correct_vals = [float(r.get("correct_rate") or 0.0) for r in rs]
    pass_vals = [float(r.get("pass@1") or 0.0) for r in rs]
    fast_vals = [float(r.get("fast_1") or 0.0) for r in rs]

    speeds = []
    for r in rs:
        c = float(r.get("correct_rate") or 0.0)
        s = float(r.get("best_score") or 0.0)
        if c > 0 and s > 0:
            speeds.append(s)

    if speeds:
        sorted_s = sorted(speeds)
        gmean = math.exp(sum(math.log(max(s, 1e-12)) for s in speeds) / len(speeds))
        mean = sum(speeds) / len(speeds)
        median = sorted_s[len(sorted_s)//2] if len(sorted_s) % 2 else (sorted_s[len(sorted_s)//2 - 1] + sorted_s[len(sorted_s)//2]) / 2
        mx = max(speeds)
    else:
        gmean = mean = median = mx = 0.0

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
        "speedup_median_correct": round(median, 3),
        "speedup_max_correct": round(mx, 3),
    })

dst = Path("results/unified_summary_corrected.csv")
with dst.open("w", newline="") as f:
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "system","level","n_tasks","runnable","correct","pass@1","fast_1",
        "n_correct_speed","speedup_gmean_correct","speedup_mean_correct",
        "speedup_median_correct","speedup_max_correct"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

print("Wrote", dst)
for r in out_rows:
    print(r)
PY

  chmod +x tools/collect_corrected_summary.py
}

make_tiny5() {
  log "Making kernelbench_tiny5"

  if [ ! -f kernelbench_subset50.txt ]; then
    echo "ERROR: kernelbench_subset50.txt not found"
    exit 1
  fi
  if [ ! -f kernelbench_subset50.csv ]; then
    echo "ERROR: kernelbench_subset50.csv not found"
    exit 1
  fi

  head -5 kernelbench_subset50.txt > kernelbench_tiny5.txt

  python - <<'PY'
import csv
from pathlib import Path

tasks = set(x.strip() for x in Path("kernelbench_tiny5.txt").read_text().splitlines() if x.strip())
with open("kernelbench_subset50.csv") as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if r["task_path"] in tasks]
if not rows:
    raise SystemExit("No rows matched tiny5 tasks")

with open("kernelbench_tiny5.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("tiny5 tasks:", len(rows))
for r in rows:
    print(r["task_path"])
PY
}

verify_env() {
  log "Verifying environment"

  source_cuda_env

  python --version
  which python || true
  echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
  echo "nvcc=$(command -v nvcc || true)"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
  fi

  python - <<'PY'
import torch, triton
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

  echo
  echo "Current official5 env files:"
  for f in systems/cudaforge.env systems/autokernel.env systems/cuda_l1.env systems/autotriton.env systems/drkernel.env; do
    echo "----- $f -----"
    sed -n '1,60p' "$f" 2>/dev/null || true
  done
}

patch_all() {
  ensure_dirs
  write_cuda_env_if_missing
  source_cuda_env
  install_min_deps
  patch_autokernel_env_check
  patch_cuda_l1_env
  patch_official_checkpoints
  write_corrected_summary_tool
  make_tiny5
  verify_env

  log "Patch complete"
}

run_tiny5() {
  patch_all
  log "Running official5 tiny5"

  source_cuda_env

  rm -rf runs/cudaforge_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/autokernel_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_cudal1_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/autotriton_autotriton8b_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/drkernel_drkernel14b_kernelbench_tiny5_round1_repeat1_temp0.2

  mkdir -p logs

  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_tiny5.txt" \
  SYSTEMS="$OFFICIAL5_SYSTEMS" \
  nohup ./run_all_with_server.sh 1 1 0.2 900 \
    > logs/a100_official5_tiny5_round1.out 2>&1 &

  echo "Started official5 tiny5."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_official5_tiny5_round1.out"
}

run_round1() {
  patch_all
  log "Running official5 subset50 round1"

  source_cuda_env

  rm -rf runs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/autokernel_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_cudal1_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/autotriton_autotriton8b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/drkernel_drkernel14b_kernelbench_subset50_round1_repeat1_temp0.2

  mkdir -p logs

  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_subset50.txt" \
  SYSTEMS="$OFFICIAL5_SYSTEMS" \
  nohup ./run_all_with_server.sh 1 1 0.2 900 \
    > logs/a100_official5_subset50_round1.out 2>&1 &

  echo "Started official5 subset50 round1."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_official5_subset50_round1.out"
}

run_round3() {
  patch_all
  log "Running official5 subset50 round3"

  source_cuda_env

  rm -rf runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2
  rm -rf runs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2
  rm -rf runs/cuda_l1_cudal1_kernelbench_subset50_round3_repeat1_temp0.2
  rm -rf runs/cuda_l1_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2
  rm -rf runs/autotriton_autotriton8b_kernelbench_subset50_round3_repeat1_temp0.2
  rm -rf runs/drkernel_drkernel14b_kernelbench_subset50_round3_repeat1_temp0.2

  mkdir -p logs

  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_subset50.txt" \
  SYSTEMS="$OFFICIAL5_SYSTEMS" \
  nohup ./run_all_with_server.sh 3 1 0.2 900 \
    > logs/a100_official5_subset50_round3.out 2>&1 &

  echo "Started official5 subset50 round3."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_official5_subset50_round3.out"
}

status() {
  log "Process status"
  ps aux | grep -E "run_all_with_server|run_system|final_eval|eval_worker|hf_openai_server|uvicorn" | grep -v grep || true

  echo
  echo "===== recent logs ====="
  for f in \
    logs/a100_official5_tiny5_round1.out \
    logs/a100_official5_subset50_round1.out \
    logs/a100_official5_subset50_round3.out; do
    if [ -f "$f" ]; then
      echo
      echo "----- $f -----"
      tail -60 "$f"
    fi
  done

  echo
  echo "===== GPU ====="
  nvidia-smi || true
}

summary() {
  log "Collecting summary"

  python collect_results.py 2>/dev/null || true
  python tools/collect_corrected_summary.py 2>/dev/null || true

  echo
  echo "===== unified_summary_corrected.csv ====="
  cat results/unified_summary_corrected.csv 2>/dev/null || true

  echo
  echo "===== unified_summary.csv ====="
  cat results/unified_summary.csv 2>/dev/null || true
}

pack() {
  summary

  log "Packing upload tar.gz"

  rm -rf debug_upload_a100_official5_expand
  mkdir -p debug_upload_a100_official5_expand

  cp logs/a100_official5_tiny5_round1.out debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp logs/a100_official5_subset50_round1.out debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp logs/a100_official5_subset50_round3.out debug_upload_a100_official5_expand/ 2>/dev/null || true

  cp results/unified_summary.csv debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp results/unified_summary_corrected.csv debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp results/unified_per_task.csv debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp results/unified_table.tex debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp results/evaluation_sheet.json debug_upload_a100_official5_expand/ 2>/dev/null || true
  cp results/repo_lock.json debug_upload_a100_official5_expand/ 2>/dev/null || true

  mkdir -p debug_upload_a100_official5_expand/systems
  cp systems/cudaforge.env debug_upload_a100_official5_expand/systems/ 2>/dev/null || true
  cp systems/autokernel.env debug_upload_a100_official5_expand/systems/ 2>/dev/null || true
  cp systems/cuda_l1.env debug_upload_a100_official5_expand/systems/ 2>/dev/null || true
  cp systems/autotriton.env debug_upload_a100_official5_expand/systems/ 2>/dev/null || true
  cp systems/drkernel.env debug_upload_a100_official5_expand/systems/ 2>/dev/null || true

  mkdir -p debug_upload_a100_official5_expand/tools
  cp tools/collect_corrected_summary.py debug_upload_a100_official5_expand/tools/ 2>/dev/null || true

  cp use_a100_cuda126_env.sh debug_upload_a100_official5_expand/ 2>/dev/null || true

  find runs logs -type f | grep -E "cudaforge|autokernel|cuda_l1|autotriton|drkernel|summary.json|verdict|candidate_.*py|round000_kernel.py|round001_kernel.py|round002_kernel.py|raw_reply|server" \
    | while read -r f; do
        mkdir -p "debug_upload_a100_official5_expand/$(dirname "$f")"
        cp "$f" "debug_upload_a100_official5_expand/$f" 2>/dev/null || true
      done

  tar -czf a100_official5_subset50_expand_debug.tar.gz debug_upload_a100_official5_expand

  echo
  echo "Upload this file:"
  echo "$ROOT/a100_official5_subset50_expand_debug.tar.gz"
}

help_msg() {
  cat <<EOF
Usage:
  cd $ROOT
  bash a100_official5_expand.sh patch
  bash a100_official5_expand.sh tiny5
  bash a100_official5_expand.sh round1
  bash a100_official5_expand.sh round3
  bash a100_official5_expand.sh status
  bash a100_official5_expand.sh summary
  bash a100_official5_expand.sh pack

Recommended order:
  1) bash a100_official5_expand.sh patch
  2) bash a100_official5_expand.sh tiny5
  3) bash a100_official5_expand.sh round1
  4) bash a100_official5_expand.sh round3
  5) bash a100_official5_expand.sh pack

Upload:
  $ROOT/a100_official5_subset50_expand_debug.tar.gz
EOF
}

case "$MODE" in
  patch)
    patch_all
    ;;
  tiny5)
    run_tiny5
    ;;
  round1)
    run_round1
    ;;
  round3)
    run_round3
    ;;
  status)
    status
    ;;
  summary)
    summary
    ;;
  pack)
    pack
    ;;
  help|--help|-h)
    help_msg
    ;;
  *)
    echo "Unknown mode: $MODE"
    help_msg
    exit 1
    ;;
esac
