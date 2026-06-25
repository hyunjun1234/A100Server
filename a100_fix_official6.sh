#!/usr/bin/env bash
# a100_fix_official6.sh
#
# 사용법:
#   cd /home/jun/unified_bench
#   bash a100_fix_official6.sh patch
#   bash a100_fix_official6.sh tiny5
#   bash a100_fix_official6.sh subset50
#   bash a100_fix_official6.sh pack
#
# 목적:
#   A100 서버에서 발견된 문제를 한 번에 패치한다.
#   - uv 없음 문제 해결
#   - datasets 확인/설치
#   - nvcc 11.5 대신 CUDA 12.x nvcc 우선 사용
#   - K-Search target GPU 문자열을 A100으로 수정
#   - AutoKernel env를 uv 기반으로 수정하고 stale kernel.py 복사 방지
#   - CUDA-L1은 A100 artifact 사용
#   - AutoTriton / DrKernel은 공식 HF checkpoint 사용
#   - llama_client_hf.py dtype -> torch_dtype 호환성 패치
#
# 기본 경로:
#   ROOT=/home/jun/unified_bench
# 필요하면:
#   ROOT=/다른/경로 bash a100_fix_official6.sh patch

set -euo pipefail

ROOT="${ROOT:-/home/jun/unified_bench}"
MODE="${1:-patch}"

OFFICIAL6_SYSTEMS="cudaforge autokernel ksearch cuda_l1 autotriton drkernel"

cd "$ROOT"

log() {
  echo
  echo "=========================================="
  echo "$*"
  echo "=========================================="
}

ensure_dirs() {
  mkdir -p models logs runs results systems drivers third_party
}

install_packages() {
  log "Installing/checking required Python packages"
  python -m pip install -U uv datasets
  python - <<'PY'
import importlib
for name in ["torch", "triton", "transformers", "openai", "requests", "datasets"]:
    m = importlib.import_module(name)
    print(f"{name}: {getattr(m, '__version__', 'OK')}")
PY
  uv --version
}

patch_cuda_env() {
  log "Writing use_a100_cuda126_env.sh"

  cat > use_a100_cuda126_env.sh <<'EOF'
#!/usr/bin/env bash

# Prefer pip-installed CUDA 12.x nvcc if available.
NVCC_ROOT=$(python - <<'PY'
from pathlib import Path
try:
    import nvidia.cuda_nvcc
    p = Path(nvidia.cuda_nvcc.__file__).resolve()
    candidates = [p.parent] + list(p.parents)
    for c in candidates:
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
else
  # fallback: system CUDA 12.x
  if [ -d /usr/local/cuda-12.6 ]; then
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
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_NAME="NVIDIA A100 80GB PCIe"
EOF

  chmod +x use_a100_cuda126_env.sh

  # shellcheck disable=SC1091
  source use_a100_cuda126_env.sh

  echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
  echo "nvcc path: $(command -v nvcc || true)"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version || true
  else
    echo "WARNING: nvcc not found after CUDA env patch"
  fi
}

patch_paths() {
  log "Replacing stale /home/jun paths with /home/jun in local config files"

  grep -RIl "/home/jun" \
    benchmark_config.env ./*.py ./*.sh drivers systems 2>/dev/null \
    | xargs -r sed -i 's|/home/jun|/home/jun|g' || true
}

patch_llama_dtype() {
  log "Patching llama_client_hf.py dtype -> torch_dtype if needed"

  if [ -f llama_client_hf.py ]; then
    cp -n llama_client_hf.py llama_client_hf.py.bak || true
    python - <<'PY'
from pathlib import Path
p = Path("llama_client_hf.py")
s = p.read_text()

import re
# idempotent: collapse any (torch_)*dtype=... to a single torch_dtype (never double-prefix)
s = re.sub(r"(?:torch_)*dtype(=(?:torch\.|self\.|dtype))", r"torch_dtype\1", s)

p.write_text(s)
print("patched llama_client_hf.py")
PY
    grep -n "torch_dtype\|dtype" llama_client_hf.py || true
  else
    echo "WARNING: llama_client_hf.py not found"
  fi
}

patch_system_envs() {
  log "Patching system env files for A100 official6"

  # Make lowercase autokernel path stable if repo was cloned as AutoKernel.
  if [ ! -e third_party/autokernel ] && [ -d third_party/AutoKernel ]; then
    ln -s AutoKernel third_party/autokernel
    echo "Created symlink: third_party/autokernel -> AutoKernel"
  fi

  # CUDA-L1: use official A100 artifact.
  cat > systems/cuda_l1.env <<'EOF'
# CUDA-L1 official released artifact adapter for A100.
# Re-evaluates official A100 JSON artifact on local A100.
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/cuda_l1_artifact_adapter.py --task {TASK} --out {CAND_DIR} --gpu-json a100.json'
CANDIDATE_GLOB='candidate_*.py'
EOF

  # K-Search: A100 target GPU string.
  cat > systems/ksearch.env <<'EOF'
# K-Search KernelBench adapter for A100.
SYSTEM_CWD='.'
SYSTEM_CMD='python3 drivers/ksearch_kernelbench_adapter.py --task {TASK} --out {CAND_DIR} --rounds {ROUNDS} --seed {SEED} --model-name qwen14b --base-url http://127.0.0.1:8000/v1 --target-gpu "NVIDIA A100 80GB PCIe" --language python'
CANDIDATE_GLOB='candidate_*.py'
EOF

  # AutoKernel: use uv and fail fast; do not copy stale kernel.py.
  cat > systems/autokernel.env <<'EOF'
# AutoKernel official repo workflow through uv.
# Removes stale kernel.py before each run, then runs official AutoKernel workflow.
SYSTEM_CWD='third_party/autokernel'
SYSTEM_CMD='bash -lc "set -euo pipefail; rm -f kernel.py kernel_*.py candidate*.py; uv run kernelbench/bridge.py setup --source hf --level {LEVEL} --problem {PROBLEM}; timeout 900s uv run autokernel.py; ls -lh kernel*.py; cp -v kernel*.py {CAND_DIR}/"'
CANDIDATE_GLOB='kernel*.py'
EOF

  # AutoTriton / DrKernel official checkpoints.
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

  echo
  echo "===== patched env summary ====="
  for f in systems/cuda_l1.env systems/ksearch.env systems/autokernel.env systems/autotriton.env systems/drkernel.env; do
    echo "----- $f -----"
    sed -n '1,80p' "$f"
  done
}

make_tiny5() {
  log "Making kernelbench_tiny5 task list"

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
    raise SystemExit("No rows matched tiny5 tasks in kernelbench_subset50.csv")

with open("kernelbench_tiny5.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("tiny5 tasks:", len(rows))
for r in rows:
    print(r["task_path"])
PY
}

verify_basic() {
  log "Basic verification"

  python --version
  which python || true
  command -v uv || true
  command -v nvcc || true
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
  echo "Remaining RTX6000 strings:"
  grep -Rni "RTX PRO 6000" benchmark_config.env systems drivers run_all_with_server.sh run_system.sh 2>/dev/null || true
}

patch_all() {
  ensure_dirs
  install_packages
  patch_cuda_env
  patch_paths
  patch_llama_dtype
  patch_system_envs
  make_tiny5
  verify_basic

  log "Patch complete"
  echo "Next:"
  echo "  bash a100_fix_official6.sh tiny5"
  echo "  bash a100_fix_official6.sh subset50"
}

run_tiny5() {
  patch_all
  log "Running A100 tiny5 envfix for AutoKernel + K-Search first"

  # shellcheck disable=SC1091
  source use_a100_cuda126_env.sh

  rm -rf runs/autokernel_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2
  rm -rf runs/ksearch_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2

  mkdir -p logs
  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_tiny5.txt" \
  SYSTEMS="autokernel ksearch" \
  nohup ./run_all_with_server.sh 1 1 0.2 900 \
    > logs/a100_autokernel_ksearch_tiny5_envfix.out 2>&1 &

  echo "Started tiny5 envfix job."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_autokernel_ksearch_tiny5_envfix.out"
}

run_subset50() {
  patch_all
  log "Running A100 official6 subset50 envfix"

  # shellcheck disable=SC1091
  source use_a100_cuda126_env.sh

  rm -rf runs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/autokernel_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/ksearch_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_cudal1_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/cuda_l1_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/autotriton_autotriton8b_kernelbench_subset50_round1_repeat1_temp0.2
  rm -rf runs/drkernel_drkernel14b_kernelbench_subset50_round1_repeat1_temp0.2

  mkdir -p logs
  CUDA_VISIBLE_DEVICES=0,1 \
  SERVER_CUDA_VISIBLE_DEVICES=1 \
  BENCH_CUDA_VISIBLE_DEVICES=0 \
  TASK_LIST="kernelbench_subset50.txt" \
  SYSTEMS="$OFFICIAL6_SYSTEMS" \
  nohup ./run_all_with_server.sh 1 1 0.2 900 \
    > logs/a100_official6_subset50_round1_envfix.out 2>&1 &

  echo "Started subset50 envfix job."
  echo "Log:"
  echo "  tail -f $ROOT/logs/a100_official6_subset50_round1_envfix.out"
}

pack_debug() {
  log "Packing debug zip"

  rm -rf debug_upload_a100_envfix
  mkdir -p debug_upload_a100_envfix

  cp logs/a100_official6_subset50_round1_envfix.out debug_upload_a100_envfix/ 2>/dev/null || true
  cp logs/a100_autokernel_ksearch_tiny5_envfix.out debug_upload_a100_envfix/ 2>/dev/null || true
  cp logs/a100_official6_subset50_round1.out debug_upload_a100_envfix/ 2>/dev/null || true
  cp logs/a100_official6_tiny5_round1.out debug_upload_a100_envfix/ 2>/dev/null || true

  cp results/unified_summary.csv debug_upload_a100_envfix/ 2>/dev/null || true
  cp results/unified_per_task.csv debug_upload_a100_envfix/ 2>/dev/null || true
  cp results/unified_table.tex debug_upload_a100_envfix/ 2>/dev/null || true
  cp results/evaluation_sheet.json debug_upload_a100_envfix/ 2>/dev/null || true
  cp results/repo_lock.json debug_upload_a100_envfix/ 2>/dev/null || true
  cp use_a100_cuda126_env.sh debug_upload_a100_envfix/ 2>/dev/null || true

  mkdir -p debug_upload_a100_envfix/systems
  cp systems/*.env debug_upload_a100_envfix/systems/ 2>/dev/null || true

  find runs logs -type f | grep -E "cudaforge|autokernel|ksearch|cuda_l1|autotriton|drkernel|summary.json|verdict|candidate_.*py|round000_kernel.py|raw_reply|server" \
    | while read -r f; do
        mkdir -p "debug_upload_a100_envfix/$(dirname "$f")"
        cp "$f" "debug_upload_a100_envfix/$f" 2>/dev/null || true
      done

  zip -r a100_official6_subset50_envfix_debug.zip debug_upload_a100_envfix
  echo
  echo "Upload this file:"
  echo "$ROOT/a100_official6_subset50_envfix_debug.zip"
}

case "$MODE" in
  patch)
    patch_all
    ;;
  tiny5)
    run_tiny5
    ;;
  subset50)
    run_subset50
    ;;
  pack)
    pack_debug
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage:"
    echo "  bash a100_fix_official6.sh patch"
    echo "  bash a100_fix_official6.sh tiny5"
    echo "  bash a100_fix_official6.sh subset50"
    echo "  bash a100_fix_official6.sh pack"
    exit 1
    ;;
esac
