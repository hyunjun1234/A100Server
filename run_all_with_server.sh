#!/bin/bash
# run_all_with_server.sh — 전체 실험 launcher (ibm 구조와 동일한 5단계 흐름)
#
# 사용: ./run_all_with_server.sh [ROUNDS] [REPEAT] [TEMP] [TASK_TIMEOUT]
# 시스템 선택: SYSTEMS="baseline_loop cudaforge" ./run_all_with_server.sh ...

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f benchmark_config.env ] && source benchmark_config.env

ROUNDS=${1:-10}
REPEAT=${2:-3}
TEMP=${3:-0.2}
TASK_TIMEOUT=${4:-3600}
TASK_LIST=${TASK_LIST:-"kernelbench_subset50.txt"}
MANIFEST_CSV="${TASK_LIST%.txt}.csv"
SYSTEMS=${SYSTEMS:-"baseline_loop cudaforge autokernel ksearch geak cuda_l1 autotriton drkernel cuda_agent"}

mkdir -p "$RUNS_DIR" "$RESULTS_DIR" "$LOGS_DIR" models third_party

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  [ -n "${CONDA_ENV:-}" ] && conda activate "$CONDA_ENV"
fi

# [auto-venv] python 환경이 없으면 프로젝트 venv로 폴백 (python, torch, transformers ... 제공)
if [ -z "${VIRTUAL_ENV:-}" ] && ! command -v python >/dev/null 2>&1 && [ -f /home/jun/llm_run/.venv/bin/activate ]; then
  source /home/jun/llm_run/.venv/bin/activate
fi

start_server () {  # $1=model_path_or_id  $2=alias
  local SERVER_LOG="${SCRIPT_DIR}/${LOGS_DIR}/hf_server_${2}_temp${TEMP}.out"
  if pgrep -f "hf_openai_server.py" > /dev/null; then
    pkill -f "hf_openai_server.py" || true; sleep 5
  fi
  echo "Starting server: $1 (alias $2) -> $SERVER_LOG"
  if [ -n "${SERVER_CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$SERVER_CUDA_VISIBLE_DEVICES" env -u PYTHONPATH nohup python "${SCRIPT_DIR}/hf_openai_server.py" \
      --model_id "$1" --host "$SERVER_HOST" --port "$SERVER_PORT" \
      --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" --top_p "$TOP_P" \
      > "$SERVER_LOG" 2>&1 &
  else
    env -u PYTHONPATH nohup python "${SCRIPT_DIR}/hf_openai_server.py" \
      --model_id "$1" --host "$SERVER_HOST" --port "$SERVER_PORT" \
      --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMP" --top_p "$TOP_P" \
      > "$SERVER_LOG" 2>&1 &
  fi
  for i in $(seq 1 240); do
    curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null && { echo "server ready"; return 0; }
    [ "$i" -eq 240 ] && { echo "ERROR: server 기동 실패 ($SERVER_LOG)"; exit 1; }
    sleep 5
  done
}

download_and_path () {  # $1=hf_id $2=alias -> echo local dir
  local d="${SCRIPT_DIR}/models/${2}"
  python "${SCRIPT_DIR}/download_model.py" --model_id "$1" --local_dir "$d" >&2
  echo "$d"
}

echo "== Step 1. repos =="
./clone_repos.sh

echo "== Step 2. task list (동결 50개) =="
[ -f "$TASK_LIST" ] || python3 make_subset_tasklist.py

echo "== Step 3. evaluation sheet (환경 스냅샷) =="
python3 - <<PY
import json, subprocess, platform
def sh(c):
    try: return subprocess.run(c, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return str(e)
sheet = {"gpu": sh(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"]),
         "nvcc": sh(["nvcc","--version"]).splitlines()[-1] if sh(["nvcc","--version"]) else "",
         "python": platform.python_version(),
         "rounds": "$ROUNDS", "repeat": "$REPEAT", "temp": "$TEMP",
         "task_list": "$TASK_LIST", "root_seed": "$ROOT_SEED",
         "repo_lock": json.load(open("results/repo_lock.json")) if __import__("os").path.exists("results/repo_lock.json") else None}
try:
    import torch; sheet["torch"] = torch.__version__; sheet["cuda_runtime"] = torch.version.cuda
except Exception: pass
json.dump(sheet, open("results/evaluation_sheet.json","w"), indent=2, ensure_ascii=False)
print("results/evaluation_sheet.json")
PY

echo "== Step 4. systems loop =="
CURRENT_MODEL=""
for SYS in $SYSTEMS; do
  source "systems/${SYS}.env"
  # checkpoint 계열: 해당 모델로 서버 교체 / 그 외: base model 유지
  if [ -n "${SYSTEM_MODEL_HF_ID:-}" ]; then
    WANT="$SYSTEM_MODEL_HF_ID"; ALIAS="${SYSTEM_MODEL_ALIAS}"
  elif [ -n "${KERNEL_ARTIFACT_DIR:-}" ] && [ -z "${SYSTEM_MODEL_HF_ID:-}" ]; then
    WANT="__artifact__"; ALIAS="artifact"
  else
    WANT="$MODEL_ID"; ALIAS="${MODEL_ALIAS}_${MODEL_SIZE_TAG}"
  fi
  if [ "$WANT" != "$CURRENT_MODEL" ] && [ "$WANT" != "__artifact__" ]; then
    LOCAL=$(download_and_path "$WANT" "$ALIAS")
    start_server "$LOCAL" "$ALIAS"
    CURRENT_MODEL="$WANT"
  fi
  # checkpoint id가 비어 있는 checkpoint 시스템은 건너뛰되 명시
  if grep -q "SYSTEM_MODEL_HF_ID=''" "systems/${SYS}.env" && [ -z "${KERNEL_ARTIFACT_DIR:-}" ]; then
    echo "[SKIP] $SYS: SYSTEM_MODEL_HF_ID 미기입 (systems/${SYS}.env)"; 
    unset SYSTEM_MODEL_HF_ID SYSTEM_MODEL_ALIAS KERNEL_ARTIFACT_DIR SYSTEM_CWD SYSTEM_CMD CANDIDATE_GLOB
    continue
  fi
  ./run_system.sh "$SYS" "$TASK_LIST" "$ROUNDS" "$REPEAT" "$TEMP" "$TASK_TIMEOUT"
  unset SYSTEM_MODEL_HF_ID SYSTEM_MODEL_ALIAS KERNEL_ARTIFACT_DIR SYSTEM_CWD SYSTEM_CMD CANDIDATE_GLOB
done

echo "== Step 5. collect =="
python3 collect_results.py --runs_dir "$RUNS_DIR" --manifest "$MANIFEST_CSV" \
  --out "${RESULTS_DIR}/unified_summary.csv"
echo "DONE -> ${RESULTS_DIR}/unified_summary.csv / unified_table.tex"
