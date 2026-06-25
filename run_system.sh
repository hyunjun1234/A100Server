#!/bin/bash
# run_system.sh — 시스템 1개 × task list 실행 (ibm run_benchmark.sh 규약의 일반화)
#
# 사용: ./run_system.sh SYSTEM [TASK_LIST] [ROUNDS] [REPEAT] [TEMP] [TASK_TIMEOUT]
#  예: ./run_system.sh cudaforge kernelbench_subset50.txt 10 3 0.2 3600
#
# 흐름 (task마다): ① 시스템 driver/repo가 후보 생성 → CAND_DIR
#                 ② final_eval.py가 통합 검증기로 전 후보 재판정 → summary.json
# resume: summary.json 존재하는 (task, repeat)는 SKIP (ibm has_summary 방식)

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f benchmark_config.env ] && source benchmark_config.env

SYSTEM=${1:?usage: run_system.sh SYSTEM [TASK_LIST] [ROUNDS] [REPEAT] [TEMP] [TIMEOUT]}
TASK_LIST=${2:-kernelbench_subset50.txt}
ROUNDS=${3:-10}
REPEAT=${4:-3}
TEMP=${5:-0.2}
TASK_TIMEOUT=${6:-3600}

SYS_ENV="systems/${SYSTEM}.env"
[ -f "$SYS_ENV" ] || { echo "ERROR: $SYS_ENV 없음"; exit 1; }
source "$SYS_ENV"

# checkpoint 계열이면 서버 모델 교체용 메타가 있을 수 있음 (run_all_with_server.sh가 처리)
EVAL_MODEL_NAME=${SYSTEM_MODEL_ALIAS:-$MODEL_ALIAS}

BASE_NAME="$(basename "$TASK_LIST" .txt)"
EXP_NAME="${SYSTEM}_${EVAL_MODEL_NAME}_${BASE_NAME}_round${ROUNDS}_repeat${REPEAT}_temp${TEMP}"
WORK_DIR="${SCRIPT_DIR}/${RUNS_DIR}/${EXP_NAME}"
LOG_DIR="${SCRIPT_DIR}/${LOGS_DIR}/${EXP_NAME}"
STATUS_CSV="${WORK_DIR}/task_status.csv"
mkdir -p "$WORK_DIR" "$LOG_DIR"
[ -f "$STATUS_CSV" ] || echo "task_path,repeat,exit_code,log_path,start_time,end_time" > "$STATUS_CSV"

# conda + CUDA toolchain (ibm 규약 그대로)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  [ -n "${CONDA_ENV:-}" ] && conda activate "$CONDA_ENV"
fi

# [auto-venv] python 환경이 없으면 프로젝트 venv로 폴백 (python, torch, transformers ... 제공)
if [ -z "${VIRTUAL_ENV:-}" ] && ! command -v python >/dev/null 2>&1 && [ -f /home/jun/llm_run/.venv/bin/activate ]; then
  source /home/jun/llm_run/.venv/bin/activate
fi
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export CUDA_PATH=${CUDA_PATH:-$CONDA_PREFIX}
export CUDACXX=${CUDACXX:-$CONDA_PREFIX/bin/nvcc}

# 통합 검증기/endpoint 환경 (driver와 final_eval이 소비)
export ROOT_SEED NUM_CORRECTNESS_TRIALS ATOL RTOL EVAL_WARMUP EVAL_TIMING_ITERS
export OPENAI_BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"
export OPENAI_API_KEY="EMPTY"
export EVAL_MODEL="$EVAL_MODEL_NAME"
export TOP_P MAX_NEW_TOKENS

echo "=========================================="
echo "System: $SYSTEM    Experiment: $EXP_NAME"
echo "GPU: $GPU_NAME (device $DEVICE_ID)  Rounds: $ROUNDS  Repeat: $REPEAT  Temp: $TEMP"
echo "=========================================="

curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null || {
  echo "ERROR: server not running at ${SERVER_HOST}:${SERVER_PORT}"; exit 1; }

has_summary () {  # WORK_DIR 내 task+repeat summary.json 존재 검사 (ibm 방식)
  [ -f "${WORK_DIR}/$(basename "$1" .py)_rep$2/final_eval/summary.json" ]
}

while read -r task; do
  [ -z "$task" ] && continue
  name=$(basename "$task" .py)
  # AutoKernel류를 위한 level/problem 파싱
  LEVEL=$(echo "$task" | grep -o 'level[0-9]' | grep -o '[0-9]')
  PROBLEM=$(echo "$name" | grep -o '^[0-9]*')

  for rep in $(seq 0 $((REPEAT-1))); do
    if has_summary "$task" "$rep"; then
      echo "SKIP completed: $task (rep $rep)"; continue
    fi
    TASK_DIR="${WORK_DIR}/${name}_rep${rep}"
    CAND_DIR="${TASK_DIR}/candidates"
    EVAL_DIR="${TASK_DIR}/final_eval"
    mkdir -p "$CAND_DIR" "$EVAL_DIR"
    log_path="${LOG_DIR}/${SYSTEM}_${name}_rep${rep}.out"
    SEED=$((ROOT_SEED + rep))

    echo "=========================================="
    echo "Running $task (rep $rep)  seed=$SEED"
    echo "=========================================="
    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    rm -rf /tmp/torch_ext_* 2>/dev/null || true

    # ① 후보 생성 — SYSTEM_CMD 템플릿 치환
    CMD=${SYSTEM_CMD//\{TASK\}/"${SCRIPT_DIR}/${task}"}
    CMD=${CMD//\{CAND_DIR\}/"$CAND_DIR"}
    CMD=${CMD//\{ROUNDS\}/"$ROUNDS"}
    CMD=${CMD//\{SEED\}/"$SEED"}
    CMD=${CMD//\{TEMP\}/"$TEMP"}
    CMD=${CMD//\{LEVEL\}/"$LEVEL"}
    CMD=${CMD//\{PROBLEM\}/"$PROBLEM"}

    # artifact 모드: 공개 산출 kernel을 후보로 직접 복사 (checkpoint 미공개 시스템용)
    if [ -n "${KERNEL_ARTIFACT_DIR:-}" ] && [ -z "${SYSTEM_MODEL_HF_ID:-}" ]; then
      echo "[artifact mode] ${KERNEL_ARTIFACT_DIR} 에서 task 매칭 kernel 복사"
      find "${SCRIPT_DIR}/${KERNEL_ARTIFACT_DIR}" -name "*${name}*" -name "*.py" \
        -exec cp -v {} "$CAND_DIR/" \; > "$log_path" 2>&1
      gen_status=$?
    else
      ( cd "${SYSTEM_CWD:+${SCRIPT_DIR}/${SYSTEM_CWD}}" 2>/dev/null || cd "$SCRIPT_DIR"
        timeout --kill-after=60s "$TASK_TIMEOUT" bash -c "$CMD" ) \
        > "$log_path" 2>&1
      gen_status=$?
    fi

    # ② 통합 재판정 (모든 시스템 공통, 보고 수치의 유일한 출처)
    timeout --kill-after=60s "$TASK_TIMEOUT" python3 final_eval.py \
      --task "${SCRIPT_DIR}/${task}" \
      --cand_dir "$CAND_DIR" --glob "${CANDIDATE_GLOB:-round*_kernel.py}" \
      --task_work_dir "$EVAL_DIR" >> "$log_path" 2>&1
    status=$?
    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    echo "\"$task\",$rep,$status,\"$log_path\",\"$start_time\",\"$end_time\"" >> "$STATUS_CSV"
    echo "Finished $task rep$rep (gen=$gen_status, eval=$status)"
    [ "$status" -eq 124 ] && echo "TIMEOUT: $task"
    if [ "$status" -ne 0 ] || [ "$gen_status" -ne 0 ]; then
      echo "ERROR SUMMARY for $task"
      grep -i "traceback\|error\|failed\|timeout\|cuda out\|killed" "$log_path" | tail -20
    fi
  done
done < "${SCRIPT_DIR}/${TASK_LIST}"

echo "All tasks finished for system: $SYSTEM"
