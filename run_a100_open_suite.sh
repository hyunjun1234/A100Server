#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-help}"
[ -f open_suite_config.env ] && source open_suite_config.env
[ -f native_benchmarks.env ] && source native_benchmarks.env
[ -f scripts/use_a100_cuda_env.sh ] && source scripts/use_a100_cuda_env.sh

make_tiny5() {
  if [ ! -f kernelbench_subset50.txt ]; then python3 make_subset_tasklist.py; fi
  head -5 kernelbench_subset50.txt > kernelbench_tiny5.txt
  python3 - <<'PY'
import csv
from pathlib import Path
tasks={x.strip() for x in Path('kernelbench_tiny5.txt').read_text().splitlines() if x.strip()}
rows=[r for r in csv.DictReader(open('kernelbench_subset50.csv')) if r['task_path'] in tasks]
if not rows: raise SystemExit('No tiny5 tasks matched kernelbench_subset50.csv')
with open('kernelbench_tiny5.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('tiny5:',len(rows))
PY
}
make_tiny1() {
  make_tiny5; head -1 kernelbench_tiny5.txt > kernelbench_tiny1.txt
  python3 - <<'PY'
import csv
from pathlib import Path
t=Path('kernelbench_tiny1.txt').read_text().strip(); rows=[r for r in csv.DictReader(open('kernelbench_subset50.csv')) if r['task_path']==t]
with open('kernelbench_tiny1.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('tiny1:',t)
PY
}
clone_all() { scripts/clone_all_open.sh; native/run_method_probes.sh; native/run_native_benchmark_probes.sh probe; }
doctor() { scripts/doctor.sh; native/run_method_probes.sh; native/run_native_benchmark_probes.sh probe; }
run_matrix() {
  local list="$1" systems="$2" rounds="$3" repeat="$4" temp="$5" timeout="$6"
  SERVER_CUDA_VISIBLE_DEVICES="${SERVER_GPU:-1}" BENCH_CUDA_VISIBLE_DEVICES="${BENCH_GPU:-0}" TASK_LIST="$list" SYSTEMS="$systems" telemetry/instrumented_run_all.sh "$rounds" "$repeat" "$temp" "$timeout"
}
tiny5() { make_tiny5; run_matrix kernelbench_tiny5.txt "${OFFICIAL5_SYSTEMS}" 1 1 "${TEMPERATURE:-0.2}" "${TASK_TIMEOUT:-1800}"; }
subset50() { run_matrix "${TASK_LIST:-kernelbench_subset50.txt}" "${OFFICIAL5_SYSTEMS}" "${ROUNDS:-3}" "${REPEAT:-1}" "${TEMPERATURE:-0.2}" "${TASK_TIMEOUT:-1800}"; }
kernelllm_smoke() { make_tiny5; run_matrix kernelbench_tiny5.txt kernelllm "${KERNELLLM_ROUNDS:-1}" 1 1.0 "${TASK_TIMEOUT:-1800}"; }
kernelllm_subset50() { run_matrix "${TASK_LIST:-kernelbench_subset50.txt}" kernelllm "${KERNELLLM_ROUNDS:-1}" "${REPEAT:-1}" 1.0 "${TASK_TIMEOUT:-1800}"; }
all_methods_smoke() { make_tiny1; run_matrix kernelbench_tiny1.txt "${OFFICIAL5_SYSTEMS} ${A100_CHECKPOINT_SYSTEMS:-kernelllm} ${OPTIONAL_SYSTEMS}" 1 1 "${TEMPERATURE:-0.2}" "${TASK_TIMEOUT:-1800}"; native/run_method_probes.sh; }
native_probe() { native/run_native_benchmark_probes.sh probe; }
native_full() { native/run_native_benchmark_probes.sh full; }
qimeng_probe() { native/run_qimeng_family.sh probe; }
qimeng_gemm() { native/run_qimeng_gemm.sh full; }
qimeng_attention() { native/run_qimeng_attention.sh full; }
qimeng_kernel() { native/run_qimeng_kernel.sh full; }
cuda_l2_probe() { native/run_cuda_l2.sh probe; }
cuda_l2_smoke() { native/run_cuda_l2.sh smoke; }
cuda_l2_full() { native/run_cuda_l2.sh full; }
summary() { python3 telemetry/aggregate.py --root "$ROOT" --task-list "${TASK_LIST:-kernelbench_subset50.txt}"; cat results/telemetry/telemetry_summary.csv 2>/dev/null || true; echo '--- bottlenecks ---'; head -21 results/telemetry/bottlenecks.csv 2>/dev/null || true; }
status() { ps aux | grep -E 'instrumented_run_all|instrumented_run_system|hf_openai_server|openai_proxy|instrumented_final_eval|instrumented_eval_worker|run_cuda_l2|run_qimeng' | grep -v grep || true; nvidia-smi || true; }
pack() { scripts/pack_results.sh; }
feedback() { scripts/make_feedback_bundle.sh; }
review() { cat SOURCES_REVIEW.md; }
help_message() { cat <<EOF
A100 open kernel suite v2
Usage:
  ./run_a100_open_suite.sh clone
  ./run_a100_open_suite.sh doctor
  ./run_a100_open_suite.sh tiny5
  ./run_a100_open_suite.sh subset50
  ./run_a100_open_suite.sh kernelllm_smoke
  ./run_a100_open_suite.sh kernelllm_subset50
  ./run_a100_open_suite.sh all_methods_smoke
  ./run_a100_open_suite.sh native_probe
  ./run_a100_open_suite.sh native_full
  ./run_a100_open_suite.sh qimeng_probe
  ./run_a100_open_suite.sh qimeng_gemm
  ./run_a100_open_suite.sh qimeng_attention
  ./run_a100_open_suite.sh qimeng_kernel
  ./run_a100_open_suite.sh cuda_l2_probe
  ./run_a100_open_suite.sh cuda_l2_smoke
  ./run_a100_open_suite.sh cuda_l2_full
  ./run_a100_open_suite.sh summary
  ./run_a100_open_suite.sh status
  ./run_a100_open_suite.sh feedback
  ./run_a100_open_suite.sh pack
  ./run_a100_open_suite.sh review
GPU ${BENCH_GPU:-0}: evaluation; GPU ${SERVER_GPU:-1}: LLM server.
EOF
}
case "$MODE" in
 clone) clone_all;; doctor) doctor;; tiny5) tiny5;; subset50) subset50;; kernelllm_smoke) kernelllm_smoke;; kernelllm_subset50) kernelllm_subset50;; all_methods_smoke) all_methods_smoke;; native_probe) native_probe;; native_full) native_full;; qimeng_probe) qimeng_probe;; qimeng_gemm) qimeng_gemm;; qimeng_attention) qimeng_attention;; qimeng_kernel) qimeng_kernel;; cuda_l2_probe) cuda_l2_probe;; cuda_l2_smoke) cuda_l2_smoke;; cuda_l2_full) cuda_l2_full;; summary) summary;; status) status;; feedback) feedback;; pack) pack;; review) review;; help|--help|-h) help_message;; *) echo "Unknown mode: $MODE"; help_message; exit 1;; esac
