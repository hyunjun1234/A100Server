#!/usr/bin/env bash
# Resume official5 subset50 round3 WITHOUT wiping completed tasks.
# run_system.sh's has_summary skips any (task,rep) that already has final_eval/summary.json,
# so this continues from where round3 stopped.
#
# IMPORTANT: do NOT use `bash a100_official5_expand.sh round3` to resume —
# that runs `rm -rf runs/*round3*` first and would delete completed progress.
set -euo pipefail
cd /home/jun/unified_bench
# always use the llm_run venv
source /home/jun/llm_run/.venv/bin/activate 2>/dev/null || true
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 \
SERVER_CUDA_VISIBLE_DEVICES=1 \
BENCH_CUDA_VISIBLE_DEVICES=0 \
TASK_LIST="kernelbench_subset50.txt" \
SYSTEMS="cudaforge autokernel cuda_l1 autotriton drkernel" \
nohup ./run_all_with_server.sh 3 1 0.2 900 \
  >> logs/a100_official5_subset50_round3.out 2>&1 &
echo "Resumed official5 round3 (PID $!). Completed tasks skipped via has_summary."
echo "  tail -f /home/jun/unified_bench/logs/a100_official5_subset50_round3.out"
