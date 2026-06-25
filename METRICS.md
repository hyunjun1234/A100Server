# 출력 metric 정의

## 표준 지표
- `candidate_compile_rate`: 생성 candidate 중 compile 성공 비율.
- `candidate_correct_rate`: 생성 candidate 중 correctness 통과 비율.
- `task_compile_success`: task에서 하나 이상의 candidate가 compile됨.
- `task_correct_success`: task에서 하나 이상의 candidate가 정답임.
- `pass_at_1`: 첫 candidate가 정답인지.
- `fast_at_1`: 첫 candidate가 정답이며 reference보다 빠른지.
- `best_speedup`: correct candidate 중 최대 `reference_latency / candidate_latency`.
- `speedup_gmean_correct`: correct task만 대상으로 한 geometric-mean speedup.

## Search efficiency
- `llm_calls_total`: kernel 하나를 처리하면서 발생한 LLM API call 수.
- `llm_calls_to_first_correct_est`: 첫 correct candidate까지의 추정 LLM call 수.
- `compile_attempts`, `correctness_trials`, `profile_runs`.
- `task_wall_s`, `generation_wall_s`, `llm_wait_s`.
- `generation_non_llm_s`: optimizer 내부 compile/search/tool 시간의 근사값.
- `unified_compile_s`, `unified_correctness_s`, `unified_benchmark_s`.
- `time_to_first_correct_s_est`.

## 병목/자원
- `bottleneck_stage`: 측정 stage 중 가장 긴 구간.
- `bottleneck_fraction`: 측정 stage 총합에서 병목 구간이 차지하는 비율.
- `gpu_energy_wh_est`: nvidia-smi power sampling의 사다리꼴 적분 추정치.
- `gpu_util_avg_pct`, `gpu_mem_max_mib`, `gpu_temp_max_c`.

## 보조 composite metric
주 지표가 아니라 진단용이다.
- `speed_gain_per_call = max(log2(best_speedup), 0) / max(llm_calls_total, 1)`
- `speed_gain_per_min = max(log2(best_speedup), 0) / max(task_wall_s / 60, eps)`
- `correct_per_call = task_correct_success / max(llm_calls_total, 1)`
