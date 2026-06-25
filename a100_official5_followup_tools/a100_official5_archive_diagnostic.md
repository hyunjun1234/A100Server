# A100 official5 subset50 archive 진단

- archive: `a100_official5_subset50_expand_debug.tar.gz`
- archive size: `1.97 MiB`
- extracted files: `5201`

## 포함된 핵심 결과 파일

- `unified_summary_corrected.csv`: `debug_upload_a100_official5_expand/unified_summary_corrected.csv`
- `unified_summary.csv`: `debug_upload_a100_official5_expand/unified_summary.csv`
- `unified_per_task.csv`: `debug_upload_a100_official5_expand/unified_per_task.csv`
- `evaluation_sheet.json`: `debug_upload_a100_official5_expand/evaluation_sheet.json`
- `repo_lock.json`: `debug_upload_a100_official5_expand/repo_lock.json`

## unified_summary_corrected.csv

```csv
system,level,n_tasks,runnable,correct,pass@1,fast_1,n_correct_speed,speedup_gmean_correct,speedup_mean_correct,speedup_median_correct,speedup_max_correct
autokernel,level1,25,1.0,0.427,0.427,0.08,13,0.789,1.44,1.0,8.874
autokernel,level2,15,1.0,0.111,0.111,0.022,3,0.63,0.749,0.996,1.002
autokernel,level3,10,0.9,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
autotriton,level1,25,0.387,0.2,0.2,0.107,6,1.795,2.73,1.013,8.523
autotriton,level2,15,0.356,0.044,0.044,0.0,2,0.864,0.873,0.873,0.999
autotriton,level3,10,0.4,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
cuda_l1,level1,25,0.88,0.44,0.44,0.16,11,1.262,1.758,1.0,8.856
cuda_l1,level2,15,0.933,0.133,0.133,0.133,2,37.493,630.805,630.805,1260.495
cuda_l1,level3,10,0.9,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
cudaforge,level1,25,0.987,0.44,0.44,0.093,13,0.995,0.995,1.0,1.002
cudaforge,level2,15,0.978,0.089,0.089,0.022,2,2.08,2.668,2.668,4.338
cudaforge,level3,10,0.867,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
drkernel,level1,25,0.84,0.307,0.307,0.187,11,1.715,2.656,2.131,5.637
drkernel,level2,15,0.667,0.044,0.044,0.022,1,121.824,121.824,121.824,121.824
drkernel,level3,10,0.6,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0
```

## unified_summary.csv

```csv
system,level,n_tasks,runnable,correct,pass@1,fast_1,best_speedup,geomean_speedup
autokernel,level1,25,1.0,0.427,0.427,0.08,0.749,0.801
autokernel,level2,15,1.0,0.111,0.111,0.022,0.15,0.749
autokernel,level3,10,0.9,0.0,0.0,0.0,0.0,0.0
autotriton,level1,25,0.387,0.2,0.2,0.107,0.655,2.712
autotriton,level2,15,0.356,0.044,0.044,0.0,0.116,0.873
autotriton,level3,10,0.4,0.0,0.0,0.0,0.0,0.0
cuda_l1,level1,25,0.88,0.44,0.44,0.16,0.773,1.758
cuda_l1,level2,15,0.933,0.133,0.133,0.133,84.107,630.805
cuda_l1,level3,10,0.9,0.0,0.0,0.0,0.0,0.0
cudaforge,level1,25,0.987,0.44,0.44,0.093,0.517,0.942
cudaforge,level2,15,0.978,0.089,0.089,0.022,0.356,1.221
cudaforge,level3,10,0.867,0.0,0.0,0.0,0.0,0.0
drkernel,level1,25,0.84,0.307,0.307,0.187,1.169,1.981
drkernel,level2,15,0.667,0.044,0.044,0.022,8.122,10.087
drkernel,level3,10,0.6,0.0,0.0,0.0,0.0,0.0
```

## 발견된 campaign

| system | rounds | repeat | temp | run dirs | task summaries |
|---|---:|---:|---:|---:|---:|
| autokernel | 1 | 1 | 0.2 | 2 | 6 |
| autokernel | 3 | 1 | 0.2 | 1 | 50 |
| autotriton | 1 | 1 | 0.2 | 2 | 6 |
| autotriton | 3 | 1 | 0.2 | 1 | 50 |
| cuda_agent | 1 | 1 | 0.2 | 1 | 1 |
| cuda_l1 | 1 | 1 | 0.2 | 2 | 6 |
| cuda_l1 | 3 | 1 | 0.2 | 1 | 50 |
| cudaforge | 1 | 1 | 0.2 | 2 | 6 |
| cudaforge | 3 | 1 | 0.2 | 1 | 50 |
| drkernel | 1 | 1 | 0.2 | 2 | 6 |
| drkernel | 3 | 1 | 0.2 | 1 | 50 |
| geak | 1 | 1 | 0.2 | 1 | 1 |
| ksearch | 1 | 1 | 0.2 | 2 | 6 |

## unified_per_task 구조

- rows: `250`
- columns: `system, level, task, n_candidates, runnable_rate, correct_rate, pass@1, fast_1, best_score, geomean_speedup`

| system | level | rows |
|---|---|---:|
| autokernel | level1 | 25 |
| autokernel | level2 | 15 |
| autokernel | level3 | 10 |
| autotriton | level1 | 25 |
| autotriton | level2 | 15 |
| autotriton | level3 | 10 |
| cuda_l1 | level1 | 25 |
| cuda_l1 | level2 | 15 |
| cuda_l1 | level3 | 10 |
| cudaforge | level1 | 25 |
| cudaforge | level2 | 15 |
| cudaforge | level3 | 10 |
| drkernel | level1 | 25 |
| drkernel | level2 | 15 |
| drkernel | level3 | 10 |

## 오류 signature 집계

| kind | matches |
|---|---:|
| oom | 12 |
| timeout | 10 |
| compile | 222 |
| correctness | 1246 |
| modelnew | 2 |
| module | 0 |
| server | 0 |
| device | 126 |
| disk | 0 |

### oom 예시

```text
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_38_L1Norm__rep0.out: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 7.11 GiB is free. Process 69901 has 29.52 GiB memory in use. Including non-PyTorch memory, this process has 42.49 GiB memory in use. Of the allocated memory 42.00 GiB is allocated by PyTorch, and 12.66 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_38_L1Norm__rep0.out: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 7.11 GiB is free. Process 69901 has 29.52 GiB memory in use. Including non-PyTorch memory, this process has 42.49 GiB memory in use. Of the allocated memory 42.00 GiB is allocated by PyTorch, and 12.66 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_38_L1Norm__rep0.out: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 7.11 GiB is free. Process 69901 has 29.52 GiB memory in use. Including non-PyTorch memory, this process has 42.49 GiB memory in use. Of the allocated memory 42.00 GiB is allocated by PyTorch, and 12.66 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/autokernel_38_L1Norm__rep0.out: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 7.19 GiB is free. Process 1859669 has 29.44 GiB memory in use. Including non-PyTorch memory, this process has 42.49 GiB memory in use. Of the allocated memory 42.00 GiB is allocated by PyTorch, and 12.66 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/38_L1Norm__rep0/candidates/llm_io/round000_judge_prompt.txt: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 264.56 MiB is free. Process 69901 has 28.39 GiB memory in use. Including non-PyTorch memory, this process has 50.48 GiB memory in use. Of the allocated memory 50.00 GiB is allocated by PyTorch, and 800.00 KiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/38_L1Norm__rep0/candidates/llm_io/round002_judge_prompt.txt: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.15 GiB of which 264.56 MiB is free. Process 69901 has 28.39 GiB memory in use. Including non-PyTorch memory, this process has 50.48 GiB memory in use. Of the allocated memory 50.00 GiB is allocated by PyTorch, and 800.00 KiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/38_L1Norm__rep0/candidates/llm_io/round002_judge_reply.txt: The error indicates that the GPU is running out of memory despite having a significant amount of total capacity. This suggests that the memory usage is inefficient or there might be other factors contributing to high memory consumption.
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/38_L1Norm__rep0/candidates/llm_io/round002_judge_reply.txt: By implementing these changes, you should be able to reduce memory usage and avoid the `CUDA out of memory` error.
```

### timeout 예시

```text
debug_upload_a100_official5_expand/runs/autokernel_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/eval_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 /home/jun/llm_run/.venv/bin/python3 telemetry/instrumented_final_eval.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/autokernel_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --glob candidate_*.py --task_work_dir /home/jun/unified_bench/runs/autokernel_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/final_eval"
debug_upload_a100_official5_expand/runs/autokernel_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/generation_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 bash -c python3 drivers/autokernel_kb_program_adapter.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --out /home/jun/unified_bench/runs/autokernel_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --rounds 1 --seed 20260611 --temperature 0.2"
debug_upload_a100_official5_expand/runs/autotriton_autotriton8b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/eval_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 /home/jun/llm_run/.venv/bin/python3 telemetry/instrumented_final_eval.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/autotriton_autotriton8b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --glob round*_kernel.py --task_work_dir /home/jun/unified_bench/runs/autotriton_autotriton8b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/final_eval"
debug_upload_a100_official5_expand/runs/autotriton_autotriton8b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/generation_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 bash -c python3 drivers/trained_model.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/autotriton_autotriton8b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --rounds 1 --seed 20260611 --temperature 0.2"
debug_upload_a100_official5_expand/runs/cuda_l1_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/eval_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 /home/jun/llm_run/.venv/bin/python3 telemetry/instrumented_final_eval.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/cuda_l1_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --glob candidate_*.py --task_work_dir /home/jun/unified_bench/runs/cuda_l1_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/final_eval"
debug_upload_a100_official5_expand/runs/cuda_l1_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/generation_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 bash -c python3 drivers/cuda_l1_artifact_adapter.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --out /home/jun/unified_bench/runs/cuda_l1_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --gpu-json a100.json"
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/eval_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 /home/jun/llm_run/.venv/bin/python3 telemetry/instrumented_final_eval.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/cudaforge_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --glob round*_kernel.py --task_work_dir /home/jun/unified_bench/runs/cudaforge_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/final_eval"
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/telemetry/generation_resource.txt: 	Command being timed: "timeout --kill-after=60s 1800 bash -c python3 drivers/cudaforge_inline.py --task /home/jun/unified_bench/third_party/KernelBench/KernelBench/level1/2_Standard_matrix_multiplication_.py --cand_dir /home/jun/unified_bench/runs/cudaforge_qwen14b_kernelbench_tiny1_round1_repeat1_temp0.2/2_Standard_matrix_multiplication__rep0/candidates --rounds 1 --seed 20260611 --temperature 0.2"
```

### compile 예시

```text
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_29_SwinMLP_rep0.out: [round 0] compiled=False correct=False latency_ms=None speedup=None error=compile: Traceback (most recent call last):
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_29_SwinMLP_rep0.out: [1/1] compiled=False correct=False speedup=None
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_59_conv_standard_3D__asymmetric_input__square_kernel_rep0.out: RuntimeError: Error building extension 'conv3d_cuda_3371aa8cb832c1c2': [1/3] /home/jun/llm_run/.cuda_env/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=conv3d_cuda_3371aa8cb832c1c2 -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_59_conv_standard_3D__asymmetric_input__square_kernel_rep0.out: RuntimeError: Error building extension 'conv3d_cuda_8a9a9806c6f207df': [1/3] /home/jun/llm_run/.cuda_env/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=conv3d_cuda_8a9a9806c6f207df -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_29_SwinMLP_rep0.out: [autokernel-kb-adapter] feedback: compiled=False correct=False speedup=None error=compile: Traceback (most recent call last):
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_29_SwinMLP_rep0.out: [autokernel-kb-adapter] feedback: compiled=False correct=False speedup=None error=compile: Traceback (most recent call last):
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_29_SwinMLP_rep0.out: [autokernel-kb-adapter] feedback: compiled=False correct=False speedup=None error=compile: Traceback (most recent call last):
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_29_SwinMLP_rep0.out: [1/3] compiled=False correct=False speedup=None
```

### correctness 예시

```text
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_80_Gemm_Max_Subtract_GELU_rep0.out: [round 0] compiled=True correct=False latency_ms=None speedup=None error=correctness: trial 0 mismatch (max abs 1.700e-0
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_80_Gemm_Max_Subtract_GELU_rep0.out: [1/1] compiled=True correct=False speedup=None
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_74_conv_transposed_1D_dilated_rep0.out: [round 0] compiled=True correct=False latency_ms=None speedup=None error=correctness: trial 0 mismatch (max abs 1.669e+0
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_74_conv_transposed_1D_dilated_rep0.out: [1/1] compiled=True correct=False speedup=None
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_13_DenseNet121TransitionLayer_rep0.out: [round 0] compiled=True correct=False latency_ms=None speedup=None error=correctness: trial 0 mismatch (max abs 1.371e+0
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_13_DenseNet121TransitionLayer_rep0.out: [1/1] compiled=True correct=False speedup=None
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_10_ResNet101_rep0.out: [round 0] compiled=True correct=False latency_ms=None speedup=None error=correctness: trial 0 mismatch (max abs 5.834e-0
debug_upload_a100_official5_expand/logs/cudaforge_qwen14b_kernelbench_subset50_round1_repeat1_temp0.2/cudaforge_10_ResNet101_rep0.out: [1/1] compiled=True correct=False speedup=None
```

### modelnew 예시

```text
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_tiny5_round1_repeat1_temp0.2/7_Matmul_with_small_K_dimension__rep0/candidates/llm_io/round000_judge_prompt.txt: TypeError: ModelNew.__init__() missing 3 required positional arguments: 'M', 'N', and 'K'
debug_upload_a100_official5_expand/runs/cudaforge_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/17_Matmul_with_transposed_B_rep0/candidates/llm_io/round002_judge_reply.txt: The error message indicates that the `forward` method of the `ModelNew` class is missing a value for the argument `B`. This suggests that the input tensors are not being passed correctly to the model during the execution.
```

### device 예시

```text
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_36_RMSNorm__rep0.out: RuntimeError: CUDA error: an illegal memory access was encountered
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_36_RMSNorm__rep0.out: RuntimeError: CUDA error: an illegal memory access was encountered
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_36_RMSNorm__rep0.out: RuntimeError: CUDA error: an illegal memory access was encountered
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_100_HingeLoss_rep0.out: RuntimeError: CUDA error: an illegal memory access was encountered
debug_upload_a100_official5_expand/logs/autokernel_qwen14b_kernelbench_subset50_round3_repeat1_temp0.2/autokernel_58_conv_transposed_3D__asymmetric_input__asymmetric_kernel_rep0.out: RuntimeError: CUDA error: an illegal memory access was encountered
debug_upload_a100_official5_expand/runs/autotriton_autotriton8b_kernelbench_subset50_round3_repeat1_temp0.2/20_LeakyReLU_rep0/candidates/llm_io/round002_raw_reply.txt: 5. We must ensure that the output tensor is on the same device and has the same dtype as the input.
debug_upload_a100_official5_expand/runs/autotriton_autotriton8b_kernelbench_subset50_round3_repeat1_temp0.2/20_LeakyReLU_rep0/candidates/llm_io/round000_raw_reply.txt: - We must ensure that the output tensor is on the same device and has the same shape as the input.
debug_upload_a100_official5_expand/runs/autotriton_autotriton8b_kernelbench_subset50_round3_repeat1_temp0.2/20_LeakyReLU_rep0/candidates/llm_io/round000_raw_reply.txt: - We'll allocate the output tensor on the same device and dtype as the input.
```

## 자동 판정

- **여러 round/repeat campaign이 같은 archive에 포함되어 있습니다.** 기존 `collect_results.py`가 전체 `runs/`를 긁는 구조라면 round1과 round3 결과가 섞일 수 있으므로 campaign별 집계가 필요합니다.
- LLM 호출 telemetry(`llm_calls.jsonl`)는 이 archive에 없습니다.

## 권장 후속 조치

1. 기존 결과를 삭제하지 않고 campaign별로 분리 집계합니다.
2. missing task만 system별 resume list로 만들어 이어서 실행합니다.
3. 확장은 `round3 repeat3`을 우선 수행하고, 그 다음 전체 250 task round1을 별도 campaign으로 실행합니다.
4. 새 실행에는 LLM call 수, compile/correctness/benchmark 시간, GPU telemetry를 함께 남깁니다.