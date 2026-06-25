# unified_bench — 공개 시스템 통합 벤치마크 (ibm 구조 규약)

기존 CudaForge 재현실험 디렉토리(ibm/)와 동일한 작업 규약을 따른다:
평면 스크립트 + `benchmark_config.env` + 위치 인자, `models/ runs/ logs/ results/`,
summary.json 기반 resume, per-task `timeout`, `task_status.csv`, 서버 health check.

## 디렉토리

```
unified_bench/
├── benchmark_config.env      # 전역 설정 (모델/서버/검증기/seed)
├── repos.yaml                # ★ 검증된 repo URL (아래 표) — clone_repos.sh가 소비
├── clone_repos.sh            # clone + SHA pin → results/repo_lock.json
├── download_model.py         # (기존 ibm 것 그대로)
├── hf_openai_server.py       # (기존 ibm 것; import 경로만 로컬화)
├── llama_client_hf.py
├── make_all_tasklist.py      # 전체 250 task (기존 것 이식)
├── make_subset_tasklist.py   # ★ 동결 50 task (L1:25/L2:15/L3:10, ROOT_SEED)
├── eval_worker.py            # 격리 프로세스 판정 (compile/correctness/timing)
├── final_eval.py             # ★ 통합 검증기 — 모든 보고 수치의 유일한 출처
├── run_system.sh             # ★ 시스템 1개 실행 (ibm run_benchmark.sh 일반화)
├── run_all_with_server.sh    # 전체 launcher (download→server→loop→collect)
├── collect_results.py        # system×level 집계 + booktabs LaTeX
├── drivers/                  # baseline_loop / cudaforge_inline / trained_model
├── systems/<s>.env           # 시스템별 실행 명령 + 후보 수확 규약
└── models/ runs/ logs/ results/ third_party/
```

## 실행

```bash
# 0) 의존성: torch(cu128)+triton+ninja, fastapi+uvicorn, openai, transformers, pyyaml
./clone_repos.sh
python3 make_subset_tasklist.py            # 50개 동결 (1회)
./run_all_with_server.sh 10 3 0.2 3600     # ROUNDS REPEAT TEMP TIMEOUT
# 또는 개별:
SYSTEMS="baseline_loop cudaforge" ./run_all_with_server.sh 10 3 0.2 3600
./run_system.sh cudaforge kernelbench_subset50.txt 10 3 0.2 3600   # 서버 떠있을 때
python3 collect_results.py                 # results/unified_summary.csv + unified_table.tex
```

runs 레이아웃: `runs/<EXP>/<task>_rep<N>/{candidates/roundNNN_kernel.py, candidates/llm_io/, final_eval/{verdicts,summary}.json}` — resume은 summary.json 존재 여부.

## 확인된 repo (2026-06-11 웹 검색)

| 시스템 | URL | 비고 |
|---|---|---|
| KernelBench | github.com/ScalingIntelligence/KernelBench | benchmark |
| CudaForge | github.com/OptimAI-Lab/CudaForge | Coder+Judge, training-free |
| AutoKernel | github.com/RightNow-AI/autokernel | uv 기반, KernelBench bridge 내장 |
| K-Search | github.com/caoshiyi/K-Search | 주 평가는 FlashInfer — KB 입력 어댑팅 필요 |
| GEAK | github.com/AMD-AGI/GEAK (+ AMD-AIG-AIMA/GEAK-eval) | AMD 지향 — NVIDIA 평가 비대칭 명시 |
| CUDA-L1 | github.com/deepreinforce-ai/CUDA-L1 | kernel 250개+평가 공개; **checkpoint 공개 여부 확인** |
| AutoTriton | github.com/AI9Stars/AutoTriton | weights 공개 — HF id는 README 링크에서 기입 |
| Dr.Kernel | github.com/hkust-nlp/KernelGYM | 환경+학습+모델+데이터 일괄 공개 |
| CUDA-Agent | github.com/BytedTsinghua-SIA/CUDA-Agent | 데이터/SKILL.md/환경/추론결과 공개; **checkpoint 미공개 가능성** |

## 남은 확정 사항 (시험 실행 1 task로 각각 확인)

1. **HF checkpoint id** — `repos.yaml`의 models 섹션과 `systems/{autotriton,drkernel,cuda_l1,cuda_agent}.env`의
   `SYSTEM_MODEL_HF_ID`. 각 repo README의 공식 링크에서만 기입 (추정 금지).
2. **repo CLI** — `systems/{autokernel,ksearch,geak,cudaforge}.env`의 `[확인 필요]` CMD와
   `CANDIDATE_GLOB`. 확인 절차: `./run_system.sh <s> <(echo "third_party/KernelBench/KernelBench/level1/19_ReLU.py") 2 1 0.2 1200`
   후 `runs/.../candidates/`에 후보가 수확되는지 확인.
3. **checkpoint 미공개 시스템** — `KERNEL_ARTIFACT_DIR` 경로(공개 kernel/추론결과)를 repo 구조에 맞게
   수정하면 artifact 모드로 동작: 공개 산출물을 후보로 간주해 동일 검증기로 재판정.
   이 경우 "모델 재실행"이 아니라 "공개 산출물 재검증"임을 evaluation sheet와 논문에 명시.

## 공정성 메모

- training-free 계열(baseline/CudaForge/AutoKernel/K-Search/GEAK)은 전부 동일 base model
  (Qwen2.5-Coder-14B) endpoint 사용 → 생성기 변인 제거.
- 시스템 내부 자체 평가는 제어 구조의 일부로 허용. **보고 수치는 final_eval.py 재판정만 사용**
  (correctness 입력 seed는 ROOT_SEED에서 파생, 시스템에 비공개).
- GPU 1대 공유: 서버가 같은 GPU에 있으면 timing 오염 — `SERVER_CUDA_VISIBLE_DEVICES`로 분리하거나,
  생성 phase 완료 후 서버 내리고 final_eval만 재실행하는 2-pass도 가능
  (candidates는 보존되므로 final_eval만 다시 돌리면 됨).
