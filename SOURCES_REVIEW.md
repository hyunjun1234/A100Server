# 추가 공개 시스템 검토 결과 (2026-06-23 기준)

## 요약

| 항목 | 공식 공개물 | A100 실행 가능성 | 이 패키지의 처리 |
|---|---|---:|---|
| QiMeng-GEMM | 공식 GitHub 코드 | 가능 | CUDA artifact native benchmark runner 제공 |
| QiMeng-TensorOp | 공식 GitHub 코드 | 현재 공개 artifact는 C920V2 전용 | A100에서는 명시적 skip 및 repo probe |
| QiMeng-Attention | 공식 GitHub 코드 | 논문은 A100 검증, repo는 CUDA/Python 포함 | clone/probe 및 configurable native runner |
| QiMeng-Kernel | 공식 GitHub 구현 | 조건부 | checkpoint/dataset이 준비된 경우 official inference/eval runner |
| KernelLLM | `facebook/KernelLLM` 8B 공식 checkpoint | 가능 | official checkpoint + KernelBench/Triton adapter |
| CUDA-L2 | 공식 GitHub, A100 1,000 HGEMM kernels | 가능 | CUTLASS v4.2.1 + official eval runner |
| CUDA-L1 | 공식 A100 artifact | 가능 | 기존 integrated artifact evaluation 유지 |

## QiMeng-GEMM

- Paper: AAAI 2025.
- Official repo: `https://github.com/QiMeng-IPRC/QiMeng-GEMM`.
- 공개 CUDA 코드는 `code/CUDA`에서 `make` 후 `./test M N K`로 benchmark한다.
- 공개 repo는 생성 과정 전체보다 최종 코드/artifact 평가에 가깝다.
- 이 패키지는 LLM call count를 0으로 기록하고, shape별 compile/run wall time, GPU telemetry, raw stdout을 남긴다.

## QiMeng-TensorOp

- Paper: IJCAI 2025.
- Official repo: `https://github.com/QiMeng-IPRC/QiMeng-TensorOp`.
- 현재 공개 repo README의 artifact는 C920V2 RISC-V SGEMM 전용이다.
- 논문 자체는 다양한 hardware와 NVIDIA GPU 결과를 다루지만, 공개 repo의 현재 artifact만으로 A100 official score를 만들 수 없다.
- 이 패키지는 A100에서 실행하지 않고 `unsupported_A100_public_artifact` 사유를 기록한다.

## QiMeng-Attention

- Paper: Findings of ACL 2025.
- Official repo: `https://github.com/QiMeng-IPRC/QiMeng-Attention`.
- 논문은 A100, RTX8000, T4에서 검증한다.
- repo는 CUDA/Python 코드를 공개하지만, 설치 시점의 entrypoint/환경을 먼저 probe한 뒤 실행해야 한다.
- `native_benchmarks.env`의 `QIMENG_ATTENTION_CMD`를 공식 README/repo 구조에 맞게 설정하면 telemetry wrapper로 실행한다.

## QiMeng-Kernel

- Paper: AAAI 2026 / arXiv 2511.20100.
- Official repo: `https://github.com/QiMeng-IPRC/QiMeng-Kernel`.
- repo는 `scripts/eval_generations.sh`, `scripts/inference.sh`, `scripts/train.sh` 인터페이스를 설명한다.
- repo는 active development 상태이며 checkpoint/training/offline dataset 공개가 진행 중이다.
- `QIMENG_KERNEL_CHECKPOINT`, dataset, run_dir가 제공된 경우에만 official runner를 실행한다. 없으면 skip reason을 남긴다.

## KernelLLM

- Official checkpoint: `facebook/KernelLLM`.
- 8B BF16 model, Llama 3.1 Instruct 기반 PyTorch-to-Triton SFT.
- 모델 카드의 공식 평가 setting은 KernelBench-Triton Level 1, H100, temperature=1.0, top_p=0.97이다.
- A100 결과는 cross-hardware 재평가이므로 H100 논문 수치와 직접 동일시하지 않는다.
- 이 패키지는 official checkpoint와 Triton-oriented prompt를 사용하고 task별 LLM calls/time을 기록한다.

## CUDA-L2

- Paper: arXiv 2512.02551.
- Official repo: `https://github.com/deepreinforce-ai/CUDA-L2`.
- A100 HGEMM 1,000 configuration artifact와 offline/server benchmark를 공개한다.
- PyTorch 2.6+, CUTLASS v4.2.1, `TORCH_CUDA_ARCH_LIST=8.0`이 필요하다.
- 이 패키지는 selected shape smoke/subset runner를 제공한다. 전체 1,000 shape 실행은 시간이 매우 길 수 있으므로 명시적으로 요청할 때만 수행한다.

## 비교 원칙

1. 생성 system 결과와 released artifact 결과를 같은 column에서 혼동하지 않는다.
2. KernelLLM의 A100 score는 official checkpoint의 A100 transfer 결과이다.
3. QiMeng-TensorOp는 공개 C920V2 artifact만으로 A100 score를 만들지 않는다.
4. QiMeng-GEMM/CUDA-L2는 operator-specific native benchmark로 보고, KernelBench 평균과 합산하지 않는다.
5. 모든 skip/failure 이유를 feedback bundle에 포함한다.
