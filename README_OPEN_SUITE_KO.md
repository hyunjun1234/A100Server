# A100 Open Kernel Suite v3 Download Bundle

이 zip은 다운로드 가능한 완전한 묶음입니다.

구성:

- `a100_open_kernel_suite_v2`의 전체 telemetry/native/runner 코드
- 실제 A100 실행 중 발견된 AutoKernel `uv sync` 및 `autokernel.py` 문제 수정
- AutoKernel helper `PYTHONPATH` 수정
- A100 official5 확장 runner
- 기존 v3 설계 문서 `README_V3_DRAFT.md`

## 설치

```bash
cd /home/jun
unzip a100_open_kernel_suite_v3.zip

bash a100_open_kernel_suite_v3/scripts/install_v3.sh \
  /home/jun/unified_bench

cd /home/jun/unified_bench
```

## 실행

먼저 tiny5:

```bash
bash a100_official5_expand.sh tiny5
tail -f logs/a100_official5_tiny5_round1.out
```

그 다음 subset50 round3:

```bash
bash a100_official5_expand.sh round3
tail -f logs/a100_official5_subset50_round3.out
```

결과 묶기:

```bash
bash a100_official5_expand.sh pack
```

업로드할 파일:

```text
/home/jun/unified_bench/a100_official5_subset50_expand_debug.tar.gz
```

## 주의

`a100_fix_official6.sh patch` 같은 예전 스크립트는 AutoKernel 설정을 되돌릴 수 있으므로 사용하지 마세요.
