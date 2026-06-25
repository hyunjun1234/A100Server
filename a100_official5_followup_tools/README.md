# A100 official5 follow-up tools

1. `a100_official5_archive_diagnostic.md`에서 업로드 archive 진단을 확인합니다.
2. `a100_official5_resume_expand.sh`를 `/home/jun/unified_bench`에 복사합니다.
3. 아래 순서로 실행합니다.

```bash
cd /home/jun/unified_bench
chmod +x a100_official5_resume_expand.sh

./a100_official5_resume_expand.sh audit
./a100_official5_resume_expand.sh resume_round3
./a100_official5_resume_expand.sh expand_rep3
./a100_official5_resume_expand.sh expand_all250_r1
./a100_official5_resume_expand.sh pack
```

최종 업로드 파일:

```text
/home/jun/unified_bench/a100_official5_followup_*.tar.gz
```
