"""make_subset_tasklist.py — L1 25 / L2 15 / L3 10 = 50개 동결 샘플링.

출력 형식은 ibm 구조의 kernelbench_all_levels.txt와 동일 (상대경로 줄단위)
→ run_system.sh가 그대로 소비. 한 번 생성되면 재생성하지 않는다 (동결).
사용: python3 make_subset_tasklist.py
"""
import csv
import os
import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KB_ROOT = Path(os.environ.get("KB_ROOT", SCRIPT_DIR / "third_party/KernelBench"))
SEED = int(os.environ.get("ROOT_SEED", "20260611"))
PLAN = {"level1": 25, "level2": 15, "level3": 10}

TXT = SCRIPT_DIR / "kernelbench_subset50.txt"
CSV = SCRIPT_DIR / "kernelbench_subset50.csv"

if TXT.exists():
    print(f"이미 동결됨: {TXT} (재샘플링하려면 파일 삭제)")
    raise SystemExit(0)


def task_id(p: Path) -> int:
    m = re.match(r"(\d+)_", p.name)
    return int(m.group(1)) if m else 10**9


rng = random.Random(SEED)
rows = []
for level, n in PLAN.items():
    d = KB_ROOT / "KernelBench" / level
    if not d.exists():
        d = KB_ROOT / level
    pool = sorted(d.glob("*.py"), key=task_id)
    assert len(pool) >= n, f"{level}: {len(pool)} < {n}"
    for p in sorted(rng.sample(pool, n), key=task_id):
        rows.append({"level": level, "task_id": task_id(p),
                     "task_path": str(p.relative_to(SCRIPT_DIR)),
                     "task_name": p.name})

TXT.write_text("\n".join(r["task_path"] for r in rows) + "\n")
with CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["level", "task_id", "task_path", "task_name"])
    w.writeheader(); w.writerows(rows)

print(f"Saved: {TXT} / {CSV}  (seed={SEED}, total={len(rows)})")
for level in PLAN:
    print(f"  {level}: {sum(1 for r in rows if r['level'] == level)}")
