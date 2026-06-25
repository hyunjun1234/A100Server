import csv
import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_KB_PARENT_ROOT = os.environ.get("KB_PARENT_ROOT", "").strip()

if CONFIG_KB_PARENT_ROOT:
    KB_PARENT_ROOT = Path(CONFIG_KB_PARENT_ROOT).resolve()
else:
    # expected layout: unified_bench (KernelBench는 third_party/ 아래)
    KB_PARENT_ROOT = (SCRIPT_DIR / "third_party").resolve()


def task_id(path: Path) -> int:
    m = re.match(r"(\d+)_", path.name)
    return int(m.group(1)) if m else 10**9


rows = []

for level in ["level1", "level2", "level3"]:
    level_dir = KB_PARENT_ROOT / "KernelBench" / level

    if not level_dir.exists():
        print(f"[WARN] Missing level directory: {level_dir}")
        continue

    tasks = sorted(level_dir.glob("*.py"), key=task_id)

    for p in tasks:
        rel = p.relative_to(KB_PARENT_ROOT)
        rows.append({
            "level": level,
            "task_id": task_id(p),
            "task_path": str(rel),
            "task_name": p.name,
        })

txt_path = SCRIPT_DIR / "kernelbench_all_levels.txt"
csv_path = SCRIPT_DIR / "kernelbench_all_levels.csv"

txt_path.write_text("\n".join(r["task_path"] for r in rows) + "\n")

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["level", "task_id", "task_path", "task_name"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"CudaForge root: {KB_PARENT_ROOT}")
print(f"Saved: {txt_path}")
print(f"Saved: {csv_path}")
print(f"Total tasks: {len(rows)}")

for level in ["level1", "level2", "level3"]:
    print(level, sum(1 for r in rows if r["level"] == level))
