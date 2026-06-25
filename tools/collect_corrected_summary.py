#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from collections import defaultdict

src = Path("results/unified_per_task.csv")
if not src.exists():
    raise SystemExit("results/unified_per_task.csv not found. Run collect_results.py first.")

rows = list(csv.DictReader(src.open()))
groups = defaultdict(list)

for r in rows:
    # ksearch/cuda_agent/geak 제외
    if r.get("system") in {"ksearch", "cuda_agent", "geak"}:
        continue
    groups[(r["system"], r["level"])].append(r)

out_rows = []
for (system, level), rs in sorted(groups.items()):
    n = len(rs)
    if n == 0:
        continue
    runnable_vals = [float(r.get("runnable_rate") or 0.0) for r in rs]
    correct_vals = [float(r.get("correct_rate") or 0.0) for r in rs]
    pass_vals = [float(r.get("pass@1") or 0.0) for r in rs]
    fast_vals = [float(r.get("fast_1") or 0.0) for r in rs]

    speeds = []
    for r in rs:
        c = float(r.get("correct_rate") or 0.0)
        s = float(r.get("best_score") or 0.0)
        if c > 0 and s > 0:
            speeds.append(s)

    if speeds:
        sorted_s = sorted(speeds)
        gmean = math.exp(sum(math.log(max(s, 1e-12)) for s in speeds) / len(speeds))
        mean = sum(speeds) / len(speeds)
        median = sorted_s[len(sorted_s)//2] if len(sorted_s) % 2 else (sorted_s[len(sorted_s)//2 - 1] + sorted_s[len(sorted_s)//2]) / 2
        mx = max(speeds)
    else:
        gmean = mean = median = mx = 0.0

    out_rows.append({
        "system": system,
        "level": level,
        "n_tasks": n,
        "runnable": round(sum(runnable_vals)/n, 3),
        "correct": round(sum(correct_vals)/n, 3),
        "pass@1": round(sum(pass_vals)/n, 3),
        "fast_1": round(sum(fast_vals)/n, 3),
        "n_correct_speed": len(speeds),
        "speedup_gmean_correct": round(gmean, 3),
        "speedup_mean_correct": round(mean, 3),
        "speedup_median_correct": round(median, 3),
        "speedup_max_correct": round(mx, 3),
    })

dst = Path("results/unified_summary_corrected.csv")
with dst.open("w", newline="") as f:
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "system","level","n_tasks","runnable","correct","pass@1","fast_1",
        "n_correct_speed","speedup_gmean_correct","speedup_mean_correct",
        "speedup_median_correct","speedup_max_correct"
    ]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

print("Wrote", dst)
for r in out_rows:
    print(r)
