r"""collect_results.py — runs/ 전체를 훑어 system×level 집계 (ibm collect 확장판).

runs/<EXP_NAME>/<task>_rep<N>/final_eval/summary.json 을 수집.
EXP_NAME 규약: <system>_<model>_<tasklist>_round..._repeat..._temp...

출력: results/unified_summary.csv (system×level), results/unified_per_task.csv,
      results/unified_table.tex (booktabs — 논문 experi.tex에 \input 가능)
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import pathlib


def read_manifest(path):
    with open(path) as f:
        return {r["task_path"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--manifest", default="kernelbench_subset50.csv")
    ap.add_argument("--out", default="results/unified_summary.csv")
    a = ap.parse_args()

    manifest = read_manifest(a.manifest)
    tasklist_tag = pathlib.Path(a.manifest).stem  # 이 실행의 task list로 한정 (tiny5/subset50 혼입 방지)
    per_task_rows = []
    for sj in Path(a.runs_dir).rglob("final_eval/summary.json"):
        exp = sj.relative_to(a.runs_dir).parts[0]
        if tasklist_tag not in exp:
            continue
        known_systems = [
            "baseline_loop",
            "cuda_agent",
            "autotriton",
            "autokernel",
            "cudaforge",
            "cuda_l1",
            "drkernel",
            "ksearch",
            "geak",
        ]
        system = next((x for x in known_systems if exp.startswith(x + "_")), exp.split("_")[0])
        for item in json.loads(sj.read_text()):
            task = item.get("task", "")
            rel = task.split("third_party/")[-1]
            rel = "third_party/" + rel if "third_party/" not in task else task
            level = next((m["level"] for p, m in manifest.items() if p in task), "?")
            per_task_rows.append({"system": system, "level": level, "task": task,
                                  **{k: item.get(k) for k in
                                     ("n_candidates", "runnable_rate", "correct_rate",
                                      "pass@1", "fast_1", "best_score",
                                      "geomean_speedup")}})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name("unified_per_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()) if per_task_rows
                           else ["system"])
        w.writeheader(); w.writerows(per_task_rows)

    # system × level 평균
    bucket = defaultdict(list)
    for r in per_task_rows:
        bucket[(r["system"], r["level"])].append(r)
    summary = []
    for (system, level), rows in sorted(bucket.items()):
        def mean(k, only_pos=False):
            vals = [r[k] for r in rows if r[k] is not None]
            if only_pos:
                vals = [v for v in vals if v]
            return round(statistics.mean(vals), 3) if vals else 0.0
        summary.append({"system": system, "level": level, "n_tasks": len(rows),
                        "runnable": mean("runnable_rate"), "correct": mean("correct_rate"),
                        "pass@1": mean("pass@1"), "fast_1": mean("fast_1"),
                        "best_speedup": mean("best_score"),
                        "geomean_speedup": mean("geomean_speedup", only_pos=True)})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["system"])
        w.writeheader(); w.writerows(summary)

    # booktabs LaTeX
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{통합 평가 환경에서의 공개 시스템 비교 (동일 GPU, 동일 검증기, "
           r"동일 budget; 환경 명세는 evaluation sheet 참조)}",
           r"\label{tab:unified_eval}",
           r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"System & Level & Runnable & Correct & pass@1 & fast$_1$ & Speedup(gm) \\",
           r"\midrule"]
    for r in summary:
        gm = f"{r['geomean_speedup']:.2f}" if r["geomean_speedup"] else "--"
        tex.append(f"{r['system'].replace('_', chr(92)+'_')} & "
                   f"{r['level'].replace('level','L')} & {r['runnable']:.2f} & "
                   f"{r['correct']:.2f} & {r['pass@1']:.2f} & {r['fast_1']:.2f} & {gm} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.with_name("unified_table.tex").write_text("\n".join(tex))

    print(f"per-task -> {out.with_name('unified_per_task.csv')}")
    print(f"summary  -> {out}")
    print(f"latex    -> {out.with_name('unified_table.tex')}")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
