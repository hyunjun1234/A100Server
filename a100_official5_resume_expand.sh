#!/usr/bin/env bash
# a100_official5_resume_expand.sh
#
# 기존 A100 official5 결과를 삭제하지 않고:
#   1) campaign별 결과를 분리 집계
#   2) 빠진 task만 재개
#   3) round3 repeat3 / 전체 250 task round1로 확장
#   4) feedback archive 생성
#
# 기본 경로:
#   /home/jun/unified_bench
#
# 사용법:
#   cd /home/jun/unified_bench
#   chmod +x a100_official5_resume_expand.sh
#
#   ./a100_official5_resume_expand.sh audit
#   ./a100_official5_resume_expand.sh collect_round1
#   ./a100_official5_resume_expand.sh collect_round3
#   ./a100_official5_resume_expand.sh resume_round1
#   ./a100_official5_resume_expand.sh resume_round3
#   ./a100_official5_resume_expand.sh expand_rep3
#   ./a100_official5_resume_expand.sh expand_all250_r1
#   ./a100_official5_resume_expand.sh pack
#
# 주의:
# - 기존 runs/를 rm -rf하지 않습니다.
# - K-Search는 제외합니다.
# - official5: cudaforge autokernel cuda_l1 autotriton drkernel

set -euo pipefail

ROOT="${ROOT:-/home/jun/unified_bench}"
MODE="${1:-help}"

SYSTEMS=(cudaforge autokernel cuda_l1 autotriton drkernel)
SYSTEMS_STR="${SYSTEMS[*]}"
TEMP="${TEMP:-0.2}"
TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"

cd "$ROOT"

log() {
  echo
  echo "=========================================="
  echo "$*"
  echo "=========================================="
}

source_env() {
  set +e
  if [ -f use_a100_cuda126_env.sh ]; then
    # shellcheck disable=SC1091
    source use_a100_cuda126_env.sh
  elif [ -f scripts/use_a100_cuda_env.sh ]; then
    # shellcheck disable=SC1091
    source scripts/use_a100_cuda_env.sh
  fi
  set -e
}

ensure_layout() {
  mkdir -p logs runs results/campaigns resume_lists feedback
}

build_all250() {
  log "Building KernelBench all250 task list"

  python3 - <<'PY'
from pathlib import Path
import csv
import re

root = Path.cwd()

candidates = [
    root / "third_party" / "KernelBench" / "KernelBench",
    root / "third_party" / "KernelBench",
    root / "KernelBench",
]

kb = None
for candidate in candidates:
    if (candidate / "level1").exists():
        kb = candidate
        break

if kb is None:
    raise SystemExit("KernelBench level1 directory not found")

def task_id(path):
    match = re.match(r"(\d+)_", path.name)
    return int(match.group(1)) if match else 10**9

rows = []
for level in ["level1", "level2", "level3"]:
    directory = kb / level
    if not directory.exists():
        continue
    for path in sorted(directory.glob("*.py"), key=task_id):
        rows.append({
            "level": level,
            "task_id": task_id(path),
            "task_path": str(path.relative_to(root)),
            "task_name": path.name,
        })

Path("kernelbench_all250.txt").write_text(
    "\n".join(row["task_path"] for row in rows) + "\n"
)

with Path("kernelbench_all250.csv").open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["level", "task_id", "task_path", "task_name"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("KernelBench root:", kb)
print("all tasks:", len(rows))
for level in ["level1", "level2", "level3"]:
    print(level, sum(row["level"] == level for row in rows))
PY
}

write_campaign_tool() {
  cat > /tmp/a100_campaign_tool.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


SYSTEMS = ["cudaforge", "autokernel", "cuda_l1", "autotriton", "drkernel"]


def read_summary(path: Path):
    try:
        obj = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return []

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get("results") or obj.get("tasks") or [obj]
    return []


def parse_repeat(path: Path):
    for part in path.parts:
        match = re.search(r"_rep(\d+)$", part)
        if match:
            return int(match.group(1))
    return 0


def norm_task(task: str):
    return str(task).replace("\\", "/")


def level_of(task: str):
    for part in Path(task).parts:
        if re.fullmatch(r"level[123]", part):
            return part
    return "unknown"


def bool_value(value):
    return str(value).lower() in {"1", "true", "yes"}


def float_value(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def geometric_mean(values):
    values = [value for value in values if value > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def matching_run_dirs(runs_dir, system, rounds, repeat, temp):
    regex = re.compile(
        rf"^{re.escape(system)}.*_round{rounds}_repeat{repeat}_temp{re.escape(str(temp))}$"
    )
    return [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and regex.match(path.name)
    ] if runs_dir.exists() else []


def collect(root, task_list, rounds, repeat, temp, output_dir):
    tasks = [
        norm_task(line.strip())
        for line in task_list.read_text().splitlines()
        if line.strip()
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"

    rows = []
    missing = []
    newest = {}

    for system in SYSTEMS:
        dirs = matching_run_dirs(runs_dir, system, rounds, repeat, temp)

        for run_dir in dirs:
            for summary_path in run_dir.rglob("final_eval/summary.json"):
                rep = parse_repeat(summary_path)
                mtime = summary_path.stat().st_mtime

                for item in read_summary(summary_path):
                    task = norm_task(item.get("task", ""))
                    root_str = norm_task(str(root))
                    if task.startswith(root_str + "/"):
                        task = task[len(root_str) + 1:]
                    if not task:
                        continue
                    key = (system, task, rep)
                    record = {
                        "system": system,
                        "task": task,
                        "level": level_of(task),
                        "repeat": rep,
                        "rounds": rounds,
                        "repeat_budget": repeat,
                        "temperature": temp,
                        "run_dir": str(run_dir),
                        "summary_path": str(summary_path),
                        "n_candidates": int(item.get("n_candidates", 0) or 0),
                        "n_compiled": int(item.get("n_compiled", 0) or 0),
                        "n_correct": int(item.get("n_correct", 0) or 0),
                        "runnable_rate": float_value(item.get("runnable_rate")),
                        "correct_rate": float_value(item.get("correct_rate")),
                        "pass_at_1": float_value(item.get("pass@1")),
                        "fast_at_1": float_value(item.get("fast_1")),
                        "best_speedup": float_value(
                            item.get("best_score", item.get("geomean_speedup", 0))
                        ),
                        "best_runnable": bool_value(item.get("best_runnable")),
                        "error": item.get("error", ""),
                        "_mtime": mtime,
                    }
                    if key not in newest or mtime > newest[key]["_mtime"]:
                        newest[key] = record

        for task in tasks:
            for rep in range(repeat):
                key = (system, task, rep)
                if key in newest:
                    rows.append(newest[key])
                else:
                    missing.append({
                        "system": system,
                        "task": task,
                        "level": level_of(task),
                        "repeat": rep,
                        "rounds": rounds,
                        "repeat_budget": repeat,
                        "temperature": temp,
                    })

    for row in rows:
        row.pop("_mtime", None)

    repeat_csv = output_dir / "per_repeat.csv"
    fields = [
        "system", "task", "level", "repeat", "rounds", "repeat_budget",
        "temperature", "run_dir", "summary_path", "n_candidates",
        "n_compiled", "n_correct", "runnable_rate", "correct_rate",
        "pass_at_1", "fast_at_1", "best_speedup", "best_runnable", "error",
    ]
    with repeat_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    missing_csv = output_dir / "missing.csv"
    with missing_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "system", "task", "level", "repeat", "rounds",
                "repeat_budget", "temperature",
            ],
        )
        writer.writeheader()
        writer.writerows(missing)

    by_task = defaultdict(list)
    for row in rows:
        by_task[(row["system"], row["task"], row["level"])].append(row)

    task_rows = []
    for (system, task, level), values in sorted(by_task.items()):
        speeds = [
            row["best_speedup"]
            for row in values
            if row["correct_rate"] > 0 and row["best_speedup"] > 0
        ]
        task_rows.append({
            "system": system,
            "task": task,
            "level": level,
            "repeats_completed": len(values),
            "repeats_expected": repeat,
            "repeat_compile_rate": sum(row["runnable_rate"] > 0 for row in values) / repeat,
            "repeat_correct_rate": sum(row["correct_rate"] > 0 for row in values) / repeat,
            "any_correct": int(any(row["correct_rate"] > 0 for row in values)),
            "all_correct": int(
                len(values) == repeat and all(row["correct_rate"] > 0 for row in values)
            ),
            "pass_at_1_mean": sum(row["pass_at_1"] for row in values) / repeat,
            "fast_at_1_mean": sum(row["fast_at_1"] for row in values) / repeat,
            "best_speedup": max(speeds) if speeds else 0.0,
            "speedup_gmean_correct_repeats": geometric_mean(speeds),
        })

    task_csv = output_dir / "per_task.csv"
    if task_rows:
        with task_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(task_rows[0].keys()))
            writer.writeheader()
            writer.writerows(task_rows)
    else:
        task_csv.write_text("system\n")

    grouped = defaultdict(list)
    for row in task_rows:
        grouped[(row["system"], row["level"])].append(row)

    summary_rows = []
    for (system, level), values in sorted(grouped.items()):
        correct_speeds = [
            row["best_speedup"]
            for row in values
            if row["any_correct"] and row["best_speedup"] > 0
        ]
        summary_rows.append({
            "system": system,
            "level": level,
            "n_tasks_expected": sum(1 for task in tasks if level_of(task) == level),
            "n_tasks_seen": len(values),
            "tasks_complete_rate": sum(
                row["repeats_completed"] == repeat for row in values
            ) / max(sum(1 for task in tasks if level_of(task) == level), 1),
            "task_any_correct_rate": sum(row["any_correct"] for row in values)
            / max(sum(1 for task in tasks if level_of(task) == level), 1),
            "task_all_correct_rate": sum(row["all_correct"] for row in values)
            / max(sum(1 for task in tasks if level_of(task) == level), 1),
            "pass_at_1_mean": sum(row["pass_at_1_mean"] for row in values)
            / max(sum(1 for task in tasks if level_of(task) == level), 1),
            "fast_at_1_mean": sum(row["fast_at_1_mean"] for row in values)
            / max(sum(1 for task in tasks if level_of(task) == level), 1),
            "n_correct_speed_tasks": len(correct_speeds),
            "speedup_gmean_correct_tasks": geometric_mean(correct_speeds),
            "speedup_median_correct_tasks": (
                sorted(correct_speeds)[len(correct_speeds)//2]
                if correct_speeds else 0.0
            ),
            "speedup_max_correct_tasks": max(correct_speeds) if correct_speeds else 0.0,
        })

    summary_csv = output_dir / "summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_csv.write_text("system\n")

    # Missing task lists by system
    manifest = task_list.with_suffix(".csv")
    manifest_rows = []
    manifest_fields = []
    if manifest.exists():
        with manifest.open() as file:
            reader = csv.DictReader(file)
            manifest_rows = list(reader)
            manifest_fields = reader.fieldnames or []

    missing_by_system = defaultdict(set)
    for row in missing:
        missing_by_system[row["system"]].add(row["task"])

    resume_dir = output_dir / "resume_lists"
    resume_dir.mkdir(exist_ok=True)

    for system in SYSTEMS:
        system_tasks = [task for task in tasks if task in missing_by_system[system]]
        txt = resume_dir / f"{system}.txt"
        txt.write_text("\n".join(system_tasks) + ("\n" if system_tasks else ""))

        if manifest_rows and manifest_fields:
            selected = [
                row for row in manifest_rows
                if norm_task(row.get("task_path", "")) in set(system_tasks)
            ]
            with (resume_dir / f"{system}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=manifest_fields)
                writer.writeheader()
                writer.writerows(selected)

    report = {
        "task_list": str(task_list),
        "rounds": rounds,
        "repeat": repeat,
        "temperature": temp,
        "expected_repeat_rows": len(tasks) * len(SYSTEMS) * repeat,
        "found_repeat_rows": len(rows),
        "missing_repeat_rows": len(missing),
        "missing_by_system": {
            system: len(missing_by_system[system]) for system in SYSTEMS
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("summary:", summary_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-list", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--temperature", default="0.2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    collect(
        Path(args.root).resolve(),
        Path(args.task_list).resolve(),
        args.rounds,
        args.repeat,
        args.temperature,
        Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
PY
}

collect_campaign() {
  local task_list="$1"
  local rounds="$2"
  local repeat="$3"
  local name="$4"

  ensure_layout
  write_campaign_tool

  python3 /tmp/a100_campaign_tool.py \
    --root "$ROOT" \
    --task-list "$ROOT/$task_list" \
    --rounds "$rounds" \
    --repeat "$repeat" \
    --temperature "$TEMP" \
    --output "$ROOT/results/campaigns/$name"
}

audit() {
  log "Campaign audit"

  collect_campaign kernelbench_subset50.txt 1 1 subset50_round1
  collect_campaign kernelbench_subset50.txt 3 1 subset50_round3

  if [ -f kernelbench_all250.txt ]; then
    collect_campaign kernelbench_all250.txt 1 1 all250_round1
  fi

  if [ -f kernelbench_subset50.txt ]; then
    collect_campaign kernelbench_subset50.txt 3 3 subset50_round3_repeat3
  fi

  echo
  find results/campaigns -name audit.json -print -exec cat {} \;
}

resume_campaign() {
  local task_list="$1"
  local rounds="$2"
  local repeat="$3"
  local name="$4"
  local output="$ROOT/results/campaigns/$name"

  collect_campaign "$task_list" "$rounds" "$repeat" "$name"

  source_env
  ensure_layout

  nohup bash -c "
    set -o pipefail
    cd '$ROOT'
    if [ -f use_a100_cuda126_env.sh ]; then source use_a100_cuda126_env.sh; fi
    if [ -f scripts/use_a100_cuda_env.sh ]; then source scripts/use_a100_cuda_env.sh; fi

    for system in $SYSTEMS_STR; do
      list='$output/resume_lists/'\$system'.txt'
      count=\$(grep -c . \"\$list\" 2>/dev/null || true)
      count=\${count:-0}
      echo \"===== \$system missing tasks: \$count =====\"
      if [ \"\$count\" -eq 0 ]; then
        continue
      fi

      CUDA_VISIBLE_DEVICES=0,1 \
      SERVER_CUDA_VISIBLE_DEVICES=1 \
      BENCH_CUDA_VISIBLE_DEVICES=0 \
      TASK_LIST=\"\${list#$ROOT/}\" \
      SYSTEMS=\"\$system\" \
      ./run_all_with_server.sh '$rounds' '$repeat' '$TEMP' '$TASK_TIMEOUT'
    done

    python3 /tmp/a100_campaign_tool.py \
      --root '$ROOT' \
      --task-list '$ROOT/$task_list' \
      --rounds '$rounds' \
      --repeat '$repeat' \
      --temperature '$TEMP' \
      --output '$output'
  " > "logs/${name}_resume.out" 2>&1 &

  echo "Started resume: $name"
  echo "tail -f $ROOT/logs/${name}_resume.out"
}

expand_rep3() {
  log "Expanding subset50 to round3 repeat3"

  collect_campaign kernelbench_subset50.txt 3 3 subset50_round3_repeat3
  resume_campaign kernelbench_subset50.txt 3 3 subset50_round3_repeat3
}

expand_all250_r1() {
  build_all250
  log "Expanding to all250 round1 repeat1"

  collect_campaign kernelbench_all250.txt 1 1 all250_round1
  resume_campaign kernelbench_all250.txt 1 1 all250_round1
}

pack() {
  audit

  log "Packing corrected campaign results"

  local stamp bundle archive
  stamp=$(date +%Y%m%d_%H%M%S)
  bundle="a100_official5_followup_${stamp}"
  archive="${bundle}.tar.gz"

  rm -rf "$bundle"
  mkdir -p "$bundle"/{results,logs,systems,config}

  cp -a results/campaigns "$bundle/results/" 2>/dev/null || true
  cp results/unified_summary*.csv "$bundle/results/" 2>/dev/null || true
  cp results/unified_per_task.csv "$bundle/results/" 2>/dev/null || true
  cp results/evaluation_sheet.json results/repo_lock.json "$bundle/results/" 2>/dev/null || true

  cp logs/*round*.out logs/*resume*.out "$bundle/logs/" 2>/dev/null || true
  cp systems/{cudaforge,autokernel,cuda_l1,autotriton,drkernel}.env "$bundle/systems/" 2>/dev/null || true
  cp benchmark_config.env use_a100_cuda126_env.sh "$bundle/config/" 2>/dev/null || true

  # 실패 분석에 필요한 작은 파일만 포함
  find runs -type f \
    \( -name "summary.json" \
       -o -name "verdicts.json" \
       -o -name "verdict_*.json" \
       -o -name "task_meta.json" \
       -o -name "*raw_reply*.txt" \
       -o -name "candidate_*.py" \
       -o -name "round*_kernel.py" \) \
    -size -2M \
    -exec cp --parents {} "$bundle/" \; 2>/dev/null || true

  tar -czf "$archive" "$bundle"
  rm -rf "$bundle"

  echo
  echo "Upload this file:"
  echo "$ROOT/$archive"
}

status() {
  echo "===== processes ====="
  ps aux | grep -E \
    "run_all_with_server|run_system|final_eval|eval_worker|hf_openai_server|resume" \
    | grep -v grep || true

  echo
  echo "===== recent resume logs ====="
  for file in logs/*resume*.out; do
    [ -f "$file" ] || continue
    echo "----- $file -----"
    tail -40 "$file"
  done

  echo
  nvidia-smi || true
}

help_message() {
  cat <<EOF
Usage:
  ./a100_official5_resume_expand.sh audit
  ./a100_official5_resume_expand.sh collect_round1
  ./a100_official5_resume_expand.sh collect_round3
  ./a100_official5_resume_expand.sh resume_round1
  ./a100_official5_resume_expand.sh resume_round3
  ./a100_official5_resume_expand.sh expand_rep3
  ./a100_official5_resume_expand.sh expand_all250_r1
  ./a100_official5_resume_expand.sh status
  ./a100_official5_resume_expand.sh pack

Recommended:
  1. audit
  2. resume_round3
  3. expand_rep3
  4. expand_all250_r1
  5. pack

Upload:
  /home/jun/unified_bench/a100_official5_followup_*.tar.gz
EOF
}

ensure_layout

case "$MODE" in
  audit)
    audit
    ;;
  collect_round1)
    collect_campaign kernelbench_subset50.txt 1 1 subset50_round1
    ;;
  collect_round3)
    collect_campaign kernelbench_subset50.txt 3 1 subset50_round3
    ;;
  resume_round1)
    resume_campaign kernelbench_subset50.txt 1 1 subset50_round1
    ;;
  resume_round3)
    resume_campaign kernelbench_subset50.txt 3 1 subset50_round3
    ;;
  expand_rep3)
    expand_rep3
    ;;
  expand_all250_r1)
    expand_all250_r1
    ;;
  status)
    status
    ;;
  pack)
    pack
    ;;
  help|--help|-h)
    help_message
    ;;
  *)
    echo "Unknown mode: $MODE"
    help_message
    exit 1
    ;;
esac
