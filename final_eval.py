"""final_eval.py — task 하나의 모든 후보를 통합 검증기로 재평가하고
ibm 규약의 summary.json을 task work dir에 기록 (resume 감지에 사용).

사용: python3 final_eval.py --task <task_path> --cand_dir <dir> --glob "round*_kernel.py" \
        --task_work_dir <dir>
설정은 benchmark_config.env 의 환경변수에서 읽는다 (run_system.sh가 export).
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def env(k, d):
    return os.environ.get(k, d)


def input_seed(root_seed: int, task_path: str) -> int:
    h = hashlib.sha256(f"{root_seed}:{task_path}".encode()).hexdigest()
    return int(h[:8], 16)


def pass_at_k(n, c, k):
    if n == 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand_dir", required=True)
    ap.add_argument("--glob", default="round*_kernel.py")
    ap.add_argument("--task_work_dir", required=True)
    a = ap.parse_args()

    cand_dir, wdir = Path(a.cand_dir), Path(a.task_work_dir)
    wdir.mkdir(parents=True, exist_ok=True)
    cands = sorted(cand_dir.glob(a.glob))
    seed_base = input_seed(int(env("ROOT_SEED", "20260611")), a.task)
    timeout = int(env("EVAL_TIMEOUT_SEC", "480"))

    verdicts = []
    for i, cp in enumerate(cands):
        out = wdir / f"verdict_{i:03d}.json"
        job = {"ref_path": a.task, "cand_path": str(cp),
               "trials": int(env("NUM_CORRECTNESS_TRIALS", "5")),
               "atol": float(env("ATOL", "1e-2")), "rtol": float(env("RTOL", "1e-2")),
               "warmup": int(env("EVAL_WARMUP", "10")),
               "timing_iters": int(env("EVAL_TIMING_ITERS", "100")),
               "input_seed_base": seed_base, "out_path": str(out)}
        jp = wdir / f"job_{i:03d}.json"
        jp.write_text(json.dumps(job))
        t0 = time.time()
        try:
            subprocess.run([sys.executable, str(SCRIPT_DIR / "eval_worker.py"), str(jp)],
                           timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            pass
        v = (json.loads(out.read_text()) if out.exists()
             else {"compiled": False, "correct": False, "latency_ms": None,
                   "ref_latency_ms": None, "speedup": None, "error": "timeout/crash"})
        v.update({"candidate": str(cp), "eval_sec": round(time.time() - t0, 1)})
        verdicts.append(v)
        print(f"[{i+1}/{len(cands)}] compiled={v['compiled']} correct={v['correct']} "
              f"speedup={v['speedup']}")

    n = len(verdicts)
    correct = sum(1 for v in verdicts if v["correct"])
    runnable = sum(1 for v in verdicts if v["compiled"])
    sp = [v["speedup"] for v in verdicts if v["correct"] and v["speedup"]]
    best = max(sp) if sp else 0.0
    summary = [{
        "task": a.task,
        "task_dir": str(wdir),
        "n_candidates": n,
        "runnable_rate": runnable / n if n else 0.0,
        "correct_rate": correct / n if n else 0.0,
        "pass@1": pass_at_k(n, correct, 1),
        "fast_1": (sum(1 for v in verdicts if v["correct"] and (v["speedup"] or 0) > 1.0)
                   / n if n else 0.0),
        "best_score": best,                 # ibm collect_results 호환 필드
        "best_runnable": runnable > 0,      # ibm collect_results 호환 필드
        "geomean_speedup": (statistics.geometric_mean(sp) if sp else 0.0),
    }]
    (wdir / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    (wdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {wdir/'summary.json'}")


if __name__ == "__main__":
    main()
