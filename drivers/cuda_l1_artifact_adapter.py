#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def extract_code(x):
    if not isinstance(x, str):
        return None
    text = x.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    markers = ["import torch", "from torch", "class ModelNew", "import triton", "from triton"]
    starts = [text.find(mk) for mk in markers if text.find(mk) != -1]
    if starts:
        text = text[min(starts):].strip()
    if "class ModelNew" not in text:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-json", default="auto")
    args = ap.parse_args()

    root = Path.cwd()
    repo = root / "third_party" / "CUDA-L1"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    task_path = Path(args.task)
    m = re.match(r"(\d+)_", task_path.name)
    task_id = int(m.group(1)) if m else None
    ml = re.search(r"level(\d+)", str(task_path))
    level_id = int(ml.group(1)) if ml else None

    json_candidates = []
    if args.gpu_json != "auto":
        json_candidates.append(repo / "optimized_cuda_code" / args.gpu_json)
    else:
        preferred = [
            "optimized_cuda_code/h100.json",
            "optimized_cuda_code/codes/h100.json",
            "optimized_cuda_code/a100.json",
            "optimized_cuda_code/codes/a100.json",
            "optimized_cuda_code/l40.json",
            "optimized_cuda_code/codes/l40.json",
        ]
        json_candidates += [repo / x for x in preferred]
        json_candidates += sorted((repo / "optimized_cuda_code").glob("**/*.json"))

    seen_files = []
    best = None  # (jp, line_no, field, code)

    for jp in json_candidates:
        if not jp.exists() or jp in seen_files:
            continue
        seen_files.append(jp)
        print("[cuda-l1-adapter] reading", jp)
        # CUDA-L1 artifact files (a100.json/h100.json ...) are JSONL: one JSON object per line.
        for line_no, line in enumerate(jp.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print("[cuda-l1-adapter] WARN cannot read json line", line_no, jp, e)
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("level_id") != level_id or obj.get("task_id") != task_id:
                continue
            for field in ("custom_code", "cuda_graph_code", "cudnn_code"):
                code = extract_code(obj.get(field))
                if code:
                    best = (jp, line_no, field, code)
                    break
            if best is not None:
                break
        if best is not None:
            break

    if best is None:
        print("[cuda-l1-adapter] WARNING: no ModelNew code found for level",
              level_id, "task", task_id, "in CUDA-L1 json artifacts")
        return 0

    jp, line_no, field, code = best
    dst = out / "candidate_0000.py"
    dst.write_text(code)
    print("[cuda-l1-adapter] selected json:", jp)
    print("[cuda-l1-adapter] selected loc :", "line", line_no, "field", field)
    print("[cuda-l1-adapter] wrote       :", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
