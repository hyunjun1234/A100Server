#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path.cwd()
    repo = root / "third_party" / "CUDA-Agent"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    candidates = [
        repo / "agent_workdir" / "model_new.py",
        repo / "agent_workdir" / "model.py",
    ]

    src = None
    for p in candidates:
        if p.exists():
            src = p
            break

    if src is None:
        print("[cuda-agent-adapter] WARNING: no model_new.py/model.py found")
        return 0

    dst = out / "candidate_0000.py"
    shutil.copy2(src, dst)

    print("[cuda-agent-adapter] copied:", src, "->", dst)
    print("[cuda-agent-adapter] NOTE: this is a public demo artifact, not necessarily matched to KernelBench task:", args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
