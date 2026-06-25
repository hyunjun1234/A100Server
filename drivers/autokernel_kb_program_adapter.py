#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests


def clean_code(code: str) -> str:
    code = (code or "").strip()
    code = code.replace("```python", "").replace("```py", "").replace("```", "").strip()

    bad_prefixes = [
        "Here's the optimized code:",
        "Here is the optimized code:",
        "Sure, here's the optimized code:",
        "Sure, here is the optimized code:",
        "Below is the optimized code:",
        "The optimized code is:",
    ]
    lines = code.splitlines()
    while lines and lines[0].strip() in bad_prefixes:
        lines = lines[1:]
    code = "\n".join(lines).strip()

    cut_markers = [
        'if __name__ == "__main__":',
        "if __name__ == '__main__':",
        "# Example usage",
        "### Explanation",
        "## Explanation",
        "Explanation:",
        "This code",
        "The code",
    ]
    for m in cut_markers:
        idx = code.find(m)
        if idx != -1:
            code = code[:idx].strip()

    return code + "\n"


def extract_code(text: str) -> str:
    text = text or ""

    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for b in blocks:
        if "class ModelNew" in b:
            return clean_code(b)

    if blocks:
        return clean_code(blocks[0])

    markers = ["import torch", "from torch", "import triton", "from triton", "class ModelNew"]
    starts = [text.find(m) for m in markers if text.find(m) != -1]
    if starts:
        return clean_code(text[min(starts):])

    return clean_code(text)


def find_autokernel_repo(root: Path) -> Path:
    for c in [root / "third_party" / "autokernel", root / "third_party" / "AutoKernel"]:
        if c.exists():
            return c
    return root / "third_party" / "autokernel"


def read_autokernel_program(root: Path) -> str:
    repo = find_autokernel_repo(root)
    paths = [
        repo / "kernelbench" / "program_kb.md",
        repo / "program.md",
        repo / "README.md",
    ]
    for p in paths:
        if p.exists():
            return f"# Source: {p}\n\n" + p.read_text(errors="ignore")[:20000]
    return "No AutoKernel program_kb.md found. Use standard KernelBench ModelNew optimization rules."


def build_prompt(task_src: str, ak_program: str, feedback: str = "") -> str:
    return f"""
You are running the AutoKernel KernelBench agent protocol.

Use the following official AutoKernel instructions as your policy.

<autokernel_instructions>
{ak_program}
</autokernel_instructions>

Now solve this KernelBench task.

Return exactly one valid Python source file.
Do not include Markdown.
Do not include explanations.
Do not include example usage.

Hard requirements:
- The file must define class ModelNew(torch.nn.Module).
- ModelNew.forward must have the same signature and output semantics as reference Model.forward.
- Never move tensors to CPU.
- All outputs must stay on the same CUDA device as inputs.
- Prefer correctness first. A correct PyTorch fallback is allowed if custom CUDA/Triton is risky.

Reference task file:

```python
{task_src}
```

Previous feedback:
{feedback}
"""


def call_server(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    url = base_url + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are AutoKernel's KernelBench coding agent. "
                    "Return only valid Python source code defining class ModelNew. "
                    "No markdown. No explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": float(os.environ.get("TOP_P", "0.95")),
        "max_tokens": max_tokens,
    }

    r = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY"),
        },
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def eval_once(root: Path, task: Path, cand: Path, io_dir: Path, round_id: int) -> str:
    job = io_dir / f"inner_job_{round_id:03d}.json"
    out = io_dir / f"inner_verdict_{round_id:03d}.json"

    job.write_text(json.dumps({
        "ref_path": str(task),
        "cand_path": str(cand),
        "trials": 1,
        "atol": float(os.environ.get("ATOL", "1e-2")),
        "rtol": float(os.environ.get("RTOL", "1e-2")),
        "warmup": 2,
        "timing_iters": 10,
        "input_seed_base": 777 + round_id,
        "out_path": str(out),
    }, indent=2))

    try:
        subprocess.run(
            [sys.executable, str(root / "eval_worker.py"), str(job)],
            cwd=str(root),
            timeout=240,
            capture_output=True,
            text=True,
        )
        if out.exists():
            v = json.loads(out.read_text(errors="ignore"))
            return (
                f"compiled={v.get('compiled')} correct={v.get('correct')} "
                f"speedup={v.get('speedup')} error={(v.get('error') or '')[:1200]}"
            )
    except Exception as e:
        return f"inner evaluation failed: {e}"

    return "inner evaluation unavailable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    task = Path(args.task).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    io_dir = out_dir / "autokernel_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    task_src = task.read_text(errors="ignore")
    ak_program = read_autokernel_program(root)

    model = os.environ.get("EVAL_MODEL") or os.environ.get("MODEL_ALIAS") or "qwen14b"
    max_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

    feedback = ""
    n_rounds = max(1, args.rounds)
    for r in range(n_rounds):
        prompt = build_prompt(task_src, ak_program, feedback)
        raw = call_server(prompt, model=model, temperature=args.temperature, max_tokens=max_tokens)

        raw_path = io_dir / f"round{r:03d}_raw_reply.txt"
        raw_path.write_text(raw)

        code = extract_code(raw)
        cand = out_dir / f"candidate_{r:04d}.py"
        cand.write_text(code)

        first = code.splitlines()[0] if code.splitlines() else "<empty>"
        print(f"[autokernel-kb-adapter] wrote {cand}")
        print(f"[autokernel-kb-adapter] raw reply {raw_path}")
        print(f"[autokernel-kb-adapter] code starts with: {first}")

        feedback = eval_once(root, task, cand, io_dir, r)
        print(f"[autokernel-kb-adapter] feedback: {feedback}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
