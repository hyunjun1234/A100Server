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

    # remove markdown remnants
    code = code.replace("```python", "").replace("```py", "").replace("```", "").strip()

    # remove leading prose
    bad_prefixes = [
        "Here's the optimized code:",
        "Here is the optimized code:",
        "Sure, here's the optimized code:",
        "Sure, here is the optimized code:",
        "The optimized code is:",
        "Below is the optimized code:",
    ]
    lines = code.splitlines()
    while lines and lines[0].strip() in bad_prefixes:
        lines = lines[1:]
    code = "\n".join(lines).strip()

    # cut trailing explanations / examples
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

    # prefer fenced block that contains ModelNew
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for b in blocks:
        if "class ModelNew" in b:
            return clean_code(b)

    if blocks:
        return clean_code(blocks[0])

    # fallback: cut from first Python marker
    markers = ["import torch", "from torch", "import triton", "from triton", "class ModelNew"]
    starts = [text.find(m) for m in markers if text.find(m) != -1]
    if starts:
        return clean_code(text[min(starts):])

    return clean_code(text)


def build_prompt(task_src: str, feedback: str = "") -> str:
    prompt = f"""
You are GEAK, an autonomous GPU kernel optimization agent.

Your task is to optimize the following KernelBench PyTorch model.

Return exactly one valid Python source file.
Do not include Markdown.
Do not include explanations.
Do not include example usage.

Hard requirements:
- The file must define class ModelNew(torch.nn.Module).
- ModelNew.forward must have the same signature and output semantics as the reference Model.forward.
- Use CUDA/Triton/custom kernels if useful.
- It is allowed to call torch operations if that is the safest correct implementation.
- Never move tensors to CPU.
- All outputs must stay on the same CUDA device as inputs.

Reference task file:

```python
{task_src}
```
"""
    if feedback:
        prompt += "\nPrevious evaluation feedback:\n" + feedback + "\n"

    return prompt


def call_openai_server(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    url = base_url + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are GEAK, a GPU kernel optimization agent. "
                    "Return only valid Python source code. "
                    "The source must define class ModelNew(torch.nn.Module). "
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


def eval_once(task: Path, cand: Path, io_dir: Path, round_id: int) -> str:
    # optional inner feedback; failures are not fatal
    root = Path(__file__).resolve().parent.parent
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
        "input_seed_base": 100 + round_id,
        "out_path": str(out),
    }, indent=2))

    try:
        subprocess.run(
            [sys.executable, str(root / "eval_worker.py"), str(job)],
            timeout=240,
            capture_output=True,
            text=True,
        )
        if out.exists():
            v = json.loads(out.read_text())
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
    ap.add_argument("--multi-turn", action="store_true")
    args = ap.parse_args()

    task = Path(args.task).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    io_dir = out_dir / "geak_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    task_src = task.read_text(errors="ignore")
    model = os.environ.get("EVAL_MODEL") or os.environ.get("MODEL_ALIAS") or "qwen14b"
    max_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))

    feedback = ""
    generated = []

    for r in range(max(1, args.rounds)):
        prompt = build_prompt(task_src, feedback)
        raw = call_openai_server(
            prompt=prompt,
            model=model,
            temperature=args.temperature,
            max_tokens=max_tokens,
        )

        raw_path = io_dir / f"round{r:03d}_raw_reply.txt"
        raw_path.write_text(raw)

        code = extract_code(raw)

        cand = out_dir / f"candidate_{r:04d}.py"
        cand.write_text(code)
        generated.append(cand)

        first = code.splitlines()[0] if code.splitlines() else "<empty>"
        print(f"[geak-adapter] wrote {cand}")
        print(f"[geak-adapter] raw reply {raw_path}")
        print(f"[geak-adapter] code starts with: {first}")

        # GEAK-like iterative feedback for rounds > 1
        feedback = eval_once(task, cand, io_dir, r)
        print(f"[geak-adapter] feedback: {feedback}")

        if not args.multi_turn:
            # for smoke, one candidate is enough when rounds=1
            continue

    if not generated:
        print("[geak-adapter] ERROR: no candidates generated")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
