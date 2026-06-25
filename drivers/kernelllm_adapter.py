#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import requests


FORMAT_EXAMPLE = r"""
Example output format:

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def example_kernel(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 256),)
        example_kernel[grid](x, y, n, BLOCK=256)
        return y
"""


def clean_code(text: str) -> str:
    text = (text or "").strip()
    blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in blocks:
        if "class ModelNew" in block:
            text = block
            break
    else:
        if blocks:
            text = blocks[0]
        else:
            markers = [
                "import torch",
                "from torch",
                "import triton",
                "from triton",
                "class ModelNew",
            ]
            starts = [text.find(marker) for marker in markers if text.find(marker) >= 0]
            if starts:
                text = text[min(starts):]

    text = text.replace("```python", "").replace("```py", "").replace("```", "").strip()

    cut_markers = [
        'if __name__ == "__main__":',
        "if __name__ == '__main__':",
        "# Example usage",
        "## Explanation",
        "### Explanation",
        "Explanation:",
    ]
    for marker in cut_markers:
        position = text.find(marker)
        if position >= 0:
            text = text[:position].rstrip()

    return text + "\n"


def call_server(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    base = os.environ.get(
        "OPENAI_BASE_URL",
        "http://127.0.0.1:8100/v1",
    ).rstrip("/")
    response = requests.post(
        base + "/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY"),
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are KernelLLM, specialized in translating PyTorch "
                        "modules into correct and efficient Triton kernels. "
                        "Return one executable Python file defining ModelNew. "
                        "Return code only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        timeout=900,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.97)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    task_path = Path(args.task).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    io_dir = out_dir / "kernelllm_io"
    io_dir.mkdir(parents=True, exist_ok=True)

    task_source = task_path.read_text(errors="ignore")
    model = (
        os.environ.get("EVAL_MODEL")
        or os.environ.get("SYSTEM_MODEL_ALIAS")
        or "kernelllm8b"
    )

    prompt = f"""
Translate the following KernelBench PyTorch module into a Triton implementation.

Requirements:
- Return one complete Python source file.
- Define class ModelNew(torch.nn.Module).
- Match the reference Model constructor and forward signature.
- Keep all tensors on CUDA.
- Do not use CPU or NumPy.
- Do not include prose or Markdown.
- Prefer a meaningful Triton kernel, not a direct torch fallback.
- Include all launch code in ModelNew.forward.

{FORMAT_EXAMPLE}

Reference module:

```python
{task_source}
```
"""

    for index in range(max(1, args.rounds)):
        raw = call_server(
            prompt=prompt,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        (io_dir / f"round{index:03d}_raw_reply.txt").write_text(raw)
        code = clean_code(raw)
        candidate = out_dir / f"candidate_{index:04d}.py"
        candidate.write_text(code)
        print(f"[kernelllm] wrote {candidate}")
        print(
            "[kernelllm] first line:",
            code.splitlines()[0] if code.splitlines() else "<empty>",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
