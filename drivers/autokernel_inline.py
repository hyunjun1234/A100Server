"""drivers/autokernel_inline.py — AutoKernel 충실 재구현 (iterative kernel optimization agent).

공식 repo(third_party/autokernel)는 외부 코딩 에이전트가 수동으로 구동하는 구조라
자율 LLM 드라이버(autokernel.py)가 존재하지 않는다. 이 driver는 autokernel의
program_kb.md 방법론(generate -> benchmark -> keep/improve 반복, level별 전략)을
로컬 LLM(OPENAI_BASE_URL)으로 재현한다.
출력 계약: --cand_dir 에 round{NNN}_kernel.py (baseline/cudaforge와 동일).
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent

SYSTEM = (
    "You are AutoKernel, an autonomous GPU kernel optimization agent for KernelBench. "
    "Given a reference PyTorch `Model`, write an optimized `ModelNew` class with the SAME "
    "__init__ and forward signature that is CORRECT (matches the reference within atol=1e-2) "
    "and as FAST as possible on an NVIDIA A100. You may optimize with a custom CUDA kernel "
    "(torch.utils.cpp_extension.load_inline -- put the wrapper prototype in cpp_sources and the "
    "__global__ kernel + torch::Tensor wrapper in cuda_sources; do NOT use extern \"C\"), with "
    "Triton, or with efficient PyTorch (operator fusion, channels_last, fast internal paths) -- "
    "whichever is fastest. A correct PyTorch implementation is ALWAYS acceptable and is far better "
    "than a custom kernel that fails to compile or is incorrect. CORRECTNESS FIRST, then speed. "
    "Keep any nn.Linear/Conv weights that Model uses; never change the reference. Be concise. "
    "Return EXACTLY ONE complete, self-contained Python file in a single ```python code block."
)


def extract_code(t):
    b = re.findall(r"```(?:python)?\n(.*?)```", t, re.DOTALL)
    return b[-1].strip() if b else t.strip()


def inner_eval(task, cand, io_dir, idx):
    out = io_dir / f"inner_verdict_{idx:03d}.json"
    jp = io_dir / f"inner_job_{idx:03d}.json"
    jp.write_text(json.dumps({"ref_path": task, "cand_path": str(cand),
        "trials": 2, "atol": 1e-2, "rtol": 1e-2, "warmup": 3, "timing_iters": 20,
        "input_seed_base": 7, "out_path": str(out)}))
    try:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "eval_worker.py"), str(jp)],
                       timeout=300, capture_output=True)
        return json.loads(out.read_text())
    except Exception:
        return {"compiled": False, "correct": False, "speedup": None,
                "latency_ms": None, "error": "inner eval crashed/timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand_dir", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    model = os.environ.get("EVAL_MODEL", "default")

    def chat(msgs, s):
        return client.chat.completions.create(
            model=model, messages=msgs, temperature=a.temperature,
            top_p=float(os.environ.get("TOP_P", "0.95")),
            max_tokens=int(os.environ.get("MAX_NEW_TOKENS", "4096")),
            seed=s).choices[0].message.content

    cand_dir = Path(a.cand_dir); cand_dir.mkdir(parents=True, exist_ok=True)
    io_dir = cand_dir / "llm_io"; io_dir.mkdir(exist_ok=True)
    ref = Path(a.task).read_text()

    prev_code, prev_report = None, None
    for r in range(a.rounds):
        if prev_code is None:
            user = (f"Reference Model to optimize:\n```python\n{ref}\n```\n"
                    "Write the optimized ModelNew now.")
        else:
            user = (f"Reference Model:\n```python\n{ref}\n```\n"
                    f"Your previous ModelNew:\n```python\n{prev_code[:6000]}\n```\n"
                    f"Benchmark result of that version: {prev_report}\n"
                    "If it failed compile/correctness, FIX it. If it was correct, apply ONE "
                    "focused change to make it faster. Return the full improved ModelNew.")
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        code = extract_code(chat(msgs, a.seed + r))
        cand = cand_dir / f"round{r:03d}_kernel.py"
        cand.write_text(code)
        (io_dir / f"round{r:03d}_raw_reply.txt").write_text(code)
        v = inner_eval(a.task, cand, io_dir, r)
        report = (f"compiled={v['compiled']} correct={v['correct']} "
                  f"latency_ms={v.get('latency_ms')} speedup={v.get('speedup')} "
                  f"error={(v.get('error') or '')[:1200]}")
        (io_dir / f"round{r:03d}_report.txt").write_text(report)
        prev_code, prev_report = code, report
        print(f"[autokernel round {r}] {report[:120]}")


if __name__ == "__main__":
    main()
