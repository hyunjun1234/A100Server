"""drivers/baseline_loop.py — 대조군 driver (KernelBench식 단일 context refinement).

계약(모든 driver/repo 공통): --cand_dir 에 round{NNN}_kernel.py 를 남긴다.
보고용 판정은 final_eval.py가 별도로 수행하므로, 여기서의 자체 평가는
loop feedback 용도로만 쓴다.

사용: python3 drivers/baseline_loop.py --task <ref.py> --cand_dir <dir> \
        --rounds 10 --seed 1 --temperature 0.2 --io_dir <llm_io dir>
endpoint: OPENAI_BASE_URL / OPENAI_API_KEY / EVAL_MODEL 환경변수
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer. Given a PyTorch reference model, "
    "write an optimized replacement class named `ModelNew` with identical "
    "semantics, using custom CUDA (torch.utils.cpp_extension.load_inline) or "
    "Triton. Return ONE self-contained Python file in a single code block."
)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else text.strip()


def quick_feedback(task: str, cand: Path, io_dir: Path, idx: int) -> str:
    """loop 내부용 약식 평가 (eval_worker 재사용, 보고에는 미사용)."""
    out = io_dir / f"inner_verdict_{idx:03d}.json"
    job = {"ref_path": task, "cand_path": str(cand),
           "trials": 2, "atol": 1e-2, "rtol": 1e-2,
           "warmup": 3, "timing_iters": 20,
           "input_seed_base": 7,           # 내부용 seed (최종 판정 seed와 다름)
           "out_path": str(out)}
    jp = io_dir / f"inner_job_{idx:03d}.json"
    jp.write_text(json.dumps(job))
    try:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "eval_worker.py"), str(jp)],
                       timeout=300, capture_output=True)
        v = json.loads(out.read_text())
    except Exception:
        return "Evaluation crashed or timed out. Simplify the kernel and fix errors."
    if not v["compiled"]:
        return f"Compilation failed:\n{(v['error'] or '')[:2000]}\nFix the error."
    if not v["correct"]:
        return f"Output mismatch:\n{(v['error'] or '')[:1000]}\nFix correctness first."
    return (f"Correct. speedup={v['speedup']:.2f}x latency={v['latency_ms']:.3f}ms. "
            f"Optimize further (memory coalescing, fusion, shared memory, vectorization).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand_dir", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--io_dir", default=None)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    model = os.environ.get("EVAL_MODEL", "default")

    cand_dir = Path(a.cand_dir); cand_dir.mkdir(parents=True, exist_ok=True)
    io_dir = Path(a.io_dir or cand_dir / "llm_io"); io_dir.mkdir(parents=True, exist_ok=True)
    ref_code = Path(a.task).read_text()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference:\n```python\n{ref_code}\n```"}]
    for r in range(a.rounds):
        (io_dir / f"round{r:03d}_prompt.txt").write_text(json.dumps(msgs, indent=2))
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=a.temperature,
            top_p=float(os.environ.get("TOP_P", "0.95")),
            max_tokens=int(os.environ.get("MAX_NEW_TOKENS", "8192")),
            seed=a.seed + r)
        text = resp.choices[0].message.content
        (io_dir / f"round{r:03d}_raw_reply.txt").write_text(text)
        cand = cand_dir / f"round{r:03d}_kernel.py"
        cand.write_text(extract_code(text))
        fb = quick_feedback(a.task, cand, io_dir, r)
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": fb}]
        print(f"[round {r}] {fb.splitlines()[0][:100]}")


if __name__ == "__main__":
    main()
