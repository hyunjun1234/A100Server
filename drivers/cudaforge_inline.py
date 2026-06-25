"""drivers/cudaforge_inline.py — CudaForge 충실 재구현 (Coder+Judge 2-agent).

공식 repo(third_party/CudaForge)의 main.py가 endpoint 주입을 지원하면
systems/cudaforge.env에서 repo 모드를 쓰고, 이 driver는 대체용이다.
출력 계약은 baseline과 동일: --cand_dir 에 round{NNN}_kernel.py
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CODER = ("You are the Coder agent. Write or revise an optimized `ModelNew` for "
         "the reference below. Follow the Judge's directives if given. "
         "Return ONE self-contained Python file in a code block.")
JUDGE = ("You are the Judge agent. You receive compile/correctness results and "
         "timing for the current kernel. Diagnose the bottleneck and give the "
         "Coder concrete revision directives (no code).")


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
        subprocess.run([sys.executable, str(SCRIPT_DIR/"eval_worker.py"), str(jp)],
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
    chat = lambda msgs, s: client.chat.completions.create(
        model=model, messages=msgs, temperature=a.temperature,
        top_p=float(os.environ.get("TOP_P", "0.95")),
        max_tokens=int(os.environ.get("MAX_NEW_TOKENS", "8192")),
        seed=s).choices[0].message.content

    cand_dir = Path(a.cand_dir); cand_dir.mkdir(parents=True, exist_ok=True)
    io_dir = cand_dir / "llm_io"; io_dir.mkdir(exist_ok=True)
    ref = Path(a.task).read_text()

    directive = "Start with a correct, straightforward kernel."
    for r in range(a.rounds):
        coder_msgs = [{"role": "system", "content": CODER},
                      {"role": "user", "content":
                       f"Reference:\n```python\n{ref}\n```\nJudge directive: {directive}"}]
        code = extract_code(chat(coder_msgs, a.seed + 2*r))
        cand = cand_dir / f"round{r:03d}_kernel.py"
        cand.write_text(code)
        v = inner_eval(a.task, cand, io_dir, r)
        report = (f"compiled={v['compiled']} correct={v['correct']} "
                  f"latency_ms={v.get('latency_ms')} speedup={v.get('speedup')} "
                  f"error={(v.get('error') or '')[:1500]}")
        (io_dir / f"round{r:03d}_judge_prompt.txt").write_text(report)
        judge_msgs = [{"role": "system", "content": JUDGE},
                      {"role": "user", "content":
                       f"Current kernel:\n```python\n{code[:6000]}\n```\nReport: {report}"}]
        directive = chat(judge_msgs, a.seed + 2*r + 1)
        (io_dir / f"round{r:03d}_judge_reply.txt").write_text(directive)
        print(f"[round {r}] {report[:110]}")


if __name__ == "__main__":
    main()
