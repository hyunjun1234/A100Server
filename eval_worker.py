"""eval_worker.py — 후보 1개를 격리 프로세스에서 판정 (통합 검증기의 실행부).

사용: python3 eval_worker.py <job.json>
job: {ref_path, cand_path, trials, atol, rtol, warmup, timing_iters,
      input_seed_base, out_path}
"""
import importlib.util
import json
import statistics
import sys
import traceback


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(job_path):
    job = json.load(open(job_path))
    v = {"compiled": False, "correct": False, "latency_ms": None,
         "ref_latency_ms": None, "speedup": None, "error": None}
    try:
        import torch
        torch.manual_seed(0)
        dev = "cuda"
        ref_mod = _load(job["ref_path"], "ref_mod")
        try:
            cand_mod = _load(job["cand_path"], "cand_mod")   # JIT 빌드 발생 지점
            v["compiled"] = True
        except Exception:
            v["error"] = "compile: " + traceback.format_exc(limit=3)
            return _emit(job, v)
        if not hasattr(cand_mod, "ModelNew"):
            v["error"] = "interface: ModelNew 없음 (KernelBench 규약)"
            return _emit(job, v)

        init = ref_mod.get_init_inputs()
        ref = ref_mod.Model(*init).to(dev).eval()
        cand = cand_mod.ModelNew(*init).to(dev).eval()

        with torch.no_grad():
            for t in range(job["trials"]):
                torch.manual_seed(job["input_seed_base"] + t)   # 비공개 seed
                xs = [x.to(dev) if hasattr(x, "to") else x for x in ref_mod.get_inputs()]
                y_ref, y = ref(*xs), cand(*xs)
                if not torch.allclose(y, y_ref, atol=job["atol"], rtol=job["rtol"]):
                    v["error"] = (f"correctness: trial {t} mismatch "
                                  f"(max abs {(y - y_ref).abs().max().item():.3e})")
                    return _emit(job, v)
        v["correct"] = True

        def t_ms(m):
            torch.manual_seed(job["input_seed_base"])
            xs = [x.to(dev) if hasattr(x, "to") else x for x in ref_mod.get_inputs()]
            with torch.no_grad():
                for _ in range(job["warmup"]):
                    m(*xs)
                torch.cuda.synchronize()
                s = []
                for _ in range(job["timing_iters"]):
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); m(*xs); b.record()
                    torch.cuda.synchronize()
                    s.append(a.elapsed_time(b))
            return statistics.median(s)

        v["ref_latency_ms"] = t_ms(ref)
        v["latency_ms"] = t_ms(cand)
        v["speedup"] = v["ref_latency_ms"] / v["latency_ms"]
    except Exception:
        v["error"] = "runtime: " + traceback.format_exc(limit=3)
    _emit(job, v)


def _emit(job, v):
    json.dump(v, open(job["out_path"], "w"))


if __name__ == "__main__":
    main(sys.argv[1])
