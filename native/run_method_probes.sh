#!/usr/bin/env bash
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/method_capabilities

python3 - <<'PY'
import csv,json,subprocess
from pathlib import Path
methods = [
    ("cudaforge", "third_party/CudaForge", "ready_A100", "official_or_near_official"),
    ("autokernel", "third_party/autokernel", "ready_A100", "official_protocol_adapter"),
    ("ksearch", "third_party/K-Search", "optional", "official_entrypoint_adapter"),
    ("cuda_l1", "third_party/CUDA-L1", "ready_A100", "official_A100_artifact"),
    ("cuda_l2", "third_party/CUDA-L2", "ready_A100", "official_A100_HGEMM_artifact"),
    ("autotriton", "third_party/AutoTriton", "ready_A100", "official_checkpoint"),
    ("drkernel", "third_party/KernelGYM", "ready_A100", "official_checkpoint"),
    ("kernelllm", "models/kernelllm8b", "ready_A100_transfer", "official_checkpoint_H100_original"),
    ("qimeng_gemm", "third_party/QiMeng-GEMM", "ready_A100", "official_CUDA_artifact"),
    ("qimeng_tensorop", "third_party/QiMeng-TensorOp", "unsupported_A100_public_artifact", "C920V2_only"),
    ("qimeng_attention", "third_party/QiMeng-Attention", "ready_with_command", "official_repo"),
    ("qimeng_kernel", "third_party/QiMeng-Kernel", "checkpoint_dataset_required", "official_repo_active_development"),
    ("geak", "third_party/GEAK", "unsupported_A100", "AMD_ROCm"),
    ("cuda_agent", "third_party/CUDA-Agent", "checkpoint_unverified", "partial_open"),
    ("kernelagent", "third_party/KernelAgent", "API_provider_required", "official_repo"),
]
rows=[]
for name,destination,status,integration in methods:
    path=Path(destination); commit=''
    if (path/'.git').exists():
        try: commit=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        except Exception: pass
    rows.append({'method':name,'repo_or_model_exists':path.exists(),'status':status,'integration':integration,'systems_env_exists':Path(f'systems/{name}.env').exists(),'commit':commit,'python_files':len(list(path.rglob('*.py'))) if path.exists() else 0})
out=Path('results/method_capabilities/capability.csv'); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
Path('results/method_capabilities/capability.json').write_text(json.dumps(rows,indent=2))
print(out)
for row in rows: print(row)
PY
