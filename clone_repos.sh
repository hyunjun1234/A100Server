#!/bin/bash
# clone_repos.sh — repos.yaml 기반 clone + SHA pin → results/repo_lock.json
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results third_party

python3 - <<'PY'
import json, subprocess, pathlib, datetime, yaml
cfg = yaml.safe_load(open("repos.yaml"))
lock = {"cloned_at": datetime.datetime.now().isoformat(), "repos": {}}
for name, spec in cfg["repos"].items():
    url, commit, dest = spec["url"], spec.get("commit", ""), pathlib.Path(spec["dest"])
    if not url:
        print(f"[SKIP] {name}: URL 없음"); continue
    if not dest.exists():
        print(f"[CLONE] {name} <- {url}")
        subprocess.run(["git","clone","--recursive",url,str(dest)], check=True)
    if commit:
        subprocess.run(["git","-C",str(dest),"checkout",commit], check=True)
    sha = subprocess.run(["git","-C",str(dest),"rev-parse","HEAD"],
                         capture_output=True,text=True,check=True).stdout.strip()
    lock["repos"][name] = {"url": url, "commit": sha}
    print(f"[PIN] {name} @ {sha[:12]}")
json.dump(lock, open("results/repo_lock.json","w"), indent=2)
print("lockfile -> results/repo_lock.json")
PY
