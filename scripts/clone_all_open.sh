#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p third_party models results

python3 - <<'PY'
import datetime
import json
import subprocess
from pathlib import Path

import yaml

config = yaml.safe_load(Path("registry/open_repos.yaml").read_text())
entries = {}

for section in ("benchmarks", "methods"):
    for name, specification in config.get(section, {}).items():
        destination = specification.get("dest", "")
        key = destination or f"__no_clone__/{section}/{name}"
        entries[key] = {
            "name": name,
            "section": section,
            **specification,
        }

lock = {
    "created_at": datetime.datetime.now().isoformat(),
    "repos": {},
    "models": {},
}

for _, specification in entries.items():
    name = specification["name"]
    url = specification["url"]
    destination = specification.get("dest", "")
    clone = specification.get("clone", True)

    if not clone:
        lock["models"][name] = {
            "url": url,
            "destination": destination,
            "hf_id": specification.get("hf_id"),
            "a100": specification.get("a100"),
            "technique": specification.get("technique"),
        }
        print(f"[MODEL/NO-CLONE] {name}: {url}")
        continue

    path = Path(destination)
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[CLONE] {name} <- {url}")
            subprocess.run(
                ["git", "clone", "--recursive", url, str(path)],
                check=True,
            )
        else:
            print(f"[EXISTS] {name} -> {path}")

        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        lock["repos"][name] = {
            "url": url,
            "destination": destination,
            "commit": commit,
            "a100": specification.get("a100"),
            "mode": specification.get("mode"),
            "technique": specification.get("technique"),
        }
    except Exception as exc:
        lock["repos"][name] = {
            "url": url,
            "destination": destination,
            "error": repr(exc),
            "a100": specification.get("a100"),
            "mode": specification.get("mode"),
        }
        print(f"[ERROR] {name}: {exc}")

Path("results/open_repo_lock.json").write_text(
    json.dumps(lock, indent=2, ensure_ascii=False)
)
print("lockfile -> results/open_repo_lock.json")
PY
