#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS = [
    ("cuda_oom", re.compile(r"cuda out of memory|outofmemoryerror", re.I)),
    ("illegal_memory", re.compile(r"illegal memory access|misaligned address", re.I)),
    ("device_mismatch", re.compile(r"same device|cuda:\d+ and cpu|expected all tensors", re.I)),
    ("timeout", re.compile(r"timed? out|timeout|exit code 124", re.I)),
    ("missing_module", re.compile(r"modulenotfounderror|no module named", re.I)),
    ("missing_file", re.compile(r"filenotfounderror|no such file|can't open file", re.I)),
    ("syntax", re.compile(r"syntaxerror|invalid syntax|modelnew not found", re.I)),
    ("compile", re.compile(r"compilationerror|error building extension|nvcc.*error|ninja: build stopped", re.I | re.S)),
    ("api_http", re.compile(r"internalservererror|connection refused|http.*(?:4\d\d|5\d\d)|api key", re.I)),
    ("format_parse", re.compile(r"no ``` code block|failed to extract|missing required xml|json.*extract", re.I)),
    ("correctness", re.compile(r"correctness.*(?:false|mismatch)|max_abs|allclose", re.I)),
    ("interface", re.compile(r"modelnew missing|forward.*tensor|did not return a tensor", re.I)),
]

ERROR_LINE = re.compile(
    r"traceback|error|exception|failed|failure|timeout|killed|missing|mismatch|incorrect|oom|out of memory",
    re.I,
)


def classify(text: str) -> str:
    for name, pattern in PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def normalize(text: str) -> str:
    text = re.sub(r"/[^\s:'\"]+", "<PATH>", text)
    text = re.sub(r"0x[0-9a-f]+", "<HEX>", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<N>", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text[:500]


def infer_metadata(path: Path, root: Path):
    relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    known = [
        "cudaforge", "autokernel", "ksearch", "cuda_l1", "cuda_l2",
        "autotriton", "drkernel", "kernelllm", "qimeng_gemm",
        "qimeng_tensorop", "qimeng_attention", "qimeng_kernel",
        "cuda_agent", "geak", "kernelagent",
    ]
    system = next((name for name in known if name in relative.lower()), "unknown")
    task = ""
    for part in path.parts:
        if re.match(r"\d+_.*", part):
            task = part
    return relative, system, task


def extract_log_errors(path: Path):
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return []
    results = []
    for index, line in enumerate(lines):
        if not ERROR_LINE.search(line):
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 4)
        excerpt = "\n".join(lines[start:end]).strip()
        results.append((index + 1, excerpt))
    return results[-1000:]


def extract_json_errors(path: Path):
    try:
        obj = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return []
    results = []
    stack = [("$", obj)]
    while stack:
        key, value = stack.pop()
        if isinstance(value, dict):
            for child_key, child in value.items():
                stack.append((f"{key}.{child_key}", child))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                stack.append((f"{key}[{index}]", child))
        elif isinstance(value, str) and ERROR_LINE.search(value):
            results.append((key, value[:4000]))
    return results[-1000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    records = []
    candidates = []
    for base in [root / "logs", root / "runs", root / "results"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
                continue
            if path.suffix.lower() in {".out", ".log", ".txt"}:
                candidates.append((path, extract_log_errors(path)))
            elif path.suffix.lower() == ".json":
                candidates.append((path, extract_json_errors(path)))

    for path, errors in candidates:
        relative, system, task = infer_metadata(path, root)
        for location, excerpt in errors:
            signature = normalize(excerpt)
            records.append({
                "category": classify(excerpt),
                "system": system,
                "task": task,
                "file": relative,
                "location": location,
                "signature_hash": hashlib.sha256(signature.encode()).hexdigest()[:12],
                "normalized_signature": signature,
                "excerpt": excerpt,
            })

    error_csv = out / "errors.csv"
    fields = ["category", "system", "task", "file", "location", "signature_hash", "normalized_signature", "excerpt"]
    with error_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    groups = defaultdict(list)
    for record in records:
        groups[(record["category"], record["system"], record["signature_hash"])].append(record)
    clusters = []
    for (category, system, signature_hash), items in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        clusters.append({
            "count": len(items),
            "category": category,
            "system": system,
            "signature_hash": signature_hash,
            "example_task": items[0]["task"],
            "example_file": items[0]["file"],
            "example_excerpt": items[0]["excerpt"][:1500],
        })
    cluster_csv = out / "error_clusters.csv"
    cluster_fields = ["count", "category", "system", "signature_hash", "example_task", "example_file", "example_excerpt"]
    with cluster_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=cluster_fields)
        writer.writeheader()
        writer.writerows(clusters)

    category_counts = Counter(record["category"] for record in records)
    system_counts = Counter(record["system"] for record in records)
    overview = [
        "# Error feedback overview",
        "",
        f"Total extracted error records: {len(records)}",
        f"Unique clusters: {len(clusters)}",
        "",
        "## By category",
    ]
    overview += [f"- {key}: {value}" for key, value in category_counts.most_common()]
    overview += ["", "## By system"]
    overview += [f"- {key}: {value}" for key, value in system_counts.most_common()]
    overview += ["", "## Top clusters"]
    for cluster in clusters[:30]:
        overview.append(
            f"- [{cluster['category']}] {cluster['system']} x{cluster['count']} "
            f"({cluster['signature_hash']}): {cluster['example_excerpt'].splitlines()[0][:200]}"
        )
    (out / "ERROR_OVERVIEW.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(error_csv)
    print(cluster_csv)


if __name__ == "__main__":
    main()
