#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dest", required=True)
args = parser.parse_args()

source = Path(args.model).expanduser()
destination = Path(args.dest).expanduser().resolve()
destination.parent.mkdir(parents=True, exist_ok=True)

if source.exists():
    source = source.resolve()

    if destination.exists() or destination.is_symlink():
        try:
            if destination.resolve() == source:
                print(destination)
                raise SystemExit(0)
        except Exception:
            pass

        if destination.is_symlink():
            destination.unlink()
        elif destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        elif destination.exists():
            print(destination)
            raise SystemExit(0)

    os.symlink(source, destination, target_is_directory=True)
else:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.model, local_dir=str(destination))

print(destination)
