#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def has_model_files(p: Path) -> bool:
    return (p / "config.json").exists() or (p / "tokenizer_config.json").exists()


def main():
    ap = argparse.ArgumentParser()

    # both positional and optional forms supported
    ap.add_argument("model_id_pos", nargs="?")
    ap.add_argument("local_dir_pos", nargs="?")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--local_dir", default=None)

    args = ap.parse_args()

    model_id = args.model_id or args.model_id_pos
    local_dir = args.local_dir or args.local_dir_pos

    if not model_id or not local_dir:
        raise SystemExit("Usage: download_model.py MODEL_ID LOCAL_DIR or --model_id MODEL_ID --local_dir LOCAL_DIR")

    src = Path(model_id).expanduser()
    dst = Path(local_dir).expanduser()

    print("==========================================")
    print("Downloading model")
    print("==========================================")
    print("Model ID :", model_id)
    print("Local dir:", dst)
    print("==========================================")

    # Case 1: model_id is already a local model path
    if src.exists():
        src = src.resolve()
        dst_parent = dst.parent
        dst_parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src:
                    print("[download_model] local model symlink/path already points to source.")
                    return
            except Exception:
                pass

            if has_model_files(dst):
                print("[download_model] local_dir already contains model files. Skip.")
                return

            # remove empty or broken previous target
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
            elif dst.exists():
                print(f"[download_model] WARNING: {dst} exists and is not empty.")
                print("[download_model] Not overwriting. If this is stale, remove it manually.")
                return

        os.symlink(src, dst, target_is_directory=True)
        print(f"[download_model] using local model path via symlink: {dst} -> {src}")
        return

    # Case 2: model_id is HuggingFace repo id
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(dst))
    print("[download_model] download complete.")


if __name__ == "__main__":
    main()
