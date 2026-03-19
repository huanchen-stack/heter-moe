"""
Download Qwen3-30B-A3B model one file at a time (avoids parallel OOM).

Usage:
    python downloader.py
    python downloader.py --base_dir /path/to/models
"""
import argparse
import os
from huggingface_hub import hf_hub_download, list_repo_files

MODEL = {
    "repo": "Qwen/Qwen3-30B-A3B",
    "dir": "Qwen3-30B-A3B",
    "desc": "30B total, 3B active, qwen3_moe (128 experts, top-8)",
}


def download_model(base_dir: str):
    repo_id = MODEL["repo"]
    local_dir = os.path.join(base_dir, MODEL["dir"])
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"  {MODEL['desc']}")
    print(f"  -> {local_dir}")
    print(f"{'='*60}")

    files = sorted(list_repo_files(repo_id))
    print(f"Found {len(files)} files")

    for i, f in enumerate(files):
        dst = os.path.join(local_dir, f)
        if os.path.exists(dst):
            print(f"[{i+1}/{len(files)}] SKIP (exists) {f}")
            continue
        print(f"[{i+1}/{len(files)}] {f}...")
        hf_hub_download(repo_id, f, local_dir=local_dir)

    print(f"Done! -> {local_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Qwen3-30B-A3B model")
    parser.add_argument("--base_dir", type=str, default="./models",
                        help="Base directory for models (default: ./models)")
    args = parser.parse_args()

    download_model(args.base_dir)
