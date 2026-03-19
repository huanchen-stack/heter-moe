"""
Download MoE models one file at a time (avoids parallel OOM).

Usage:
    python downloader.py                          # download all models
    python downloader.py --model qwen1.5-moe      # download one model
    python downloader.py --model qwen3-30b-a3b
"""
import argparse
import os
from huggingface_hub import hf_hub_download, list_repo_files

MODELS = {
    "qwen1.5-moe": {
        "repo": "Qwen/Qwen1.5-MoE-A2.7B",
        "dir": "Qwen1.5-MoE-A2.7B",
        "desc": "14.3B total, 2.7B active, qwen2_moe",
    },
    "qwen3-30b-a3b": {
        "repo": "Qwen/Qwen3-30B-A3B",
        "dir": "Qwen3-30B-A3B",
        "desc": "30B total, 3B active, qwen3_moe (128 experts, top-8)",
    },
}


def download_model(key: str, base_dir: str):
    info = MODELS[key]
    repo_id = info["repo"]
    local_dir = os.path.join(base_dir, info["dir"])
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"  {info['desc']}")
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
    parser = argparse.ArgumentParser(description="Download MoE models")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODELS.keys()),
                        help="Download a specific model (default: all)")
    parser.add_argument("--base_dir", type=str, default="./models",
                        help="Base directory for models (default: ./models)")
    args = parser.parse_args()

    if args.model:
        download_model(args.model, args.base_dir)
    else:
        for key in MODELS:
            download_model(key, args.base_dir)
