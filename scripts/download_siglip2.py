#!/usr/bin/env python3
"""Download SigLIP2 weights into data/models for offline / real-time use.

Example:
  python3 scripts/download_siglip2.py
  python3 scripts/download_siglip2.py --out /home/cyf/vslam_ws/src/vslam/data/models/siglip2-so400m-patch14-384

After this finishes, point semantic_siglip_model (or --model) at the local folder.
The semantic_index_node loads with local_files_only=True and will NOT hit the network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_REPO = "google/siglip2-so400m-patch14-384"


def default_out() -> Path:
    # scripts/ -> package root -> data/models/<name>
    return Path(__file__).resolve().parents[1] / "data" / "models" / DEFAULT_REPO.split("/")[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SigLIP2 into data/models")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=None, help="Local directory for the snapshot")
    args = parser.parse_args()

    out = args.out or default_out()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first: pip3 install --user huggingface_hub", file=sys.stderr)
        return 1

    print(f"[download_siglip2] repo={args.repo}", flush=True)
    print(f"[download_siglip2] out ={out}", flush=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
    )
    # Sanity: config + weights present
    has_config = (out / "config.json").is_file()
    has_weight = any(out.glob("*.safetensors")) or any(out.glob("pytorch_model*.bin")) or any(
        out.glob("model.safetensors*")
    )
    print(f"[download_siglip2] config.json={has_config} weights={has_weight}", flush=True)
    if not has_config or not has_weight:
        print("[download_siglip2] incomplete snapshot", file=sys.stderr)
        return 2
    print("[download_siglip2] done. Use this path as semantic_siglip_model / --model:", flush=True)
    print(str(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
