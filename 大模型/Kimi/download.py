#!/usr/bin/env python3
"""Download one model from this standalone provider catalog.

This does not integrate with any other project and is never run automatically.
Requires: pip install huggingface_hub
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    catalog = json.loads((base / "models.json").read_text(encoding="utf-8"))
    available = {m["name"]: m for m in catalog["models"]}
    parser = argparse.ArgumentParser(description="Download a model snapshot from the official publisher.")
    parser.add_argument("--list", action="store_true", help="List registered models without downloading.")
    parser.add_argument("--model", choices=sorted(available), help="Choose a model.")
    parser.add_argument("--destination", help="Destination directory (defaults to this folder/weights/<model>).")
    parser.add_argument("--revision", help="Pin an exact upstream revision or commit for reproducibility.")
    parser.add_argument("--metadata-only", action="store_true", help="Only retrieve small text/config files, not weights.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without downloading.")
    args = parser.parse_args()

    if args.list:
        for name, item in sorted(available.items()):
            print(f"{name}: https://huggingface.co/{item['hf_repo']}")
        return 0
    if not args.model:
        parser.error("--model is required unless --list was provided")
    chosen = available[args.model]
    target = (
        Path(args.destination).expanduser().resolve()
        if args.destination
        else base / "weights" / args.model
    )
    print(f"Official source: https://huggingface.co/{chosen['hf_repo']}")
    print(f"Destination: {target}")
    print("Weights in this GitHub repository: NO")
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
        return 2

    kwargs = {"repo_id": chosen["hf_repo"], "local_dir": str(target)}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.metadata_only:
        kwargs["allow_patterns"] = ["*.json", "*.md", "*.txt", "*.jinja", "LICENSE*", "NOTICE*"]
    try:
        location = snapshot_download(**kwargs)
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        print("Check network, available disk space, model access approval and HF_TOKEN.", file=sys.stderr)
        return 1
    print(f"Downloaded snapshot: {location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
