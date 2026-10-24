#!/usr/bin/env python3
"""Download only pinned upstream commits. No automatic branch-following."""
import argparse
import json
import re
import sys
from pathlib import Path

def main():
    here = Path(__file__).resolve().parent
    models = json.loads((here / "models.json").read_text(encoding="utf-8"))["models"]
    locks = json.loads((here.parent / "versions.lock.json").read_text(encoding="utf-8"))["revisions"]
    choices = {m["name"]: m for m in models}
    parser = argparse.ArgumentParser(description="Download a fixed-model snapshot.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", choices=sorted(choices))
    parser.add_argument("--destination")
    parser.add_argument("--revision", help="Explicit manual override: full 40-character commit SHA")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, m in sorted(choices.items()):
            print(name, m["hf_repo"], locks[m["hf_repo"]])
        return 0
    if not args.model:
        parser.error("Select --model or use --list")
    selected = choices[args.model]
    repo_id = selected["hf_repo"]
    revision = args.revision or locks[repo_id]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        parser.error("Only a full frozen commit SHA is allowed, not main or latest")
    destination = Path(args.destination).expanduser().resolve() if args.destination else here / "weights" / args.model
    print("Frozen version:", "https://huggingface.co/" + repo_id + "/tree/" + revision)
    print("Destination:", destination)
    if args.dry_run:
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install: pip install huggingface_hub", file=sys.stderr)
        return 2
    opts = {"repo_id": repo_id, "revision": revision, "local_dir": str(destination)}
    if args.metadata_only:
        opts["allow_patterns"] = ["*.json", "*.md", "*.txt", "*.jinja", "LICENSE*", "NOTICE*"]
    try:
        snapshot_download(**opts)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "SNAPSHOT_LOCK.json").write_text(
            json.dumps({"repo_id": repo_id, "revision": revision,
                        "metadata_only": args.metadata_only},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    except Exception as exc:
        print("Snapshot unavailable or download failed:", exc, file=sys.stderr)
        return 1
    print("Saved a copy to:", destination)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
