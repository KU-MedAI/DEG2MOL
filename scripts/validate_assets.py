#!/usr/bin/env python3
"""Validate externally distributed DEG2MOL data and checkpoints."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(item, role, split):
    role_matches = role == "all" or role in item["roles"]
    split_matches = "splits" not in item or split in item["splits"]
    return role_matches and split_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["all", "train", "test", "inference"], default="all")
    parser.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    parser.add_argument("--root", type=Path, default=ROOT, help="Asset tree root")
    parser.add_argument("--skip-hash", action="store_true", help="Only check paths and sizes")
    args = parser.parse_args()
    asset_root = args.root.expanduser().resolve()

    with (ROOT / "assets_manifest.json").open() as handle:
        manifest = json.load(handle)
    failures = []

    for item in manifest["files"]:
        if not selected(item, args.role, args.split):
            continue
        path = asset_root / item["path"]
        if not path.is_file():
            failures.append(f"MISSING FILE: {item['path']}")
            continue
        if path.stat().st_size != item["size"]:
            failures.append(
                f"SIZE MISMATCH: {item['path']} "
                f"({path.stat().st_size} != {item['size']})"
            )
            continue
        if not args.skip_hash and sha256(path) != item["sha256"]:
            failures.append(f"SHA256 MISMATCH: {item['path']}")

    for item in manifest["directories"]:
        if not selected(item, args.role, args.split):
            continue
        path = asset_root / item["path"]
        if not path.is_dir():
            failures.append(f"MISSING DIRECTORY: {item['path']}")
            continue
        count = sum(child.is_file() for child in path.iterdir())
        if count != item["file_count"]:
            failures.append(
                f"FILE COUNT MISMATCH: {item['path']} ({count} != {item['file_count']})"
            )

    if failures:
        print("Asset validation failed:")
        print("\n".join(f"  - {failure}" for failure in failures))
        return 1
    print(f"Assets valid for role={args.role}, split={args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
