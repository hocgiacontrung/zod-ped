"""
Delete lidar_velodyne/ dirs for sequences that have no pedestrian annotations.
Run after extracting each LiDAR batch tarball.

Usage:
    python scripts/prune_lidar.py              # dry-run (shows what would be deleted)
    python scripts/prune_lidar.py --delete     # actually delete
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_JSON = ROOT / "data" / "pedestrian_sequences.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    with open(PED_JSON) as f:
        ped_seqs = {s["seq_id"] for s in json.load(f)}

    to_delete = []
    for lidar_dir in sorted(SEQ_DIR.glob("*/lidar_velodyne")):
        seq_id = lidar_dir.parent.name
        if seq_id not in ped_seqs:
            to_delete.append(lidar_dir)

    total_size = sum(
        f.stat().st_size for d in to_delete for f in d.rglob("*") if f.is_file()
    )

    print(f"{'DRY RUN — ' if not args.delete else ''}Found {len(to_delete)} non-pedestrian lidar_velodyne dirs")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print()

    for d in to_delete:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  {'[DELETE]' if args.delete else '[would delete]'} {d.parent.name}/lidar_velodyne/  ({size/1e6:.0f} MB)")
        if args.delete:
            shutil.rmtree(d)

    if not args.delete:
        print(f"\nRe-run with --delete to free {total_size / 1e9:.1f} GB")
    else:
        print(f"\nDone. Freed {total_size / 1e9:.1f} GB.")


if __name__ == "__main__":
    main()
