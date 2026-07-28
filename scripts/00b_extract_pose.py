"""Step 0b — cache 2D pose keypoints (run the pose model ONCE).

Runs a COCO pose estimator over every camera image of the pedestrian sequences and writes the
per-image person skeletons to data/processed/pose/{seq_id}.json. PedGraph+ (committee member #2)
then reads this cache to build its 32-frame keypoint windows, instead of paying for the pose model
each time it scores.

This mirrors Step 0 (`scripts/00_detect.py`): the pose model is the only expensive stage, so cache
it once at a LOW --conf floor (a run raises the threshold for free at load time) and re-run this ONLY
when the pose GEOMETRY changes (model / imgsz). It is independent of Step 0 — the two caches coexist;
Step 1 keeps using the box detections, PedGraph uses these skeletons.

Association to a tracked pedestrian happens downstream, not here: this cache stores every person the
model finds; `pose_cache.best_pose_for_box` picks the right skeleton by IoU against a known box.

Usage:
    python scripts/00b_extract_pose.py                 # full set (~3-4 h at imgsz 2560)
    python scripts/00b_extract_pose.py --max-seqs 2    # smoke test
    python scripts/00b_extract_pose.py --imgsz 1280    # faster, but invalidates the imgsz-2560 baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from zodped.labeling.pose_cache import cache_path, write_poses
from zodped.labeling.pose_estimator import make_pose_estimator

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "pose"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x-pose.pt", help="pose model weights (yolo*-pose)")
    ap.add_argument("--conf", type=float, default=0.1, help="person confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560, help="pose inference image size")
    ap.add_argument("--camera", default="camera_front_blur")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-seq pose JSONs")
    ap.add_argument("--overwrite", action="store_true", help="re-extract sequences already cached")
    args = ap.parse_args()

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    meta = {"model": args.model, "conf": args.conf, "imgsz": args.imgsz, "camera": args.camera}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimator = make_pose_estimator(args.model, conf=args.conf, imgsz=args.imgsz)
    print(f"Step 0b: caching poses for {len(seq_ids)} sequences → {args.out_dir}\n  config: {meta}")

    total_imgs = 0
    t0 = time.perf_counter()
    for i, seq_id in enumerate(seq_ids):
        out_path = cache_path(args.out_dir, seq_id)
        if out_path.exists() and not args.overwrite:
            continue
        imgs = sorted((SEQ_DIR / seq_id / args.camera).glob("*.jpg"))
        per_image = {img.name: estimator(img) for img in imgs}
        write_poses(out_path, meta, per_image)
        total_imgs += len(imgs)
        if (i + 1) % 10 == 0:
            rate = total_imgs / max(time.perf_counter() - t0, 1e-6)
            print(f"  ...{i + 1}/{len(seq_ids)} seqs  ({total_imgs} images, {rate:.1f} img/s)")

    dt = time.perf_counter() - t0
    print(f"\nDone. {total_imgs} images over {len(seq_ids)} sequences in {dt / 60:.1f} min → {args.out_dir}")


if __name__ == "__main__":
    main()
