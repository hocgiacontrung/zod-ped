"""Step 0 — cache 2D detections (run YOLO ONCE).

Runs the COCO 2D detector over every camera image of the pedestrian sequences and writes the
per-image person boxes to data/processed/detections/{seq_id}.json. Step 1
(scripts/01_generate_trajectories.py) then reads this cache instead of running YOLO, so the
lift / track / box-assembly loop re-runs in minutes rather than hours.

The detector is the only expensive stage (~170 ms/image at imgsz 2560 → ~3-4 h for the full set).
Re-run this ONLY when the detector GEOMETRY changes (model / imgsz); geometry and tracking
parameters do not affect the cache. Cache at a LOW --conf floor: a run raises the threshold for free
at load time, so conf is not an invalidator.

Usage:
    python scripts/00_detect.py                 # full set (~3-4 h at imgsz 2560)
    python scripts/00_detect.py --max-seqs 2    # smoke test
    python scripts/00_detect.py --imgsz 1280    # faster, but invalidates the imgsz-2560 baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from zodped.labeling.detection_cache import cache_path, write_detections
from zodped.labeling.detector import make_detector

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "detections"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x.pt", help="2D detector weights (yolo*/rtdetr*)")
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560, help="detector inference image size")
    ap.add_argument("--camera", default="camera_front_blur")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-seq detection JSONs")
    ap.add_argument("--overwrite", action="store_true", help="re-detect sequences already cached")
    args = ap.parse_args()

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    meta = {"model": args.model, "conf": args.conf, "imgsz": args.imgsz, "camera": args.camera}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = make_detector(args.model, conf=args.conf, imgsz=args.imgsz)
    print(f"Step 0: caching detections for {len(seq_ids)} sequences → {args.out_dir}\n  config: {meta}")

    total_imgs = 0
    t0 = time.perf_counter()
    for i, seq_id in enumerate(seq_ids):
        out_path = cache_path(args.out_dir, seq_id)
        if out_path.exists() and not args.overwrite:
            continue
        imgs = sorted((SEQ_DIR / seq_id / args.camera).glob("*.jpg"))
        per_image = {img.name: detector(img) for img in imgs}
        write_detections(out_path, meta, per_image)
        total_imgs += len(imgs)
        if (i + 1) % 10 == 0:
            rate = total_imgs / max(time.perf_counter() - t0, 1e-6)
            print(f"  ...{i + 1}/{len(seq_ids)} seqs  ({total_imgs} images, {rate:.1f} img/s)")

    dt = time.perf_counter() - t0
    print(f"\nDone. {total_imgs} images over {len(seq_ids)} sequences in {dt / 60:.1f} min → {args.out_dir}")


if __name__ == "__main__":
    main()
