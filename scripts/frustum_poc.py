"""Frustum POC — lift YOLO 2D detections to 3D via LiDAR, compare to the 3D keyframe boxes.

The architecture question after gates #2/#3: a LiDAR detector (PointPillars) generalizes
poorly to ZOD, but a COCO 2D detector (YOLO) recalls pedestrians well. Can we get the 3D
*position* we actually need by combining them?

Pipeline, per sequence (keyframe only):
  1. YOLO on the keyframe IMAGE -> 2D pedestrian boxes.
  2. Project the nearest keyframe LiDAR scan into the image (Kannala, src/utils/projection).
  3. For each 2D box, gather the LiDAR points whose projection lands inside it (the frustum).
  4. Estimate the pedestrian's 3D centre from those points via a NEAREST-DEPTH SLAB (reject
     the building/road behind and keep the closest compact cluster), median for robustness.
  5. Match the estimated 3D centres to the verified `location_3d` keyframe boxes by BEV centre
     distance (same gate as the PointPillars eval) -> recall / precision / LOCALIZATION ERROR.

The localization error is the key new number: it says whether 2D+LiDAR yields *accurate* 3D
positions, not just whether a pedestrian is somewhere in the frustum.

Usage:
    python scripts/frustum_poc.py --max-seqs 5         # smoke test
    python scripts/frustum_poc.py                      # full set
    python scripts/frustum_poc.py --conf 0.1 --imgsz 2560 --box-shrink 0.6 --slab 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.keyframe import (  # noqa: E402
    MAX_LIDAR_GAP_S, GtBox, keyframe_image_path, lidar_ts_from_filename,
    load_keyframe_pedestrians, load_xyzi, motion_compensate_to_keyframe,
    nearest_lidar_scan, parse_zod_ts, points_in_box,
)
from utils.projection import load_calibration, project_lidar_to_image  # noqa: E402

SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "frustum_poc_report.json"

RANGE_BINS = [0.0, 10.0, 20.0, 30.0, 40.0, np.inf]
POINT_BINS = [1, 5, 10, 20, 50, np.inf]


@dataclass
class Frustum3D:
    """A 2D detection lifted to a 3D centre in the LiDAR frame."""

    center: np.ndarray   # (3,) x, y, z
    score: float
    n_points: int        # LiDAR points used for the estimate


def lift_box_to_3d(box_xyxy: np.ndarray, uv: np.ndarray, pts: np.ndarray,
                   box_shrink: float, slab_m: float, min_pts: int) -> Optional[Tuple[np.ndarray, int]]:
    """Estimate a 3D centre from the LiDAR points inside a (shrunk) 2D box.

    `uv` is the (M,2) pixel projection of the visible LiDAR points `pts` (M,3, LiDAR frame).
    We shrink the box toward its centre column (pedestrians are tall+thin; the box edges leak
    background), keep the points inside it, then take the NEAREST depth slab and its median.
    """
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    hw = 0.5 * (x2 - x1) * box_shrink
    # keep central column horizontally; trim head/feet a little vertically (blur + ground)
    y_lo = y1 + 0.10 * (y2 - y1)
    y_hi = y1 + 0.90 * (y2 - y1)
    inside = (uv[:, 0] >= cx - hw) & (uv[:, 0] <= cx + hw) & (uv[:, 1] >= y_lo) & (uv[:, 1] <= y_hi)
    fp = pts[inside]
    if len(fp) < min_pts:
        return None
    rng = np.linalg.norm(fp[:, :2], axis=1)
    r0 = np.percentile(rng, 15)            # robust "nearest" range
    slab = fp[rng <= r0 + slab_m]          # closest compact cluster (reject background)
    if len(slab) < min_pts:
        slab = fp
    return np.median(slab, axis=0), len(slab)


def match_greedy_3d(dets: List[Frustum3D], gts: List[GtBox], match_dist_m: float):
    """Greedy BEV-centre matching (highest score first). Returns (gt_matched, errors_m)."""
    gt_matched = [False] * len(gts)
    errors: List[float] = []
    if not gts:
        return gt_matched, errors
    gt_xy = np.array([g.center[:2] for g in gts])
    for det in sorted(dets, key=lambda d: -d.score):
        d = np.linalg.norm(gt_xy - det.center[:2], axis=1)
        d[gt_matched] = np.inf
        best = int(np.argmin(d))
        if d[best] <= match_dist_m:
            gt_matched[best] = True
            errors.append(float(d[best]))
    return gt_matched, errors


@dataclass
class GtRecord:
    range_xy: float
    n_points: int
    matched: bool


@dataclass
class Accumulator:
    records: List[GtRecord] = field(default_factory=list)
    n_pred: int = 0
    n_pred_matched: int = 0
    loc_errors: List[float] = field(default_factory=list)
    per_seq: dict = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)

    def add(self, seq_id, gts, points, matched, errors, n_pred):
        tp = sum(matched)
        self.n_pred += n_pred
        self.n_pred_matched += tp
        self.loc_errors += errors
        self.per_seq[seq_id] = {"n_gt": len(gts), "recall": _safe_div(tp, len(gts))}
        for box, hit in zip(gts, matched):
            self.records.append(GtRecord(box.range_xy, points_in_box(points, box), hit))


def _safe_div(a, b):
    return round(a / b, 4) if b else None


def _binned(records, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in records if lo <= key(r) < hi]
        hit = sum(r.matched for r in sel)
        hl = "inf" if hi == np.inf else int(hi)
        out.append({"bin": f"[{int(lo)},{hl})", "n_gt": len(sel), "recall": _safe_div(hit, len(sel))})
    return out


def build_report(acc: Accumulator, config: dict) -> dict:
    n_gt = len(acc.records)
    n_hit = sum(r.matched for r in acc.records)
    err = np.array(acc.loc_errors) if acc.loc_errors else np.zeros(0)
    return {
        "config": config,
        "overall": {
            "n_sequences": len(acc.per_seq), "n_gt": n_gt, "n_pred": acc.n_pred,
            "tp": n_hit, "fn": n_gt - n_hit, "fp": acc.n_pred - acc.n_pred_matched,
            "recall": _safe_div(n_hit, n_gt), "precision": _safe_div(acc.n_pred_matched, acc.n_pred),
        },
        "localization_error_m": {
            "n_matched": int(err.size),
            "mean": round(float(err.mean()), 3) if err.size else None,
            "median": round(float(np.median(err)), 3) if err.size else None,
            "p90": round(float(np.percentile(err, 90)), 3) if err.size else None,
            "errors": [round(float(e), 3) for e in acc.loc_errors],  # raw, for histograms
        },
        "by_range_xy_m": _binned(acc.records, lambda r: r.range_xy, RANGE_BINS),
        "by_lidar_points": _binned(acc.records, lambda r: r.n_points, POINT_BINS),
        "per_sequence": [{"seq_id": s, **v} for s, v in sorted(acc.per_seq.items())],
        "failures": acc.failures,
    }


def print_summary(report: dict) -> None:
    o, le = report["overall"], report["localization_error_m"]
    print("\n=== Frustum POC: YOLO 2D -> LiDAR frustum -> 3D vs keyframe boxes ===")
    print(f"sequences={o['n_sequences']}  GT={o['n_gt']}  pred={o['n_pred']}")
    print(f"recall={o['recall']}  precision={o['precision']}  (TP={o['tp']} FN={o['fn']} FP={o['fp']})")
    print(f"localization error (m): median={le['median']}  mean={le['mean']}  p90={le['p90']}  "
          f"(n={le['n_matched']})")
    print("\n  recall by horizontal range:")
    for b in report["by_range_xy_m"]:
        print(f"    {b['bin']+' m':>12}  n={b['n_gt']:<5} recall={b['recall']}")
    print("\n  recall by #LiDAR points in GT box:")
    for b in report["by_lidar_points"]:
        print(f"    {b['bin']:>12}  n={b['n_gt']:<5} recall={b['recall']}")
    if report["failures"]:
        print(f"\n  {len(report['failures'])} sequence(s) skipped — see report['failures'].")


def run(detector, seq_ids, args, config) -> dict:
    acc = Accumulator()
    for i, seq_id in enumerate(seq_ids):
        seq_dir = SEQ_DIR / seq_id
        try:
            img_path, _ = keyframe_image_path(seq_dir)
            keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])
            scan_path, gap = nearest_lidar_scan(seq_dir / "lidar_velodyne", keyframe_ts)
            if img_path is None or scan_path is None or gap > MAX_LIDAR_GAP_S:
                acc.failures.append({"seq_id": seq_id, "reason": f"missing image/LiDAR (gap={gap:.3f})"})
                continue
            calib = load_calibration(seq_dir / "calibration.json")
            gts = load_keyframe_pedestrians(seq_dir)
            # Compensate the +37ms scan into the LiDAR frame at the keyframe instant, where the
            # GT boxes live (annotated on a keyframe-compensated cloud). Removes the ego-motion
            # offset that otherwise shows up as localization error.
            scan_ts = lidar_ts_from_filename(scan_path.name)
            xyzi = motion_compensate_to_keyframe(load_xyzi(scan_path), seq_dir, scan_ts, keyframe_ts)
            points = xyzi[:, :3]
            uv, valid = project_lidar_to_image(points, calib)
            pts_vis = points[valid]

            dets: List[Frustum3D] = []
            for box in detector(img_path):
                est = lift_box_to_3d(box.xyxy, uv, pts_vis, args.box_shrink, args.slab, args.min_pts)
                if est is not None:
                    center, n = est
                    dets.append(Frustum3D(center=center, score=box.score, n_points=n))

            matched, errors = match_greedy_3d(dets, gts, args.match_dist)
            acc.add(seq_id, gts, points, matched, errors, len(dets))
        except Exception as exc:  # continue-on-error
            acc.failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences")
    return build_report(acc, config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--match-dist", type=float, default=2.0, help="BEV centre match threshold (m)")
    ap.add_argument("--box-shrink", type=float, default=0.6, help="kept central width fraction of each 2D box")
    ap.add_argument("--slab", type=float, default=1.5, help="nearest-depth slab thickness (m)")
    ap.add_argument("--min-pts", type=int, default=3, help="min LiDAR points to lift a box")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    from eval_detector2d_recall import make_yolo  # noqa: E402
    detector = make_yolo(args.model, conf=args.conf, imgsz=args.imgsz)

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    config = {"detector": "frustum(yolo2d+lidar)", "model": args.model, "conf": args.conf,
              "imgsz": args.imgsz, "match_dist_m": args.match_dist, "box_shrink": args.box_shrink,
              "slab_m": args.slab, "min_pts": args.min_pts, "keyframe_compensated": True,
              "n_sequences_requested": len(seq_ids)}
    print(f"Running frustum POC: sequences={len(seq_ids)}")
    report = run(detector, seq_ids, args, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nReport written → {args.output}")


if __name__ == "__main__":
    main()
