"""Trajectory bring-up gate #1 — validate the compensate-before-associate transform chain.

The detector architecture (and the tracker) lift every scan's points to a single world frame via
`p_world = pose(t) @ T_ego_lidar @ p_lidar`. If that chain is wrong (pose interpolation,
extrinsics, or frame conventions), ego motion leaks into the trajectories and every downstream
metric is poisoned — silently.

Idea: a STATIC object (sign / signal / pole) must stay PINNED in the world frame as the ego moves.
We seed on a static keyframe box, then for each nearby scan compute the gate-centroid of points
around the fixed seed position, two ways:
  * WORLD frame (compensated)   — points lifted by the per-scan pose; centroid should barely move.
  * LiDAR frame (uncompensated) — points left in the sensor frame; centroid drifts as the ego moves.
PASS = world spread is small in absolute terms AND much smaller than the uncompensated spread (and
than the ego's own displacement over the window).

Usage:
    python scripts/validate_world_frame.py                 # seq 000007 (verified)
    python scripts/validate_world_frame.py --seq 000123 --window 1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.keyframe import (  # noqa: E402
    lidar_ts_from_filename,
    load_xyzi,
    parse_zod_ts,
)
from utils.ego_motion import get_T_world_lidar, interpolate_pose, load_ego_motion, transform_points  # noqa: E402
from utils.projection import load_calibration  # noqa: E402

SEQ_DIR = ROOT / "data" / "raw" / "sequences"

# Classes that move — never valid as a static reference.
DYNAMIC_CLASSES = {"Pedestrian", "Vehicle", "VulnerableVehicle", "Animal"}

PASS_WORLD_STD_M = 0.30   # world-frame centroid std must stay tighter than this
PASS_RATIO = 3.0          # ...and be at least this many x tighter than the uncompensated std


def pick_static_target(seq_dir: Path, points_lidar: np.ndarray, margin: float) -> Optional[dict]:
    """Choose the static keyframe object with the MOST LiDAR points (strongest signal)."""
    detections = json.loads((seq_dir / "annotations" / "object_detection.json").read_text())
    best, best_n = None, 0
    for d in detections:
        pr = d["properties"]
        if pr.get("class") in DYNAMIC_CLASSES or "location_3d" not in pr:
            continue
        center = np.array(pr["location_3d"]["coordinates"], dtype=np.float64)
        size = np.array([pr["size_3d_length"], pr["size_3d_width"], pr["size_3d_height"]])
        half = size / 2.0 + margin
        n = int(np.all(np.abs(points_lidar[:, :3] - center) <= half, axis=1).sum())
        if n > best_n:
            best, best_n = {"class": pr.get("class"), "center": center, "size": size, "n": n}, n
    return best


def gate_centroid(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    inside = np.all(np.abs(points[:, :3] - center) <= half, axis=1)
    n = int(inside.sum())
    return (points[inside, :3].mean(axis=0), n) if n else (None, 0)


def _spread(centroids: List[np.ndarray]) -> Optional[float]:
    """Std magnitude (norm of per-axis std) of a list of centroids, or None if < 2 samples."""
    if len(centroids) < 2:
        return None
    return float(np.linalg.norm(np.std(np.array(centroids), axis=0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="000007")
    ap.add_argument("--window", type=float, default=1.0, help="+/- seconds around the keyframe")
    ap.add_argument("--margin", type=float, default=1.0, help="gate half-margin beyond the box (m)")
    args = ap.parse_args()

    seq_dir = SEQ_DIR / args.seq
    keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    em = load_ego_motion(seq_dir / "ego_motion.json")

    scans = sorted((seq_dir / "lidar_velodyne").glob("*.npy"))
    window_scans = [s for s in scans if abs(lidar_ts_from_filename(s.name) - keyframe_ts) <= args.window]
    if len(window_scans) < 2:
        sys.exit(f"Not enough scans within +/-{args.window}s of the keyframe (got {len(window_scans)}).")

    # Seed on the keyframe scan.
    kf_scan = min(scans, key=lambda s: abs(lidar_ts_from_filename(s.name) - keyframe_ts))
    kf_points = load_xyzi(kf_scan)
    target = pick_static_target(seq_dir, kf_points, args.margin)
    if target is None:
        sys.exit("No static keyframe object with location_3d found in this sequence.")

    seed_lidar = target["center"]
    half = target["size"] / 2.0 + args.margin
    T_kf = get_T_world_lidar(em, lidar_ext, keyframe_ts)
    seed_world = transform_points(T_kf, seed_lidar[None, :])[0]

    print(f"seq {args.seq}: static target = {target['class']} ({target['n']} pts at keyframe)")
    print(f"  scans in window: {len(window_scans)}  (+/-{args.window}s)")

    world_centroids, lidar_centroids = [], []
    for scan in window_scans:
        ts = lidar_ts_from_filename(scan.name)
        pts = load_xyzi(scan)
        pts_world = transform_points(get_T_world_lidar(em, lidar_ext, ts), pts[:, :3])
        c_world, _ = gate_centroid(pts_world, seed_world, half)
        c_lidar, _ = gate_centroid(pts, seed_lidar, half)   # uncompensated baseline
        if c_world is not None:
            world_centroids.append(c_world)
        if c_lidar is not None:
            lidar_centroids.append(c_lidar)

    # Ego displacement across the window (context: how much motion we're compensating for).
    ego_disp = float(np.linalg.norm(
        interpolate_pose(em, lidar_ts_from_filename(window_scans[-1].name))[:3, 3]
        - interpolate_pose(em, lidar_ts_from_filename(window_scans[0].name))[:3, 3]
    ))

    world_std = _spread(world_centroids)
    lidar_std = _spread(lidar_centroids)
    print(f"  ego displacement over window : {ego_disp:.2f} m")
    print(f"  WORLD-frame centroid std     : {world_std:.3f} m  (compensated; want small)" if world_std is not None
          else "  WORLD-frame centroid std     : n/a (object not gated across scans)")
    print(f"  LiDAR-frame centroid std     : {lidar_std:.3f} m  (uncompensated baseline)" if lidar_std is not None
          else "  LiDAR-frame centroid std     : n/a")

    ok = (
        world_std is not None
        and world_std <= PASS_WORLD_STD_M
        and (lidar_std is None or world_std * PASS_RATIO <= lidar_std)
    )
    print(f"\n  {'PASS' if ok else 'FAIL'} — "
          f"world std {'<=' if ok else '>'} {PASS_WORLD_STD_M} m"
          + ("" if lidar_std is None else f" and >= {PASS_RATIO}x tighter than uncompensated"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
