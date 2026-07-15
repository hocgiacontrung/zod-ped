"""Frustum lift — a 2D detector box → a 3D position via in-frustum LiDAR depth.

This is the Step 1 MEASUREMENT source (see docs/PIPELINE.md "Direction & open options"): a
strong COCO 2D detector recalls pedestrians well where an off-the-shelf 3D detector does not, so
we recover the 3D *position* by intersecting each 2D box with the LiDAR. Both the bring-up POC
(`scripts/bringup_frustum_poc.py`) and the Step 1 driver (`scripts/01_generate_trajectories.py`) consume
this module, so the 2D→3D logic lives in exactly one place (mirrors `src/zodped/dataset/keyframe.py`).

The lift runs in the LiDAR sensor frame (project points to the image with the static cam←lidar
calibration, gather the points under the box, take the nearest-depth slab). `lift_image_detections`
then transforms only the resulting centre to the world frame, which is cheaper than compensating
the whole cloud and equivalent for a single point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# A per-frame candidate pool entry: (scan unix ts, world positions (K,3), per-candidate point counts (K,)).
FrameCandidates = Tuple[float, np.ndarray, np.ndarray]


def lift_box_to_3d(box_xyxy: np.ndarray, uv: np.ndarray, pts: np.ndarray,
                   box_shrink: float, slab_m: float, min_pts: int) -> Optional[Tuple[np.ndarray, int]]:
    """Estimate a 3D centre from the LiDAR points inside a (shrunk) 2D box.

    `uv` is the (M,2) pixel projection of the visible LiDAR points `pts` (M,3, LiDAR frame).
    We shrink the box toward its centre column (pedestrians are tall+thin; the box edges leak
    background), keep the points inside it, then take the NEAREST depth slab and its median.

    Returns (centre_lidar (3,), n_slab_points) or None if too few points fall in the frustum.
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


@dataclass
class FrustumLift:
    """A 2D detection lifted to a 3D centre in the world frame."""

    center_world: np.ndarray   # (3,) x, y, z [m], world frame
    n_points: int              # LiDAR points in the nearest-depth slab
    score: float               # source detector confidence


def lift_image_detections(
    boxes: Iterable,
    points_lidar: np.ndarray,
    calib: dict,
    T_world_lidar: np.ndarray,
    box_shrink: float,
    slab_m: float,
    min_pts: int,
) -> List[FrustumLift]:
    """Lift every 2D detection in one frame to a world-frame 3D centre.

    Args:
        boxes:         detections exposing ``.xyxy`` (4,) pixels and ``.score`` (float).
        points_lidar:  (N, 3) LiDAR points in the sensor frame for the scan paired with this image.
        calib:         parsed calibration.json (for the static cam←lidar projection).
        T_world_lidar: 4×4 T[world←lidar] at the scan time (lifts the centre out of the sensor frame).
        box_shrink, slab_m, min_pts: passed through to :func:`lift_box_to_3d`.

    Returns the successful lifts; boxes with too few in-frustum points are dropped.
    """
    from zodped.utils.projection import project_lidar_to_image

    uv, valid = project_lidar_to_image(points_lidar, calib)
    pts_vis = points_lidar[valid]

    lifts: List[FrustumLift] = []
    for box in boxes:
        est = lift_box_to_3d(box.xyxy, uv, pts_vis, box_shrink, slab_m, min_pts)
        if est is None:
            continue
        center_lidar, n = est
        center_world = (T_world_lidar @ np.append(center_lidar, 1.0))[:3]
        lifts.append(FrustumLift(center_world=center_world, n_points=n, score=float(box.score)))
    return lifts


def build_candidate_pool(
    seq_dir: Path,
    detector,
    em: dict,
    lidar_ext: np.ndarray,
    calib: dict,
    *,
    camera: str,
    max_gap: float,
    box_shrink: float,
    slab: float,
    min_pts: int,
) -> List[FrameCandidates]:
    """Lift every LiDAR scan's 2D detections to world-frame candidates (once per sequence).

    Scan-driven (LiDAR is the depth source): each scan is paired with the nearest camera image
    within `max_gap`; scans with no close image are skipped. Detector results are cached per image,
    since consecutive scans can share one image. The pool is pedestrian-independent — GOLD threads
    keyframe-anchored peds through it (scripts/01_generate_trajectories.py) and SILVER births tracks
    from the candidates GOLD leaves unclaimed (scripts/01b_generate_silver.py).
    """
    from zodped.dataset.keyframe import lidar_ts_from_filename, load_xyzi, parse_zod_ts
    from zodped.utils.ego_motion import get_T_world_lidar

    img_paths = sorted((seq_dir / camera).glob("*.jpg"))
    img_ts = np.array([parse_zod_ts(p.stem.rsplit("_", 1)[-1]) for p in img_paths])
    scans = sorted((seq_dir / "lidar_velodyne").glob("*.npy"))
    if not len(img_paths) or not scans:
        return []

    det_cache: dict = {}
    pool: List[FrameCandidates] = []
    for scan in scans:
        scan_ts = lidar_ts_from_filename(scan.name)
        j = int(np.argmin(np.abs(img_ts - scan_ts)))
        if abs(img_ts[j] - scan_ts) > max_gap:
            continue
        img_path = img_paths[j]
        if img_path not in det_cache:
            det_cache[img_path] = detector(img_path)
        boxes = det_cache[img_path]

        points = load_xyzi(scan)[:, :3].astype(np.float64)
        T_world_lidar = get_T_world_lidar(em, lidar_ext, scan_ts)
        lifts = lift_image_detections(boxes, points, calib, T_world_lidar, box_shrink, slab, min_pts)
        cand_world = np.array([lf.center_world for lf in lifts], dtype=np.float64).reshape(-1, 3)
        cand_n = np.array([lf.n_points for lf in lifts], dtype=int)
        pool.append((scan_ts, cand_world, cand_n))
    return pool
