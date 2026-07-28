"""Persisted 2D-pose cache — run the pose model once, reuse forever (Step 0b).

PedGraph+ (committee member #2) scores crossing from POSE KEYPOINTS, not boxes, so it needs a
per-frame skeleton for each pedestrian. Running the pose model is the expensive part (same order as
the 2D detector); everything downstream — windowing keypoints, IoU-matching a skeleton to a known
box track, PedGraph inference — is cheap. This module persists the per-image person skeletons so
that association and scoring can iterate in minutes instead of paying for the pose model each time.

Mirrors `detection_cache.py` on purpose: same cache-once philosophy, same filename keying, same
load-time `min_conf` floor. Invalidate the cache — re-run `scripts/00b_extract_pose.py` — only when
the pose model GEOMETRY changes (model / imgsz), since those change the keypoints themselves. `conf`
is NOT an invalidator: cache at a low floor and a run raises the threshold for free at load time.

The pose model finds people on its own; it does not know WHICH skeleton is the pedestrian we track.
`best_pose_for_box` closes that gap: given a known 2D box (the projected-3D box from Step 3, or a
detector box), it returns the cached skeleton whose person box overlaps it most.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from zodped.labeling.pose_estimator import PersonPose

SCHEMA = "pose/v1"


def cache_path(pose_dir: Path, seq_id: str) -> Path:
    """Per-sequence cache file path."""
    return pose_dir / f"{seq_id}.json"


def write_poses(path: Path, meta: dict, per_image: Dict[str, List[PersonPose]]) -> None:
    """Persist one sequence's skeletons.

    `per_image` maps image filename -> its `PersonPose` list. Each pose keeps its person box (for
    downstream IoU association) and 17 COCO keypoints as (x, y, conf) rows.
    """
    images = {
        fn: [{"box": [float(p.xyxy[0]), float(p.xyxy[1]), float(p.xyxy[2]), float(p.xyxy[3]),
                      float(p.score)],
              "kpts": [[float(x), float(y), float(c)] for x, y, c in p.keypoints]}
             for p in poses]
        for fn, poses in per_image.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "meta": meta, "images": images}))


def read_doc(path: Path) -> dict:
    """Load one sequence's raw cache document (schema / meta / images) without decoding poses."""
    return json.loads(Path(path).read_text())


def decode_poses(doc: dict, min_conf: float = 0.0) -> Dict[str, List[PersonPose]]:
    """Decode a cache document to {filename: [PersonPose, ...]}, dropping people below `min_conf`.

    `min_conf` is the run's person-confidence threshold applied at LOAD time: the cache is built once
    at a low floor (``meta["conf"]``) so a run can raise the threshold for free. Requesting a
    threshold below the cached floor cannot recover skeletons that were never stored.
    """
    return {
        fn: [PersonPose(xyxy=np.asarray(row["box"][:4], dtype=np.float64),
                        score=float(row["box"][4]),
                        keypoints=np.asarray(row["kpts"], dtype=np.float64))
             for row in rows if row["box"][4] >= min_conf]
        for fn, rows in doc["images"].items()
    }


def load_poses(path: Path, min_conf: float = 0.0) -> Dict[str, List[PersonPose]]:
    """Read + decode a per-sequence cache in one call (read_doc + decode_poses)."""
    return decode_poses(read_doc(path), min_conf)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes (0.0 if they do not overlap)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def best_pose_for_box(poses: List[PersonPose], box_xyxy: np.ndarray,
                      min_iou: float = 0.3) -> Optional[PersonPose]:
    """Return the skeleton whose person box best overlaps `box_xyxy`, or None below `min_iou`.

    This is the association step that grounds a free-floating pose skeleton to the pedestrian we
    already track: the pose model finds every person in the frame, and we keep the one that lines up
    with the known 2D box. `min_iou` guards against grabbing a neighbouring pedestrian when ours was
    missed (returns None so the caller records a gap rather than a wrong skeleton).
    """
    if not poses:
        return None
    box = np.asarray(box_xyxy, dtype=np.float64)
    best, best_iou = None, min_iou
    for p in poses:
        i = _iou(p.xyxy, box)
        if i >= best_iou:
            best, best_iou = p, i
    return best
