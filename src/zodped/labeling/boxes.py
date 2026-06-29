"""Per-frame 3D box assembly for a smoothed pedestrian track.

The shipped GOLD box is the simplest construction that the keyframe GT-box gate validated
(docs/EXPERIMENTS_LOG.md "Boxfit cluster experiment"):

    centre = the tracked position,  size = the keyframe anchor extent (rigid per pedestrian),
    yaw    = the heading of the smoothed velocity.

The gate showed the two rejected alternatives are worse: a per-frame LiDAR-cluster centroid is no
better than the tracked (slab) centre and far less robust, and a cluster's measured extent / PCA
yaw are unusable (sparse → height underestimated, round cross-section → yaw random). So this module
needs no LiDAR — it works purely on the smoothed track plus the keyframe size.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _fill_yaw(yaw: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Fill yaw on stationary frames from the nearest moving frame; flag the source per frame."""
    n = len(yaw)
    source = ["velocity" if m else "static_fill" for m in moving]
    if not moving.any():
        return np.zeros(n), ["undefined"] * n
    moving_idx = np.where(moving)[0]
    out = yaw.copy()
    for i in range(n):
        if not moving[i]:
            out[i] = yaw[moving_idx[int(np.argmin(np.abs(moving_idx - i)))]]
    return out, source


def assemble_track_boxes(frames: List[dict], anchor_size_lwh: np.ndarray,
                         min_speed: float = 0.3) -> List[dict]:
    """Attach the shipped per-frame 3D box to an RTS-smoothed track (in place).

    Yaw comes from the smoothed velocity; stationary frames (speed < `min_speed`) inherit the
    nearest moving frame's heading, and a wholly stationary track is flagged `undefined`.
    """
    from zodped.dataset.keyframe import parse_zod_ts

    if not frames:
        return frames
    pos = np.array([f["position_world"] for f in frames], dtype=float)
    t = np.array([parse_zod_ts(f["timestamp"]) for f in frames])
    vel = np.gradient(pos, t, axis=0) if len(frames) > 1 else np.zeros_like(pos)

    speed_h = np.hypot(vel[:, 0], vel[:, 1])
    yaw, source = _fill_yaw(np.arctan2(vel[:, 1], vel[:, 0]), speed_h >= min_speed)
    size = np.asarray(anchor_size_lwh, dtype=float).tolist()

    for i, f in enumerate(frames):
        f["box"] = {
            "center_world": f["position_world"],
            "size_lwh": size,
            "yaw_world": float(yaw[i]),
            "yaw_source": source[i],
        }
    return frames
