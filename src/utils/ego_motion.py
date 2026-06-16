"""Ego-motion utilities for ZOD sequences.

ego_motion.json layout:
  timestamps:     list of Unix float timestamps (~19 Hz; same scale as LiDAR filenames)
  poses:          list of 4×4 SE(3) matrices, T[world←ego] at each timestamp
  velocities:     list of [vx, vy, vz] in ego frame
  origin_lat_lon: [lat, lon] of the sequence origin
"""

from __future__ import annotations

import json

from pathlib import Path

from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_ego_motion(path: Union[str, Path]) -> dict:
    with open(path) as f:
        em = json.load(f)
    return {
        "timestamps": np.array(em["timestamps"]),
        "poses": np.array(em["poses"]),
    }


def interpolate_pose(em: dict, t: float) -> np.ndarray:
    """Return 4×4 T[world←ego] at Unix timestamp t via SLERP + linear translation."""
    timestamps = em["timestamps"]
    poses = em["poses"]

    idx = int(np.searchsorted(timestamps, t, side="right"))
    idx = max(1, min(idx, len(timestamps) - 1))

    t0, t1 = timestamps[idx - 1], timestamps[idx]
    P0, P1 = poses[idx - 1], poses[idx]

    alpha = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)) if t1 > t0 else 0.0

    trans = (1.0 - alpha) * P0[:3, 3] + alpha * P1[:3, 3]

    rots = Rotation.from_matrix([P0[:3, :3], P1[:3, :3]])
    rot = Slerp([0.0, 1.0], rots)(alpha).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


def get_T_world_lidar(em: dict, lidar_extrinsics: np.ndarray, t: float) -> np.ndarray:
    """Return 4×4 T[world←lidar] at Unix timestamp t.

    lidar_extrinsics: 4×4 T[ego←lidar] from calibration.json["FC"]["lidar_extrinsics"].
    """
    return interpolate_pose(em, t) @ lidar_extrinsics


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4×4 homogeneous transform T to (N, 3) points. Returns (N, 3)."""
    pts_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (T @ pts_h.T).T[:, :3]
