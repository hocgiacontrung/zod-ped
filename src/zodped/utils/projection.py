"""Camera projection utilities for ZOD annotations.

Coordinate frame note
---------------------
Both `location_3d` annotation coordinates and LiDAR point cloud coordinates are
in the **LiDAR sensor frame** (not the vehicle ego frame). The two frames differ
by the LiDAR mount transform (~1.75m height, small rotation) encoded in
`calibration.json["FC"]["lidar_extrinsics"]`.

Verified on seq 000007: projecting via `inv(cam_ext) @ lid_ext` places pedestrian
centroids within ±35px of annotated 2D bbox centers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def load_calibration(calib_path: Union[str, Path]) -> Dict:
    with open(calib_path) as f:
        return json.load(f)


def get_T_cam_lidar(calib: Dict) -> np.ndarray:
    """4×4 transform: LiDAR sensor frame → front-camera frame.

    Both `location_3d` and `.npy` point clouds are in the LiDAR frame,
    so this is the only transform needed to project either into the image.
    """
    cam_ext = np.array(calib["FC"]["extrinsics"])    # T[ego←cam]
    lid_ext = np.array(calib["FC"]["lidar_extrinsics"])  # T[ego←lidar]
    return np.linalg.inv(cam_ext) @ lid_ext          # T[cam←lidar]


# ---------------------------------------------------------------------------
# Kannala-Brandt fisheye projection
# ---------------------------------------------------------------------------

def _kannala_distort(
    pts_cam: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Project camera-frame 3D points to pixel coordinates via Kannala model.

    Args:
        pts_cam:    (N, 3) points in camera frame (x right, y down, z forward).
        intrinsics: (3, 4) or (3, 3) camera matrix [fx 0 cx; 0 fy cy; 0 0 1].
        distortion: (4,) Kannala coefficients [k1, k2, k3, k4].

    Returns:
        (N, 2) pixel coordinates [u, v].
    """
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(r, z)
    t2 = theta ** 2
    td = theta * (1 + distortion[0] * t2
                    + distortion[1] * t2 ** 2
                    + distortion[2] * t2 ** 3
                    + distortion[3] * t2 ** 4)

    # avoid divide-by-zero for points on the optical axis
    safe_r = np.where(r < 1e-9, 1.0, r)
    scale = np.where(r < 1e-9, 0.0, td / safe_r)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = fx * scale * x + cx
    v = fy * scale * y + cy
    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def project_lidar_to_image(
    points: np.ndarray,
    calib: Dict,
    return_depth: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR-frame (or annotation `location_3d`) points into the front camera.

    Args:
        points:       (N, 3) coordinates in LiDAR sensor frame.
        calib:        Parsed `calibration.json` dict (top-level key "FC").
        return_depth: If True, append z_cam as a third column in the first return value.

    Returns:
        uv:    (M, 2) pixel coordinates of visible points. (M, 3) if return_depth.
        valid: (N,) boolean mask — True where the point projects inside the image
               and is in front of the camera.
    """
    T = get_T_cam_lidar(calib)
    intrinsics = np.array(calib["FC"]["intrinsics"])[:3, :3]
    distortion = np.array(calib["FC"]["distortion"])
    img_w, img_h = calib["FC"]["image_dimensions"]

    pts_hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    pts_cam = (T @ pts_hom.T).T[:, :3]

    in_front = pts_cam[:, 2] > 0
    uv_all = np.full((len(points), 2), np.nan)
    if in_front.any():
        uv_all[in_front] = _kannala_distort(pts_cam[in_front], intrinsics, distortion)

    in_image = (
        (uv_all[:, 0] >= 0) & (uv_all[:, 0] < img_w) &
        (uv_all[:, 1] >= 0) & (uv_all[:, 1] < img_h)
    )
    valid = in_front & in_image

    if return_depth:
        z_cam = np.where(valid, pts_cam[:, 2], np.nan)
        out = np.stack([uv_all[:, 0], uv_all[:, 1], z_cam], axis=-1)
    else:
        out = uv_all

    return out[valid], valid


def project_world_to_image(
    points_world: np.ndarray,
    calib: Dict,
    T_world_lidar: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-frame points into the front camera, given T[world←lidar] at the capture instant.

    Chains world → LiDAR frame → image. Use the KEYFRAME `T_world_lidar` to test world points against
    keyframe image-space annotations (e.g. the ego_road polygon).

    Args:
        points_world: (N, 3) coordinates in the world frame.
        calib:        parsed calibration.json dict.
        T_world_lidar: 4×4 T[world←lidar] at the capture instant (e.g. get_T_world_lidar at keyframe).

    Returns:
        uv:    (N, 2) pixel coordinates ALIGNED with points_world; NaN rows where the point is behind
               the camera or out of frame (not compressed — unlike project_lidar_to_image).
        valid: (N,) boolean mask, True where uv is finite.
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    pts_lidar = (np.linalg.inv(T_world_lidar) @ np.c_[points_world, np.ones(len(points_world))].T).T[:, :3]
    uv_packed, valid = project_lidar_to_image(pts_lidar, calib, return_depth=False)

    uv = np.full((len(points_world), 2), np.nan)
    uv[valid] = uv_packed
    return uv, valid
