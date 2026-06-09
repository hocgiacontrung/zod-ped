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


def get_T_cam_ego(calib: Dict) -> np.ndarray:
    """4×4 transform: vehicle ego frame → front-camera frame."""
    return np.linalg.inv(np.array(calib["FC"]["extrinsics"]))


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


def is_point_in_road_polygon(
    points: np.ndarray,
    road_polygons: list,
    calib: Dict,
) -> np.ndarray:
    """Test whether LiDAR-frame 3D points fall inside the ego_road polygon (image space).

    `ego_road.json` vertices are in image pixel coordinates. We project the 3D points
    first, then do a 2D point-in-polygon test.

    Args:
        points:        (N, 3) coordinates in LiDAR sensor frame.
        road_polygons: list of polygon dicts from `ego_road.json`
                       (each has `geometry.coordinates` as a list of rings).
        calib:         Parsed `calibration.json` dict.

    Returns:
        (N,) boolean mask — True where the projected point is inside any road polygon
        AND is visible in the image.
    """
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        raise ImportError("shapely is required for polygon containment checks: pip install shapely")

    uv, valid = project_lidar_to_image(points, calib)

    polys = []
    for entry in road_polygons:
        outer_ring = entry["geometry"]["coordinates"][0]
        polys.append(Polygon(outer_ring))

    result = np.zeros(len(points), dtype=bool)
    valid_indices = np.where(valid)[0]
    for i, idx in enumerate(valid_indices):
        pt = Point(uv[i, 0], uv[i, 1])
        if any(poly.contains(pt) for poly in polys):
            result[idx] = True

    return result
