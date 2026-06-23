"""Keyframe-scan and ground-truth loaders shared across Step 1 (trajectory generation) and its
bring-up experiments.

ZOD annotates a single keyframe per 20s clip. These helpers locate the LiDAR scan nearest the
keyframe and read the verified pedestrian 3D boxes from it. Both the frustum POC
(`scripts/frustum_poc.py`) and the Step 1 driver consume this module, so the keyframe
logic lives in exactly one place. This module is the single source of truth for the
keyframe (seq, pedestrian) set — it selects pedestrians directly from the annotations
(`class == "Pedestrian"` with `location_3d`), so nothing upstream needs to pre-enumerate them.

Frame conventions (verified during exploration, seq 000007):
  * LiDAR `.npy` is a STRUCTURED array — fields x, y, z, timestamp, intensity, diode_index.
  * `location_3d` and box sizes in object_detection.json are in the LiDAR SENSOR frame, the same
    frame as the raw points — so GT boxes and points can be compared directly, no transform.
  * LiDAR filenames are UTC timestamps; match the keyframe by timestamp, never by index.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Max gap tolerated between the keyframe time and the nearest LiDAR scan.
MAX_LIDAR_GAP_S = 0.055


@dataclass(frozen=True)
class GtBox:
    """A verified keyframe pedestrian box, in the LiDAR sensor frame."""

    uuid: str
    center: np.ndarray          # (3,) x, y, z  [m]
    size: np.ndarray            # (3,) length, width, height  [m]
    occlusion: Optional[str]    # ZOD occlusion_ratio string, or None

    @property
    def range_xy(self) -> float:
        """Horizontal distance from the sensor origin — the axis recall degrades along."""
        return float(np.hypot(self.center[0], self.center[1]))


def parse_zod_ts(ts_str: str) -> float:
    """Parse a ZOD ISO-8601 timestamp (e.g. info.json['keyframe_time']) to Unix seconds."""
    return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def lidar_ts_from_filename(name: str) -> float:
    """Parse the Unix timestamp from a LiDAR filename of the form ``*_<ISO>.npy``."""
    iso = name.rsplit("_", 1)[-1][: -len(".npy")]
    return parse_zod_ts(iso)


def nearest_lidar_scan(lidar_dir: Path, target_ts: float) -> Tuple[Optional[Path], float]:
    """Return (scan nearest to target_ts, |gap| in seconds), or (None, inf) if no scans."""
    files = sorted(lidar_dir.glob("*.npy"))
    if not files:
        return None, float("inf")
    gaps = np.array([abs(lidar_ts_from_filename(f.name) - target_ts) for f in files])
    i = int(np.argmin(gaps))
    return files[i], float(gaps[i])


def load_xyzi(scan_path: Path) -> np.ndarray:
    """Load a structured LiDAR `.npy` as a contiguous (N, 4) float32 array [x, y, z, intensity]."""
    cloud = np.load(scan_path)
    return np.stack(
        [cloud["x"], cloud["y"], cloud["z"], cloud["intensity"]], axis=1
    ).astype(np.float32)


def motion_compensate_to_keyframe(
    xyzi: np.ndarray, seq_dir: Path, scan_ts: float, target_ts: float
) -> np.ndarray:
    """Scan-wise motion-compensate a LiDAR cloud into the LiDAR frame at ``target_ts``.

    The keyframe 3D boxes are annotated on a cloud compensated to ``keyframe_time``, but the
    on-disk ``.npy`` is only POINTWISE-deskewed to its own core timestamp (~37 ms before the
    keyframe). Comparing a raw scan to those boxes leaves an uncompensated ego-motion offset
    (~0.2-0.5 m at city speed) that inflates localization error and recall-at-range. This lifts
    every point into the ego pose at ``target_ts`` so measurements and GT share one instant.

    Mirrors the devkit's ``motion_compensate_scanwise``; we reuse its ``EgoMotion`` interpolator
    (linear translation + slerp) and apply the frame math explicitly:

        p_lidar(target) = inv(T_ego_lidar) @ inv(pose_target) @ pose_scan @ T_ego_lidar @ p_lidar(scan)

    Only x, y, z are transformed; the intensity column is passed through unchanged.
    """
    from zod.data_classes.ego_motion import EgoMotion  # devkit interpolation (slerp); local import

    ego = EgoMotion.from_json_path(str(seq_dir / "ego_motion.json"))
    pose_scan = ego.get_poses(scan_ts)        # T[world<-ego] at scan time
    pose_target = ego.get_poses(target_ts)    # T[world<-ego] at keyframe time
    odometry = np.linalg.inv(pose_target) @ pose_scan  # T[ego(target)<-ego(scan)]

    calib = json.loads((seq_dir / "calibration.json").read_text())
    t_ego_lidar = np.asarray(calib["FC"]["lidar_extrinsics"])  # T[ego<-lidar]
    transform = np.linalg.inv(t_ego_lidar) @ odometry @ t_ego_lidar  # T[lidar(target)<-lidar(scan)]

    out = xyzi.copy()
    pts_h = np.c_[xyzi[:, :3], np.ones(len(xyzi))]
    out[:, :3] = (pts_h @ transform.T)[:, :3]
    return out


def load_keyframe_pedestrians(seq_dir: Path) -> List[GtBox]:
    """Read 3D pedestrian boxes from a sequence's keyframe annotation.

    Keeps only entries with ``class == "Pedestrian"`` that carry ``location_3d`` (2D-only
    pedestrians cannot anchor a 3D box and are skipped). This selection is the single source
    of truth for the GOLD (seq, pedestrian) set.
    """
    det_path = seq_dir / "annotations" / "object_detection.json"
    detections = json.loads(det_path.read_text())

    boxes: List[GtBox] = []
    for d in detections:
        pr = d["properties"]
        if pr.get("class") != "Pedestrian" or "location_3d" not in pr:
            continue
        boxes.append(
            GtBox(
                uuid=pr["annotation_uuid"],
                center=np.array(pr["location_3d"]["coordinates"], dtype=np.float64),
                size=np.array(
                    [pr["size_3d_length"], pr["size_3d_width"], pr["size_3d_height"]],
                    dtype=np.float64,
                ),
                occlusion=pr.get("occlusion_ratio"),
            )
        )
    return boxes


def points_in_box(points_xyz: np.ndarray, box: GtBox, margin: float = 0.0) -> int:
    """Count points inside an axis-aligned box around ``box.center`` (sparsity proxy).

    Axis-aligned (yaw ignored) on purpose: it mirrors the tracker's gate and is a stable,
    cheap measure of how many returns a pedestrian generates — which is what drives detector
    recall, independent of exact box orientation.
    """
    half = box.size / 2.0 + margin
    inside = np.all(np.abs(points_xyz[:, :3] - box.center) <= half, axis=1)
    return int(inside.sum())


# ---------------------------------------------------------------------------
# 2D (image-plane) keyframe ground truth — for the 2D-detector bring-up gate.
# ---------------------------------------------------------------------------
# The keyframe is a CAMERA frame: `info.json["keyframe_time"]` matches a camera image
# exactly (0 ms), while the nearest LiDAR scan is ~37-55 ms away. So a 2D detector run on
# the keyframe image compares to GT at zero temporal offset — cleaner than the 3D gate.
# Each pedestrian's `geometry` is a MultiPoint of the box's 4 extreme points (top/right/
# bottom/left); min/max over them gives the axis-aligned box. 2D GT exists for ALL
# pedestrians, including the "2D-only" ones the 3D pipeline skips (no `location_3d`).


@dataclass(frozen=True)
class GtBox2D:
    """A verified keyframe pedestrian box in image pixels (x1, y1, x2, y2)."""

    uuid: str
    xyxy: np.ndarray            # (4,) x1, y1, x2, y2  [px]
    occlusion: Optional[str]    # ZOD occlusion_ratio string, or None
    has_3d: bool                # whether this pedestrian also carries location_3d

    @property
    def height(self) -> float:
        return float(self.xyxy[3] - self.xyxy[1])

    @property
    def width(self) -> float:
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def area(self) -> float:
        return self.width * self.height


def keyframe_image_path(seq_dir: Path, camera: str = "camera_front_blur") -> Tuple[Optional[Path], float]:
    """Return (camera image nearest the keyframe, |gap| in seconds), or (None, inf)."""
    keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])
    files = sorted((seq_dir / camera).glob("*.jpg"))
    if not files:
        return None, float("inf")
    gaps = np.array([abs(parse_zod_ts(f.stem.rsplit("_", 1)[-1]) - keyframe_ts) for f in files])
    i = int(np.argmin(gaps))
    return files[i], float(gaps[i])


def load_keyframe_pedestrians_2d(seq_dir: Path) -> List[GtBox2D]:
    """Read 2D pedestrian boxes from a sequence's keyframe annotation (image pixels).

    Keeps every ``class == "Pedestrian"`` entry whose ``geometry`` is a MultiPoint with at
    least two points (so an axis-aligned box can be formed) — unlike the 3D loader, this
    does NOT require ``location_3d``.
    """
    detections = json.loads((seq_dir / "annotations" / "object_detection.json").read_text())

    boxes: List[GtBox2D] = []
    for d in detections:
        pr = d["properties"]
        geom = d.get("geometry") or {}
        if pr.get("class") != "Pedestrian" or geom.get("type") != "MultiPoint":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        c = np.asarray(coords, dtype=np.float64)
        boxes.append(
            GtBox2D(
                uuid=pr.get("annotation_uuid"),
                xyxy=np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]),
                occlusion=pr.get("occlusion_ratio"),
                has_3d="location_3d" in pr,
            )
        )
    return boxes
