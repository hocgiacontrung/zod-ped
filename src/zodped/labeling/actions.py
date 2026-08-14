"""Track-level ACTION labeling — Step 2 (see docs/PIPELINE.md "Pipeline Overview").

ACTION is a geometric fact about the WHOLE track: did this pedestrian cross the ego road, and WHEN.
Computed once per track: pure geometry over the trajectory Step 1 produced.

The crossing ACTION is `crosses_ego_road` — does the track land on the ego_road drivable-surface
polygon, the JAAD/PIE "crossing the roadway" notion. The polygon is image-pixel and annotated at
the KEYFRAME, so each world point is projected through the keyframe camera and tested for
containment (FOV/range-limited by construction). Per-window intent is derived downstream in Step 3.

Two known defects, measured and deliberately NOT fixed (the splits are frozen and the snapshot is
checksummed, so changing the rule would move every reported number) — EXPERIMENTS_LOG 2026-08-11:
the projected point is the box CENTRE rather than the feet, and `EgoRoad_Debris` polygons count as
road. Both inflate false positives.

The ego-corridor swept-path signal that used to be the primary label here was benched 2026-07-08
and removed; it is recoverable at git tag `experiments/committee-pose-bringup`.

EMPTY tracks (no real detection beyond the anchor; the trajectory is pure Kalman coast) carry no
usable motion, so their action is `undetermined` — kept and flagged: a verified pedestrian we could
not track.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from zodped.utils.projection import project_world_to_image


def _points_in_polygon(points_uv: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test. NaN points test False.

    points_uv: (N, 2) pixel coordinates; polygon: (M, 2) vertices. Returns (N,) bool.
    """
    x, y = points_uv[:, 0], points_uv[:, 1]
    inside = np.zeros(len(points_uv), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        straddles = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= straddles & (x < x_cross)
        j = i
    return inside & np.isfinite(x)


def points_in_any_polygon(points_uv: np.ndarray, polygons: Sequence[np.ndarray]) -> np.ndarray:
    """Union membership: True where a pixel point lies inside ANY of the given polygons.

    Shared by the Step-2 road test and the Step-3 road-surface extraction, so the two stages
    agree by construction on what counts as "on the ego road".
    """
    inside = np.zeros(len(points_uv), dtype=bool)
    for poly in polygons:
        inside |= _points_in_polygon(points_uv, poly)
    return inside


def _road_action(
    frames: Sequence[dict],
    T_world_lidar_keyframe: np.ndarray,
    calib: dict,
    polygons: Sequence[np.ndarray],
) -> dict:
    """Test whether the track crosses the ego_road polygon, via the keyframe camera."""
    positions = np.array([f["position_world"] for f in frames], dtype=np.float64)
    uv, valid = project_world_to_image(positions, calib, T_world_lidar_keyframe)

    on_road = points_in_any_polygon(uv, polygons)

    crosses = bool(on_road.any())
    idx = int(np.argmax(on_road)) if crosses else None
    return {
        "crosses_ego_road": crosses,
        "crossing_frame_timestamp": frames[idx]["timestamp"] if crosses else None,
        "crossing_observed": bool(frames[idx]["in_observation"]) if crosses else None,
        "road_frames_in_image": int(valid.sum()),
    }


def _n_real_observations(frames: Sequence[dict]) -> int:
    """Count real detections (in-gate, non-anchor). Zero ⇒ the track is anchor + coast only."""
    return sum(f["in_observation"] and f["tracking_method"] in ("forward", "backward") for f in frames)


def label_track_action(
    trajectory: dict,
    calib: dict,
    ego_road_polygons: Optional[List[np.ndarray]],
    T_world_lidar_keyframe: np.ndarray,
) -> dict:
    """Label one track's crossing ACTION (feet on the ego road). Returns the record (ready to serialise).

    Args:
        trajectory:             a Step-1 trajectory doc (data/processed/trajectories/*.json).
        calib:                  parsed calibration.json for the sequence.
        ego_road_polygons:      ego_road vertices as a list of (M, 2) pixel arrays, or None if the
                                sequence has no ego_road annotation (→ crosses_ego_road = null).
        T_world_lidar_keyframe: T[world←lidar] at the keyframe (get_T_world_lidar at the keyframe ts);
                                used to project world points into the keyframe camera.

    EMPTY tracks (no real observation) → status="undetermined", all label fields null.
    """
    frames = trajectory["frames"]
    n_real = _n_real_observations(frames)

    record = {
        "schema": "action/v0.3",
        "sequence_id": trajectory["sequence_id"],
        "pedestrian_id": trajectory["pedestrian_id"],
        "is_in_gold_standard": trajectory.get("is_in_gold_standard", True),
        "label_confidence_tier": trajectory.get("label_confidence_tier", "high"),
        "method": "rule_based_geometry",
    }

    # Undetermined: no real motion to label a crossing from — keep the ped, null the labels.
    if not frames or n_real == 0:
        record.update({
            "status": "undetermined",
            "crosses_ego_road": None,
            "crossing_frame_timestamp": None,
            "diagnostics": {"n_frames": len(frames), "n_real_obs": n_real},
        })
        return record

    road = (_road_action(frames, T_world_lidar_keyframe, calib, ego_road_polygons)
            if ego_road_polygons else
            {"crosses_ego_road": None, "crossing_frame_timestamp": None,
             "crossing_observed": None, "road_frames_in_image": 0})

    record.update({
        "status": "determined",
        "crosses_ego_road": road["crosses_ego_road"],
        "crossing_frame_timestamp": road["crossing_frame_timestamp"],
        "diagnostics": {
            "n_frames": len(frames),
            "n_real_obs": n_real,
            "crossing_observed": road["crossing_observed"],
            "road_frames_in_image": road["road_frames_in_image"],
        },
    })
    return record


def load_ego_road_polygons(ego_road_json: list) -> List[np.ndarray]:
    """Extract ego_road polygons as a list of (M, 2) pixel-vertex arrays from the parsed JSON."""
    return [np.asarray(poly["geometry"]["coordinates"][0], dtype=np.float64) for poly in ego_road_json]
