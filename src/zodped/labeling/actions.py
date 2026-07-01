"""Track-level ACTION labeling — Step 2 (see docs/PIPELINE.md "Pipeline Overview").

ACTION is a geometric fact about the WHOLE track: did this pedestrian cross, and WHEN. Computed once per track: pure geometry over the trajectory Step 1 produced. 
The forward-looking per-window INTENT label (Step 3) is *derived* from this action's crossing onset (`t_c`); 
the two are kept strictly separate so action (what happened) is never conflated with intent (what we predict before it happens).

Two labels, by design (docs/PIPELINE.md):
  * crosses_ego_corridor (PRIMARY) — does the track enter the ego vehicle's SWEPT PATH: the ribbon of
    half-width `corridor_half_width_m` around the ego's ACTUAL world-frame trajectory (ego_motion.poses),
    within `corridor_lookahead_m` of arc-length ahead of the ego (≈ JAAD/PIE "crossing in front of the ego" notion).
    The ribbon follows the road through turns, doesn't falsely flag stationary bystanders (see _corridor_action + docs/EXPERIMENTS_LOG.md).
  * crosses_ego_road (COMPLEMENT) — does the track cross the ego_road drivable-surface polygon. The
    polygon is image-pixel and annotated at the keyframe, so each world point is projected through
    the KEYFRAME camera and tested for containment. FOV/range-limited by construction (≈ JAAD's broader "crossing the roadway").

EMPTY tracks (no real detection beyond the anchor; the trajectory is pure Kalman coast) carry no 
usable motion, so their action is `undetermined` — never forced to a crossing decision. They are
kept and flagged: a verified pedestrian we could not track.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts
from zodped.utils.projection import project_world_to_image

# --- Corridor defaults (mirror configs/; see docs/PIPELINE.md) -----------------------
CORRIDOR_HALF_WIDTH_M = 1.5     # m, half-width of the ego swept-path ribbon (≈ a 3 m lane: ego ~1.9 m + margin)
CORRIDOR_LOOKAHEAD_M = 50.0     # m, max arc-length ahead counted as "in front" (beyond it the lift is unreliable)
CORRIDOR_MIN_FORWARD_M = 0.0    # m, near edge of the ribbon; ≥ 0 keeps it strictly ahead of the ego
_MIN_PATH_LENGTH_M = 2.0        # m, below this the ego is ~stationary → no meaningful swept path (no crossing)


def _ego_world_path(em: dict) -> tuple:
    """Ego trajectory as a world-frame polyline: (xy vertices, timestamps, cumulative arc-length, segments)."""
    poses = np.asarray(em["poses"])
    xy = poses[:, :2, 3]
    ts = np.asarray(em["timestamps"])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return xy, ts, s, seg


def _project_to_path(p_xy: np.ndarray, xy: np.ndarray, s: np.ndarray, seg: np.ndarray) -> tuple:
    """Nearest point on the ego polyline to p_xy: (perp distance, arc-length at foot, signed side).

    side = +1 when the ped is left of the ego's direction of travel, -1 when right (a diagnostic for
    whether the ped actually traversed the path rather than walking alongside it).
    """
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(((p_xy - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-12), 0.0, 1.0)
    foot = a + t[:, None] * ab
    diff = p_xy - foot
    d = np.hypot(diff[:, 0], diff[:, 1])
    k = int(d.argmin())
    arclen = s[k] + t[k] * seg[k]
    side = np.sign(ab[k, 0] * diff[k, 1] - ab[k, 1] * diff[k, 0])
    return d[k], arclen, float(side)


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


def _corridor_action(
    frames: Sequence[dict],
    em: dict,
    half_width_m: float,
    lookahead_m: float,
    min_forward_m: float,
) -> dict:
    """Detect entry into the ego's CURVED swept path over the whole track. Returns the sub-record.

    The corridor is the ego's ACTUAL trajectory — a ribbon of half-width `half_width_m` around the
    world-frame path from ego_motion.poses — NOT a straight strip ahead of the instantaneous heading.
    A straight strip rotates with the ego and, over a turn, sweeps across the pavement, falsely
    flagging stationary bystanders (verified: seq 000041, a 90° turn, flagged 2 non-crossers). The
    swept path follows the road through the turn, so those bystanders stay outside it. At each ped
    frame the ped's world position is projected onto the path; it is IN the corridor when that foot
    lies `min_forward_m..lookahead_m` of arc-length AHEAD of the ego and within the half-width.
    """
    xy, ts, s, seg = _ego_world_path(em)
    n = len(frames)
    perp = np.full(n, np.nan)        # perpendicular distance to the ego path
    fwd = np.full(n, np.nan)         # arc-length of the ped's foot ahead of the ego
    side = np.zeros(n)               # +left / -right of travel
    ego_dist = np.full(n, np.nan)    # euclidean ego→ped (horizontal)

    stationary = xy.shape[0] < 2 or s[-1] < _MIN_PATH_LENGTH_M
    if not stationary:
        for i, f in enumerate(frames):
            t = parse_zod_ts(f["timestamp"])
            p = np.asarray(f["position_world"])[:2]
            perp[i], s_foot, side[i] = _project_to_path(p, xy, s, seg)
            fwd[i] = s_foot - float(np.interp(t, ts, s))
            ego_xy = np.array([np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1])])
            ego_dist[i] = float(np.hypot(*(p - ego_xy)))

    ahead = (fwd >= min_forward_m) & (fwd <= lookahead_m)
    in_corridor = ahead & (perp <= half_width_m)

    crosses = bool(in_corridor.any())
    idx = int(np.argmax(in_corridor)) if crosses else None

    # Side change among in-front frames = the ped traversed the path (true crossing, not merely
    # walking alongside it). Diagnostic only, NOT a gate (occlusion can hide one side).
    side_ahead = side[ahead][perp[ahead] > 1e-6] if ahead.any() else np.array([])
    sign_change = bool(side_ahead.size and side_ahead.min() < 0.0 < side_ahead.max())

    return {
        "crosses_ego_corridor": crosses,
        "crossing_frame_timestamp": frames[idx]["timestamp"] if crosses else None,
        "ego_distance_at_crossing_m": round(float(ego_dist[idx]), 3) if crosses else None,
        "crossing_observed": bool(frames[idx]["in_observation"]) if crosses else None,
        "n_frames_ahead": int(ahead.sum()),
        "min_lateral_dist_m": round(float(perp[ahead].min()), 3) if ahead.any() else None,
        "lateral_sign_change": sign_change,
    }


def _road_action(
    frames: Sequence[dict],
    T_world_lidar_keyframe: np.ndarray,
    calib: dict,
    polygons: Sequence[np.ndarray],
) -> dict:
    """Test whether the track crosses the ego_road polygon, via the keyframe camera."""
    positions = np.array([f["position_world"] for f in frames], dtype=np.float64)
    uv, valid = project_world_to_image(positions, calib, T_world_lidar_keyframe)

    on_road = np.zeros(len(positions), dtype=bool)
    for poly in polygons:
        on_road |= _points_in_polygon(uv, poly)

    return {
        "crosses_ego_road": bool(on_road.any()),
        "road_frames_in_image": int(valid.sum()),
    }


def _n_real_observations(frames: Sequence[dict]) -> int:
    """Count real detections (in-gate, non-anchor). Zero ⇒ the track is anchor + coast only."""
    return sum(f["in_observation"] and f["tracking_method"] in ("forward", "backward") for f in frames)


def label_track_action(
    trajectory: dict,
    em: dict,
    calib: dict,
    ego_road_polygons: Optional[List[np.ndarray]],
    T_world_lidar_keyframe: np.ndarray,
    half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    lookahead_m: float = CORRIDOR_LOOKAHEAD_M,
    min_forward_m: float = CORRIDOR_MIN_FORWARD_M,
) -> dict:
    """Label one GOLD track's crossing ACTION. Returns the action record (ready to serialise).

    Args:
        trajectory:             a Step-1 trajectory doc (data/processed/trajectories/*.json).
        em:                     ego motion for the sequence (load_ego_motion).
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
        "schema": "action/v0.2",
        "sequence_id": trajectory["sequence_id"],
        "pedestrian_id": trajectory["pedestrian_id"],
        "is_in_gold_standard": trajectory.get("is_in_gold_standard", True),
        "label_confidence_tier": trajectory.get("label_confidence_tier", "high"),
        "method": "rule_based_geometry",
        "params": {
            "corridor_half_width_m": half_width_m,
            "corridor_lookahead_m": lookahead_m,
            "corridor_min_forward_m": min_forward_m,
        },
    }

    # Undetermined: no real motion to label a crossing from — keep the ped, null the labels.
    if not frames or n_real == 0:
        record.update({
            "status": "undetermined",
            "crosses_ego_corridor": None,
            "crossing_frame_timestamp": None,
            "ego_distance_at_crossing_m": None,
            "crosses_ego_road": None,
            "diagnostics": {"n_frames": len(frames), "n_real_obs": n_real},
        })
        return record

    corridor = _corridor_action(frames, em, half_width_m, lookahead_m, min_forward_m)
    road = (_road_action(frames, T_world_lidar_keyframe, calib, ego_road_polygons)
            if ego_road_polygons else {"crosses_ego_road": None, "road_frames_in_image": 0})

    record.update({
        "status": "determined",
        "crosses_ego_corridor": corridor["crosses_ego_corridor"],
        "crossing_frame_timestamp": corridor["crossing_frame_timestamp"],
        "ego_distance_at_crossing_m": corridor["ego_distance_at_crossing_m"],
        "crosses_ego_road": road["crosses_ego_road"],
        "diagnostics": {
            "n_frames": len(frames),
            "n_real_obs": n_real,
            "n_frames_ahead": corridor["n_frames_ahead"],
            "min_lateral_dist_m": corridor["min_lateral_dist_m"],
            "lateral_sign_change": corridor["lateral_sign_change"],
            "crossing_observed": corridor["crossing_observed"],
            "road_frames_in_image": road["road_frames_in_image"],
        },
    })
    return record


def load_ego_road_polygons(ego_road_json: list) -> List[np.ndarray]:
    """Extract ego_road polygons as a list of (M, 2) pixel-vertex arrays from the parsed JSON."""
    return [np.asarray(poly["geometry"]["coordinates"][0], dtype=np.float64) for poly in ego_road_json]
