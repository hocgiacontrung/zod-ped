"""Ego-corridor (swept-path) crossing geometry — BENCHED (2026-07-08).

This module is intentionally NOT called by the active pipeline. It was Step 2's original PRIMARY
label ("did the pedestrian enter the ego vehicle's swept path"), computed from ego_motion.poses +
vehicle width in the world frame. We demoted the crossing ACTION to `crosses_ego_road` (feet on the
ego road; see zodped.labeling.actions) and benched the corridor rather than deleting it: the
computation is pure, deterministic, and re-derivable in minutes, so it costs nothing to keep dormant
and can be revived as a Step-4 aux feature (ego-relevance / metric range-to-crossing) with a single
call — no git archaeology. See docs/PIPELINE.md and docs/EXPERIMENTS_LOG.md.

Nothing here runs unless something imports `corridor_crossing` explicitly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts

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


def corridor_crossing(
    frames: Sequence[dict],
    em: dict,
    half_width_m: float = CORRIDOR_HALF_WIDTH_M,
    lookahead_m: float = CORRIDOR_LOOKAHEAD_M,
    min_forward_m: float = CORRIDOR_MIN_FORWARD_M,
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
