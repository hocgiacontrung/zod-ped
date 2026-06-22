"""LiDAR single-pedestrian tracker — Kalman/RTS linker + gated-centroid measurement.

STATUS (2026-06-17): Step 2 moved to a DETECTOR-as-measurement architecture (see
docs/PIPELINE.md). This module is retained with a split role:
  * KEPT as the reusable LINKER — the constant-velocity Kalman (`_F`/`_Q`, predict/update)
    and RTS smoother (`_rts_smooth`) link per-scan measurements into a smooth world-frame
    track. The detector driver (`scripts/02_generate_trajectories.py`) will feed detector
    boxes into this same machinery; only the measurement source changes.
  * DEMOTED — the gated-centroid measurement (`_gate_centroid`/`_associate_pass`) is no
    longer the primary position source; it survives only as a COAST FALLBACK for scans where
    the detector returns nothing. The gated-centroid full run was NOT executed.

The original gated-centroid design (now the fallback path) is described below:

  1. Compensate-before-associate: every scan's points are lifted to a single world
     frame [ pose(t) @ T_ego_lidar @ p_lidar ] before any association, so ego motion
     is never mistaken for pedestrian motion.
  2. Gated centroid association (anchor-outward, forward + backward from the keyframe):
     an online constant-velocity Kalman filter PREDICTS the pedestrian position each
     scan; the gate (fixed keyframe box size + margin) is centred there; the centroid
     of the enclosed world points is the measurement. Empty gate => coast
     (in_observation=False); a pass terminates after `max_consecutive_misses`.
  3. RTS smoothing: the merged chronological measurements are run through a single
     forward constant-velocity Kalman filter + Rauch-Tung-Striebel backward pass to
     produce the final world-frame trajectory and per-frame kalman_confidence.

All positions are in the world frame. `position_ego_rel` is per-window and is added
during sample assembly, NOT here.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils.ego_motion import get_T_world_lidar, transform_points


_TS_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}T.+)\.npy$")

# --- Kalman / association defaults (mirror configs/; see docs/PIPELINE.md) ---------
_STATE_DIM = 6                  # [x, y, z, vx, vy, vz]
_DT_NOMINAL = 0.111             # s, fallback scan spacing (~9 Hz LiDAR)
_MEAS_SIGMA = 0.1               # m, measurement (centroid) noise std
_Q_POS_SIGMA = 0.05             # m, process noise std on position
_Q_VEL_SIGMA = 0.3             # m/s, process noise std on velocity
_INIT_POS_SIGMA = 0.5          # m, initial uncertainty on the seed position
_INIT_VEL_SIGMA = 2.0          # m/s, initial uncertainty on the (unknown) seed velocity
_MAHAL_CLIP = 25.0             # cap on squared Mahalanobis distance for confidence
_FRUSTUM_MEAS_SIGMA = 0.3      # m, measurement noise std for the frustum lift (noisier than centroid)
_GATE_MAHAL2 = 9.0            # squared-Mahalanobis gate for detection association (~3 dof, 97%)

# Measurement matrix: observe position, not velocity.
_H = np.hstack([np.eye(3), np.zeros((3, 3))])


def _lidar_unix_ts(path: Path) -> float:
    """Parse the Unix timestamp from a LiDAR filename like *_<ISO>.npy."""
    m = _TS_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse timestamp from LiDAR filename: {path.name}")
    ts_str = m.group(1).replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str).timestamp()


def _iso_utc(unix_ts: float) -> str:
    """Format a Unix timestamp as a UTC ISO-8601 string (schema `timestamp` field)."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def _load_xyz(path: Path) -> np.ndarray:
    """Load a structured LiDAR .npy and return (N, 3) float64 XYZ."""
    cloud = np.load(path)
    return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float64)


def _gate_centroid(
    pts_world: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    min_points: int,
) -> Tuple[Optional[np.ndarray], int]:
    """Centroid of points inside the axis-aligned gate, or (None, 0) if too few.

    Args:
        pts_world:    (N, 3) points in world frame.
        center:       (3,) gate centre — the Kalman-PREDICTED position.
        half_extents: (3,) gate half-widths [dx, dy, dz].
        min_points:   minimum hits required to accept a measurement.

    Returns:
        (centroid, n_points), or (None, 0) when fewer than min_points fall inside.
    """
    inside = np.all(np.abs(pts_world - center) <= half_extents, axis=1)
    n = int(inside.sum())
    if n < min_points:
        return None, 0
    return pts_world[inside].mean(axis=0), n


def _F(dt: float) -> np.ndarray:
    """Constant-velocity state-transition matrix for step dt."""
    F = np.eye(_STATE_DIM)
    F[:3, 3:] = np.eye(3) * dt
    return F


def _Q(dt: float) -> np.ndarray:
    """Process-noise covariance from a piecewise-constant white-acceleration model."""
    G = np.vstack([np.eye(3) * dt ** 2 / 2, np.eye(3) * dt])
    Q = G @ G.T * _Q_VEL_SIGMA ** 2
    Q[:3, :3] += np.eye(3) * _Q_POS_SIGMA ** 2
    return Q


def _associate_pass(
    files: List[Path],
    em: dict,
    lidar_extrinsics: np.ndarray,
    seed_pos_world: np.ndarray,
    half_extents: np.ndarray,
    direction: str,
    min_points: int,
    max_consecutive_misses: int,
) -> List[dict]:
    """Gated-centroid association along an ordered list of scans (one pass).

    Runs an online constant-velocity Kalman filter seeded at `seed_pos_world` purely
    to PREDICT the gate centre each scan. Produces one measurement record per scan
    consumed; terminates early after `max_consecutive_misses` empty gates.

    Args:
        files:                   Ordered scans (chronological for forward,
                                 reverse-chronological for backward).
        em:                      ego_motion dict from load_ego_motion().
        lidar_extrinsics:        4x4 T[ego<-lidar].
        seed_pos_world:          (3,) keyframe centroid in world frame.
        half_extents:            (3,) gate half-widths.
        direction:               "forward" or "backward" (stored per record).
        min_points:              Min points in the gate to accept a measurement.
        max_consecutive_misses:  Coasted scans tolerated before the pass terminates.

    Returns:
        List of measurement records (NOT including the seed/anchor frame), each with:
          unix_timestamp, measurement ((3,) or None), tracking_method,
          num_lidar_points, in_observation, tracking_lost.
    """
    x = np.concatenate([seed_pos_world, np.zeros(3)])
    P = np.diag([_INIT_POS_SIGMA ** 2] * 3 + [_INIT_VEL_SIGMA ** 2] * 3)
    R = np.eye(3) * _MEAS_SIGMA ** 2

    records: List[dict] = []
    misses = 0
    prev_ts: Optional[float] = None

    for fpath in files:
        ts = _lidar_unix_ts(fpath)
        dt = _DT_NOMINAL if prev_ts is None else abs(ts - prev_ts)
        if dt <= 0:
            dt = _DT_NOMINAL
        prev_ts = ts

        # Predict — the prior position centres the gate.
        F = _F(dt)
        x = F @ x
        P = F @ P @ F.T + _Q(dt)
        predicted_pos = x[:3]

        T = get_T_world_lidar(em, lidar_extrinsics, ts)
        pts_world = transform_points(T, _load_xyz(fpath))
        centroid, n = _gate_centroid(pts_world, predicted_pos, half_extents, min_points)

        if centroid is not None:
            # Update — keep the gate filter tracking the pedestrian.
            y = centroid - _H @ x
            S = _H @ P @ _H.T + R
            K = P @ _H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(_STATE_DIM) - K @ _H) @ P
            misses = 0
            in_observation = True
        else:
            misses += 1
            in_observation = False

        records.append({
            "unix_timestamp": ts,
            "measurement": None if centroid is None else centroid,
            "tracking_method": direction,
            "num_lidar_points": n if centroid is not None else -1,
            "in_observation": in_observation,
            "tracking_lost": not in_observation,
        })

        if misses >= max_consecutive_misses:
            break

    return records


def _rts_smooth(records: List[dict], meas_sigma: float = _MEAS_SIGMA) -> List[dict]:
    """Forward constant-velocity Kalman + RTS backward smoothing over measurements.

    Consumes chronological association records (measurement present or None) and emits
    the final trajectory frames with smoothed `position_world` and `kalman_confidence`.
    Coasted records (measurement is None) contribute no update; their position is the
    propagated state and confidence is 0.0.

    `meas_sigma` is the measurement noise std (m); the frustum measurement is noisier than
    the gated centroid, so the detector driver passes a larger value than the default.
    """
    n_steps = len(records)
    R = np.eye(3) * meas_sigma ** 2

    # Seed the filter at the first available measurement (the anchor in practice).
    seed = next((r["measurement"] for r in records if r["measurement"] is not None), None)
    if seed is None:
        seed = np.zeros(3)
    x = np.concatenate([seed, np.zeros(3)])
    P = np.diag([_INIT_POS_SIGMA ** 2] * 3 + [_INIT_VEL_SIGMA ** 2] * 3)

    # Storage for the RTS backward pass.
    x_prior = [None] * n_steps
    P_prior = [None] * n_steps
    x_filt = [None] * n_steps
    P_filt = [None] * n_steps
    F_step = [None] * n_steps
    confidence = [0.0] * n_steps

    prev_ts: Optional[float] = None
    for i, rec in enumerate(records):
        ts = rec["unix_timestamp"]
        dt = _DT_NOMINAL if prev_ts is None else ts - prev_ts
        if dt <= 0:
            dt = _DT_NOMINAL
        prev_ts = ts

        F = _F(dt)
        x = F @ x
        P = F @ P @ F.T + _Q(dt)
        x_prior[i] = x.copy()
        P_prior[i] = P.copy()
        F_step[i] = F

        z = rec["measurement"]
        if z is not None:
            y = z - _H @ x
            S = _H @ P @ _H.T + R
            K = P @ _H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(_STATE_DIM) - K @ _H) @ P
            mahal2 = float(y @ np.linalg.inv(S) @ y)
            confidence[i] = float(np.exp(-0.5 * min(mahal2, _MAHAL_CLIP)))

        x_filt[i] = x.copy()
        P_filt[i] = P.copy()

    # RTS backward pass.
    x_smooth = list(x_filt)
    for i in range(n_steps - 2, -1, -1):
        C = P_filt[i] @ F_step[i + 1].T @ np.linalg.inv(P_prior[i + 1])
        x_smooth[i] = x_filt[i] + C @ (x_smooth[i + 1] - x_prior[i + 1])

    frames: List[dict] = []
    for i, rec in enumerate(records):
        frames.append({
            "timestamp": _iso_utc(rec["unix_timestamp"]),
            "position_world": x_smooth[i][:3].tolist(),
            "in_observation": rec["in_observation"],
            "num_lidar_points": rec["num_lidar_points"],
            "tracking_lost": rec["tracking_lost"],
            "kalman_confidence": confidence[i],
            "tracking_method": rec["tracking_method"],
        })
    return frames


def track_pedestrian(
    lidar_dir: Path,
    em: dict,
    lidar_extrinsics: np.ndarray,
    keyframe_unix_ts: float,
    seed_pos_world: np.ndarray,
    box_size: Tuple[float, float, float],
    search_margin: float = 0.75,
    min_points: int = 3,
    max_consecutive_misses: int = 5,
) -> List[dict]:
    """Track one keyframe-annotated pedestrian across a sequence's LiDAR scans.

    Seeds at the keyframe (anchor), associates forward and backward via gated
    centroids, then RTS-smooths the merged track. Returns one frame per consumed
    scan, sorted chronologically; the anchor carries tracking_method="anchor".

    Args:
        lidar_dir:               Directory of .npy LiDAR scans.
        em:                      ego_motion dict (load_ego_motion()).
        lidar_extrinsics:        4x4 T[ego<-lidar] from calibration.json.
        keyframe_unix_ts:        Unix timestamp of the annotation keyframe.
        seed_pos_world:          (3,) keyframe centroid in world frame.
        box_size:                (length, width, height) from the annotation (m).
        search_margin:           Gate expansion beyond the box, per axis (m).
        min_points:              Min LiDAR points in the gate for a measurement.
        max_consecutive_misses:  Coasted scans tolerated before a pass terminates.

    Returns:
        List of smoothed frame dicts (schema: timestamp, position_world,
        in_observation, num_lidar_points, tracking_lost, kalman_confidence,
        tracking_method), or [] if the sequence has no scans.
    """
    all_files = sorted(lidar_dir.glob("*.npy"), key=_lidar_unix_ts)
    if not all_files:
        return []

    xy_half = max(box_size[0], box_size[1]) / 2 + search_margin
    z_half = box_size[2] / 2 + search_margin
    half_extents = np.array([xy_half, xy_half, z_half])

    file_ts = np.array([_lidar_unix_ts(f) for f in all_files])
    anchor_idx = int(np.argmin(np.abs(file_ts - keyframe_unix_ts)))

    anchor_record = {
        "unix_timestamp": file_ts[anchor_idx],
        "measurement": np.asarray(seed_pos_world, dtype=np.float64),
        "tracking_method": "anchor",
        "num_lidar_points": -1,
        "in_observation": True,
        "tracking_lost": False,
    }

    fwd = _associate_pass(
        all_files[anchor_idx + 1:], em, lidar_extrinsics, seed_pos_world,
        half_extents, "forward", min_points, max_consecutive_misses,
    )
    bwd = _associate_pass(
        all_files[:anchor_idx][::-1], em, lidar_extrinsics, seed_pos_world,
        half_extents, "backward", min_points, max_consecutive_misses,
    )

    records = sorted(bwd + [anchor_record] + fwd, key=lambda r: r["unix_timestamp"])
    return _rts_smooth(records)


# ===========================================================================
# Detector-as-measurement linker (Step 2 PRIMARY path; see docs/PIPELINE.md)
# ---------------------------------------------------------------------------
# Same constant-velocity Kalman + RTS smoother as above; only the measurement source
# differs. Per frame the driver supplies a SET of candidate world positions (the frame's
# frustum-lifted 2D detections). The filter PREDICTS, gates the candidates by squared
# Mahalanobis distance to that prediction, and takes the nearest in-gate candidate as the
# measurement (empty gate => coast). This is the data-association the gated-centroid path
# does not need (it has one centroid per scan); everything downstream is shared.
# ===========================================================================

# A per-frame candidate pool: (unix_ts, world positions (K,3), per-candidate LiDAR-point counts (K,)).
FrameCandidates = Tuple[float, np.ndarray, np.ndarray]


def _associate_detection_pass(
    frames: List[FrameCandidates],
    seed_pos_world: np.ndarray,
    direction: str,
    gate_mahal2: float,
    max_consecutive_misses: int,
    meas_sigma: float,
) -> List[dict]:
    """Gate frustum detections to a CV-Kalman prediction along one ordered pass.

    Args:
        frames:                  Frames in pass order (chronological for forward,
                                 reverse-chronological for backward), each a FrameCandidates
                                 tuple. A frame with no candidates (K=0) counts as a miss.
        seed_pos_world:          (3,) anchor position seeding the predictor.
        direction:               "forward" or "backward" (stored per record).
        gate_mahal2:             Squared-Mahalanobis gate; candidates beyond it are rejected.
        max_consecutive_misses:  Coasted frames tolerated before the pass terminates.
        meas_sigma:              Measurement noise std (m) for the gate/update.

    Returns one measurement record per frame consumed (same schema as `_associate_pass`).
    """
    x = np.concatenate([seed_pos_world, np.zeros(3)])
    P = np.diag([_INIT_POS_SIGMA ** 2] * 3 + [_INIT_VEL_SIGMA ** 2] * 3)
    R = np.eye(3) * meas_sigma ** 2

    records: List[dict] = []
    misses = 0
    prev_ts: Optional[float] = None

    for ts, cand_world, cand_npoints in frames:
        dt = _DT_NOMINAL if prev_ts is None else abs(ts - prev_ts)
        if dt <= 0:
            dt = _DT_NOMINAL
        prev_ts = ts

        # Predict — the prior position centres the association gate.
        F = _F(dt)
        x = F @ x
        P = F @ P @ F.T + _Q(dt)

        chosen: Optional[np.ndarray] = None
        chosen_n = -1
        if len(cand_world):
            S = _H @ P @ _H.T + R
            S_inv = np.linalg.inv(S)
            innov = cand_world - (_H @ x)                       # (K, 3)
            mahal2 = np.einsum("ki,ij,kj->k", innov, S_inv, innov)
            j = int(np.argmin(mahal2))
            if mahal2[j] <= gate_mahal2:
                chosen = cand_world[j]
                chosen_n = int(cand_npoints[j])

        if chosen is not None:
            # Update — keep the filter tracking the pedestrian.
            y = chosen - _H @ x
            S = _H @ P @ _H.T + R
            K = P @ _H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(_STATE_DIM) - K @ _H) @ P
            misses = 0
            in_observation = True
        else:
            misses += 1
            in_observation = False

        records.append({
            "unix_timestamp": ts,
            "measurement": chosen,
            "tracking_method": direction,
            "num_lidar_points": chosen_n,
            "in_observation": in_observation,
            "tracking_lost": not in_observation,
        })

        if misses >= max_consecutive_misses:
            break

    return records


def track_pedestrian_from_detections(
    frames: List[FrameCandidates],
    anchor_unix_ts: float,
    seed_pos_world: np.ndarray,
    gate_mahal2: float = _GATE_MAHAL2,
    max_consecutive_misses: int = 5,
    meas_sigma: float = _FRUSTUM_MEAS_SIGMA,
) -> List[dict]:
    """Track one keyframe-anchored pedestrian from per-frame frustum detections.

    The detector-as-measurement counterpart of `track_pedestrian`: seeds at the keyframe
    (the verified GOLD anchor), associates frustum detections forward and backward, then
    RTS-smooths the merged track. The candidate POOL is built once per sequence by the
    Step 2 driver and shared across that sequence's pedestrians.

    Args:
        frames:                  Per-frame candidate pools (any order; sorted internally).
        anchor_unix_ts:          Keyframe timestamp; the nearest frame becomes the anchor.
        seed_pos_world:          (3,) keyframe box centre in the world frame.
        gate_mahal2:             Squared-Mahalanobis association gate.
        max_consecutive_misses:  Coasted frames tolerated before a pass terminates.
        meas_sigma:              Frustum measurement noise std (m).

    Returns smoothed frame dicts (schema matches `track_pedestrian`), or [] if no frames.
    """
    if not frames:
        return []

    frames = sorted(frames, key=lambda f: f[0])
    frame_ts = np.array([f[0] for f in frames])
    anchor_idx = int(np.argmin(np.abs(frame_ts - anchor_unix_ts)))

    anchor_record = {
        "unix_timestamp": frame_ts[anchor_idx],
        "measurement": np.asarray(seed_pos_world, dtype=np.float64),
        "tracking_method": "anchor",
        "num_lidar_points": -1,
        "in_observation": True,
        "tracking_lost": False,
    }

    fwd = _associate_detection_pass(
        frames[anchor_idx + 1:], seed_pos_world, "forward",
        gate_mahal2, max_consecutive_misses, meas_sigma,
    )
    bwd = _associate_detection_pass(
        frames[:anchor_idx][::-1], seed_pos_world, "backward",
        gate_mahal2, max_consecutive_misses, meas_sigma,
    )

    records = sorted(bwd + [anchor_record] + fwd, key=lambda r: r["unix_timestamp"])
    return _rts_smooth(records, meas_sigma=meas_sigma)
