"""Single-pedestrian linker — detector-as-measurement Kalman/RTS tracker.

Step 1 (trajectory generation) uses a DETECTOR-AS-MEASUREMENT architecture (see docs/PIPELINE.md):
the 2D detector + frustum lift (src/labeling/frustum.py) supplies, per frame, a SET of candidate
world-frame positions; this module LINKS them into one smooth track. The driver
(scripts/01_generate_trajectories.py) builds the candidate pool once per sequence and calls
`track_pedestrian_from_detections` once per keyframe-anchored pedestrian.

The linker is a constant-velocity Kalman filter (`_F`/`_Q`, predict/update) followed by a
Rauch-Tung-Striebel backward smoother (`_rts_smooth`):

  1. Seed at the verified keyframe box (world frame), then associate forward and backward from
     that anchor. Each frame: PREDICT the position, gate the frame's candidates by squared
     Mahalanobis distance to the prediction, take the nearest in-gate candidate as the
     measurement. An empty gate (or a frame with no candidates) coasts; a pass terminates after
     `max_consecutive_misses` consecutive coasts.
  2. RTS-smooth the merged chronological records into the final world-frame trajectory and a
     per-frame `kalman_confidence`.

All positions are in the world frame; candidates arrive pre-lifted, so this module needs no ego
motion or LiDAR I/O. `position_ego_rel` is per-window and is added during sample assembly, NOT here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Kalman / association defaults (mirror configs/; see docs/PIPELINE.md) ---------
_STATE_DIM = 6                  # [x, y, z, vx, vy, vz]
_DT_NOMINAL = 0.111             # s, fallback scan spacing (~9 Hz LiDAR)
_Q_POS_SIGMA = 0.05             # m, process noise std on position
_Q_VEL_SIGMA = 0.3             # m/s, process noise std on velocity
_INIT_POS_SIGMA = 0.5          # m, initial uncertainty on the seed position
_INIT_VEL_SIGMA = 2.0          # m/s, initial uncertainty on the (unknown) seed velocity
_MAHAL_CLIP = 25.0             # cap on squared Mahalanobis distance for confidence
_FRUSTUM_MEAS_SIGMA = 0.3      # m, measurement noise std for the frustum lift
_GATE_MAHAL2 = 9.0            # squared-Mahalanobis gate for detection association (~3 dof, 97%)

# Measurement matrix: observe position, not velocity.
_H = np.hstack([np.eye(3), np.zeros((3, 3))])


def _iso_utc(unix_ts: float) -> str:
    """Format a Unix timestamp as a UTC ISO-8601 string (schema `timestamp` field)."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


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


def _rts_smooth(records: List[dict], meas_sigma: float = _FRUSTUM_MEAS_SIGMA) -> List[dict]:
    """Forward constant-velocity Kalman + RTS backward smoothing over measurements.

    Consumes chronological association records (measurement present or None) and emits
    the final trajectory frames with smoothed `position_world` and `kalman_confidence`.
    Coasted records (measurement is None) contribute no update; their position is the
    propagated state and confidence is 0.0.

    `meas_sigma` is the measurement noise std (m) for the frustum lift.
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
            "kalman_confidence": confidence[i],
            "tracking_method": rec["tracking_method"],
        })
    return frames


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

    Returns one measurement record per frame consumed, each with: unix_timestamp,
    measurement ((3,) or None), tracking_method, num_lidar_points, in_observation.
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

    Seeds at the keyframe (the verified GOLD anchor), associates frustum detections forward and
    backward, then RTS-smooths the merged track. The candidate POOL is built once per sequence by
    the Step 1 driver and shared across that sequence's pedestrians.

    Args:
        frames:                  Per-frame candidate pools (any order; sorted internally).
        anchor_unix_ts:          Keyframe timestamp; the nearest frame becomes the anchor.
        seed_pos_world:          (3,) keyframe box centre in the world frame.
        gate_mahal2:             Squared-Mahalanobis association gate.
        max_consecutive_misses:  Coasted frames tolerated before a pass terminates.
        meas_sigma:              Frustum measurement noise std (m).

    Returns smoothed frame dicts (schema: timestamp, position_world, in_observation,
    num_lidar_points, kalman_confidence, tracking_method), or [] if no frames.
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


def birth_tracks_from_residual_pool(
    pool: List[FrameCandidates],
    *,
    gate_mahal2: float = _GATE_MAHAL2,
    max_consecutive_misses: int = 5,
    meas_sigma: float = _FRUSTUM_MEAS_SIGMA,
    min_support: int = 4,
    min_duration_s: float = 1.0,
) -> List[List[dict]]:
    """Birth SILVER tracks from frustum candidates that no GOLD anchor claimed (Step 1b).

    Unlike the GOLD linker there is NO keyframe seed, so this is an online multi-object tracker over
    the UNCLAIMED candidates. Per frame, active CV-Kalman tracks predict and greedily claim the
    nearest in-gate candidate (cheapest track↔candidate pair first, one candidate per track);
    candidates nobody claims SEED new tentative tracks; a track ends after `max_consecutive_misses`
    coasts. A finished track survives only if it gathered `min_support` real observations spanning
    `min_duration_s` (this filters detector flicker and one-off ghost lifts), then is RTS-smoothed
    into the standard frame schema. Boxes/size are the caller's responsibility — SILVER uses a
    pedestrian size PRIOR, since no keyframe extent exists for a detector-born track.

    Returns a list of smoothed tracks (each a list of frame dicts).
    """
    R = np.eye(3) * meas_sigma ** 2
    P0 = np.diag([_INIT_POS_SIGMA ** 2] * 3 + [_INIT_VEL_SIGMA ** 2] * 3)

    def _record(ts: float, meas: Optional[np.ndarray], n_pts: int, observed: bool) -> dict:
        return {"unix_timestamp": ts, "measurement": meas, "tracking_method": "forward",
                "num_lidar_points": n_pts, "in_observation": observed}

    active: List[dict] = []
    finished: List[dict] = []

    for ts, cand_world, cand_n in sorted(pool, key=lambda f: f[0]):
        # 1. Predict every active track to this frame's time.
        for tr in active:
            dt = ts - tr["last_ts"]
            if dt <= 0:
                dt = _DT_NOMINAL
            F = _F(dt)
            tr["x"] = F @ tr["x"]
            tr["P"] = F @ tr["P"] @ F.T + _Q(dt)
            tr["last_ts"] = ts

        # 2. Global nearest-first association: cheapest (track, candidate) pair wins, one each.
        matched: Dict[int, int] = {}
        taken_c: set = set()
        if len(cand_world) and active:
            pairs: List[Tuple[float, int, int]] = []
            for ti, tr in enumerate(active):
                S_inv = np.linalg.inv(_H @ tr["P"] @ _H.T + R)
                innov = cand_world - (_H @ tr["x"])
                m2 = np.einsum("ki,ij,kj->k", innov, S_inv, innov)
                for cj in range(len(cand_world)):
                    if m2[cj] <= gate_mahal2:
                        pairs.append((float(m2[cj]), ti, cj))
            taken_t: set = set()
            for _, ti, cj in sorted(pairs):
                if ti in taken_t or cj in taken_c:
                    continue
                taken_t.add(ti)
                taken_c.add(cj)
                matched[ti] = cj

        # 3. Update matched tracks; coast the rest.
        for ti, tr in enumerate(active):
            if ti in matched:
                cj = matched[ti]
                z = cand_world[cj]
                S = _H @ tr["P"] @ _H.T + R
                K = tr["P"] @ _H.T @ np.linalg.inv(S)
                tr["x"] = tr["x"] + K @ (z - _H @ tr["x"])
                tr["P"] = (np.eye(_STATE_DIM) - K @ _H) @ tr["P"]
                tr["misses"] = 0
                tr["records"].append(_record(ts, z, int(cand_n[cj]), True))
            else:
                tr["misses"] += 1
                tr["records"].append(_record(ts, None, -1, False))

        # 4. Birth new tentative tracks from unclaimed candidates.
        for cj in range(len(cand_world)):
            if cj in taken_c:
                continue
            active.append({
                "x": np.concatenate([cand_world[cj], np.zeros(3)]),
                "P": P0.copy(),
                "last_ts": ts,
                "misses": 0,
                "records": [_record(ts, cand_world[cj], int(cand_n[cj]), True)],
            })

        # 5. Retire tracks that have coasted past the miss budget.
        keep: List[dict] = []
        for tr in active:
            (finished if tr["misses"] >= max_consecutive_misses else keep).append(tr)
        active = keep

    finished.extend(active)

    # 6. Confirm + smooth: drop trailing coasts, gate on support/duration, RTS-smooth survivors.
    tracks: List[List[dict]] = []
    for tr in finished:
        records = tr["records"]
        while records and records[-1]["measurement"] is None:
            records.pop()
        real = [r for r in records if r["measurement"] is not None]
        if len(real) < min_support:
            continue
        if real[-1]["unix_timestamp"] - real[0]["unix_timestamp"] < min_duration_s:
            continue
        tracks.append(_rts_smooth(records, meas_sigma=meas_sigma))
    return tracks
