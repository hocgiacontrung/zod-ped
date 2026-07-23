"""Step 3 — sample assembly + v1 ROUGH intent labeling (docs/PIPELINE.md "Pipeline Overview").

A SAMPLE is one (pedestrian, observation window) — the dataset's training unit. This module turns
a Step-1 trajectory plus its Step-2 action record into TTE-anchored samples:

  * window proposal — for crossers, one window per prediction horizon, ENDING at t_c − TTE (the
    model observes motion BEFORE the event; a dense stride grid would re-introduce trivial
    in/post-crossing windows). For determined non-crossers, one comparison window ending at the
    track's closest OBSERVED approach to the ego road, gated on a minimum real-detection
    fraction — so the negative class is not built from coasted segments (a label-correlated
    smoothness artifact models would shortcut on).
  * per-window filters — proximity (ego / road distance at the window midpoint) + data
    availability (LiDAR scans and camera frames on disk), per configs/dataset_schema_v0.2.yaml.
  * v1 ROUGH intent — derived straight from the action's t_c: per horizon h the label is
    `crossing` iff t_c ∈ [window_end, window_end + h]. Forward-looking by construction — NOT
    "crosses inside the window". Behavioral intent is a separate later step.
  * per-window geometry — position_ego_rel (relative to the ego pose at window_start) for every
    trajectory frame in [window_start, window_end + max horizon].
  * bbox sequences — per camera frame in the observation window, the tracked 3D box projected
    into the image (xyxy pixels). This is the observation input of the JAAD/PIE model family
    (PV-LSTM / PCPA / pip-thesis all consume bbox+crop windows), and projecting the 3D box gives
    a box on COASTED frames too, which raw Step-0 detections cannot.

Distance to the road is measured against the road SURFACE in 3D: LiDAR points of the keyframe
scan that project inside the ego_road polygon, lifted to the world frame. This reuses the exact
point-in-polygon test of Step 2, so "on the road" means the same thing in both steps, and it
needs no fisheye unprojection.

Tracks that cannot be labeled produce NO samples and are COUNTED, never silently dropped:
`undetermined` actions (EMPTY tracks), tracks in sequences without an ego_road annotation, and
sequences whose keyframe scan yields no road-surface points.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json

import numpy as np

from zodped.dataset.keyframe import (
    load_xyzi,
    motion_compensate_to_keyframe,
    nearest_lidar_scan,
    parse_zod_ts,
)
from zodped.labeling.actions import load_ego_road_polygons, points_in_any_polygon
from zodped.utils.ego_motion import (
    get_T_world_lidar,
    interpolate_pose,
    load_ego_motion,
    transform_points,
)
from zodped.utils.projection import load_calibration, project_lidar_to_image, project_world_to_image
from zodped.utils.vehicle_data import VehicleSignals

CAMERA = "camera_front_blur"
SAMPLE_SCHEMA = "sample/v0.2"

_EPS_S = 1e-4                # slack for timestamp-containment tests (float ISO round-trips)
_MAX_ROAD_POINTS = 60_000    # cap road-surface points for distance queries (memory bound)


# ---------------------------------------------------------------------------
# Configuration + window spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssemblyConfig:
    """Step-3 windowing and filter parameters (defaults = schema v0.2 CONFIRMED values)."""

    observation_window_s: float = 0.5
    prediction_horizons_s: Tuple[float, ...] = (1.0, 1.5, 2.0)
    max_distance_to_ego_m: float = 50.0
    max_distance_to_road_m: float = 15.0
    min_lidar_frames_in_window: int = 3
    # Comparison (negative) windows must contain at least this fraction of REAL detections, so
    # the negative class is not built from coasted segments (see propose_windows).
    min_observed_fraction_comparison: float = 0.7

    @property
    def max_horizon_s(self) -> float:
        return max(self.prediction_horizons_s)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WindowSpec:
    """One proposed observation window, in absolute Unix seconds."""

    start: float
    end: float
    anchor_tte_s: Optional[float]   # TTE this window was anchored at; None = comparison window
    kind: str                       # "tte_anchored" | "closest_approach"


# ---------------------------------------------------------------------------
# Track timeline — interpolation over a Step-1 trajectory
# ---------------------------------------------------------------------------

class TrackTimeline:
    """A Step-1 trajectory as time-indexed arrays, with interpolation at arbitrary instants.

    Positions are linearly interpolated (the RTS-smoothed track is locally near-linear at scan
    cadence); box yaw/size are taken from the NEAREST frame — yaw is circular and size is rigid,
    so interpolation would be wrong or pointless respectively.
    """

    def __init__(self, trajectory: dict) -> None:
        self.frames: List[dict] = trajectory["frames"]
        self.ts = np.array([parse_zod_ts(f["timestamp"]) for f in self.frames])
        self.pos = np.array([f["position_world"] for f in self.frames], dtype=np.float64)
        self.observed = np.array([f["in_observation"] for f in self.frames], dtype=bool)

    @property
    def t0(self) -> float:
        return float(self.ts[0])

    @property
    def t1(self) -> float:
        return float(self.ts[-1])

    def position_at(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.ts, self.pos[:, i]) for i in range(3)])

    def nearest_box(self, t: float) -> dict:
        return self.frames[int(np.argmin(np.abs(self.ts - t)))]["box"]

    def observed_fraction(self, start: float, end: float) -> Optional[float]:
        """Fraction of trajectory frames inside [start, end] with a real detection."""
        mask = (self.ts >= start - _EPS_S) & (self.ts <= end + _EPS_S)
        return float(self.observed[mask].mean()) if mask.any() else None


def ego_xy_at(em: dict, t) -> Tuple[np.ndarray, np.ndarray]:
    """Ego world x, y at Unix time(s) t — translation lerp, matching interpolate_pose."""
    x = np.interp(t, em["timestamps"], em["poses"][:, 0, 3])
    y = np.interp(t, em["timestamps"], em["poses"][:, 1, 3])
    return x, y


# ---------------------------------------------------------------------------
# Sequence context — everything per-sequence a sample needs
# ---------------------------------------------------------------------------

def _sensor_listing(seq_dir: Path, subdir: str, suffix: str) -> List[Tuple[float, str]]:
    """(unix_ts, path-relative-to-seq_dir) for every sensor file, sorted by time."""
    files = sorted((seq_dir / subdir).glob(f"*{suffix}"))
    return [(parse_zod_ts(f.stem.rsplit("_", 1)[-1]), f"{subdir}/{f.name}") for f in files]


def _road_surface_world(
    seq_dir: Path,
    calib: dict,
    em: dict,
    lidar_ext: np.ndarray,
    keyframe_ts: float,
    polygons: Optional[List[np.ndarray]],
) -> Optional[np.ndarray]:
    """Ego-road surface as world-frame 3D points, or None if it cannot be derived.

    Keyframe-nearest LiDAR scan → motion-compensate to the keyframe instant (the polygon is
    drawn on the keyframe image) → keep points projecting inside ego_road → lift to world.
    """
    if not polygons:
        return None
    scan_path, _ = nearest_lidar_scan(seq_dir / "lidar_velodyne", keyframe_ts)
    if scan_path is None:
        return None
    scan_ts = parse_zod_ts(scan_path.stem.rsplit("_", 1)[-1])
    xyzi = motion_compensate_to_keyframe(load_xyzi(scan_path), seq_dir, scan_ts, keyframe_ts)
    uv, valid = project_lidar_to_image(xyzi[:, :3], calib)   # uv rows align with xyzi[valid]
    road = xyzi[valid][points_in_any_polygon(uv, polygons)][:, :3].astype(np.float64)
    if len(road) == 0:
        return None
    if len(road) > _MAX_ROAD_POINTS:
        road = road[:: int(np.ceil(len(road) / _MAX_ROAD_POINTS))]
    T_world_lidar_kf = get_T_world_lidar(em, lidar_ext, keyframe_ts)
    return transform_points(T_world_lidar_kf, road)


def _min_road_distance(points_xy: np.ndarray, road_pts: np.ndarray) -> np.ndarray:
    """Min horizontal distance [m] from each query point to the road surface. (N,) for (N, 2)."""
    d = points_xy[:, None, :] - road_pts[None, :, :2]
    return np.sqrt((d ** 2).sum(-1)).min(axis=1)


class SequenceContext:
    """Per-sequence assets shared by every sample assembled from that sequence."""

    def __init__(self, seq_id: str, seq_dir: Path) -> None:
        self.seq_id = seq_id
        self.seq_dir = seq_dir
        self.calib = load_calibration(seq_dir / "calibration.json")
        self.lidar_ext = np.array(self.calib["FC"]["lidar_extrinsics"])
        self.em = load_ego_motion(seq_dir / "ego_motion.json")
        self.vehicle = VehicleSignals.load(seq_dir / "vehicle_data.hdf5")

        info = json.loads((seq_dir / "info.json").read_text())
        self.keyframe_iso: str = info["keyframe_time"]
        self.keyframe_ts = parse_zod_ts(self.keyframe_iso)

        meta = json.loads((seq_dir / "metadata.json").read_text())
        self.location: Optional[str] = meta.get("country_code")
        self.collection_date: Optional[str] = (meta.get("start_time") or "")[:10] or None

        road_file = seq_dir / "annotations" / "ego_road.json"
        polygons = (
            load_ego_road_polygons(json.loads(road_file.read_text())) if road_file.exists() else None
        )
        self.road_pts_world = _road_surface_world(
            seq_dir, self.calib, self.em, self.lidar_ext, self.keyframe_ts, polygons
        )

        self.camera_frames = _sensor_listing(seq_dir, CAMERA, ".jpg")
        self.lidar_scans = _sensor_listing(seq_dir, "lidar_velodyne", ".npy")
        radar = sorted((seq_dir / "radar_front").glob("*.npy"))
        self.radar_path: Optional[str] = f"radar_front/{radar[0].name}" if radar else None


# ---------------------------------------------------------------------------
# Window proposal
# ---------------------------------------------------------------------------

def propose_windows(
    action: dict,
    timeline: TrackTimeline,
    road_pts: Optional[np.ndarray],
    cfg: AssemblyConfig,
) -> Tuple[List[WindowSpec], Counter]:
    """Propose the TTE-anchored windows for one determined track.

    Returns (windows, skips) — window-proposal skips only (`window_outside_track`); the caller
    is responsible for filtering. Callers must not pass undetermined/unlabeled actions.
    """
    obs = cfg.observation_window_s
    skips: Counter = Counter()
    windows: List[WindowSpec] = []

    if action["crosses_ego_road"]:
        t_c = parse_zod_ts(action["crossing_frame_timestamp"])
        for tte in cfg.prediction_horizons_s:
            end = t_c - tte
            if end - obs >= timeline.t0 - _EPS_S and end <= timeline.t1 + _EPS_S:
                windows.append(WindowSpec(end - obs, end, tte, "tte_anchored"))
            else:
                skips["window_outside_track"] += 1
    else:
        # Comparison window at the closest OBSERVED approach to the road. Both the anchor frame
        # and the window's content are restricted to real detections: unconstrained
        # closest-approach anchoring lands in coasted (Kalman-extrapolated) segments — smooth
        # synthetic motion that only the NEGATIVE class would carry, handing models a
        # "coasting ⇒ not_crossing" shortcut (v1 run: comparison windows averaged 0.42
        # observed_frac vs 0.92 for TTE windows). Candidates are tried in ascending
        # road-distance order; the first window meeting the observed-fraction floor wins.
        if timeline.t1 - timeline.t0 < obs - _EPS_S:
            skips["window_outside_track"] += 1   # track shorter than one observation window
            return [], skips
        dists = _min_road_distance(timeline.pos[:, :2], road_pts)
        order = np.argsort(dists)
        for i in order[timeline.observed[order]]:
            end = float(np.clip(timeline.ts[i], timeline.t0 + obs, timeline.t1))
            frac = timeline.observed_fraction(end - obs, end)
            if frac is not None and frac >= cfg.min_observed_fraction_comparison:
                windows.append(WindowSpec(end - obs, end, None, "closest_approach"))
                break
        else:
            skips["comparison_window_unobserved"] += 1   # no observed segment clean enough

    unique: Dict[int, WindowSpec] = {int(round(w.start * 1000)): w for w in windows}
    return list(unique.values()), skips


# ---------------------------------------------------------------------------
# Per-sample blocks
# ---------------------------------------------------------------------------

def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _intent_block(
    action: dict, w: WindowSpec, cfg: AssemblyConfig, observed_frac: float, dist_road_m: float
) -> dict:
    """v1 ROUGH intent: per-horizon forward-looking labels derived from the action's t_c.

    For a TTE-anchored window, t_c sits ON the anchor-horizon boundary by construction, so
    boundary proximity is NOT a confidence penalty (it is the sampling scheme, as in JAAD/PIE).
    Confidence instead reflects track quality: the observed fraction inside the window, a
    penalty when t_c itself comes from a coasted frame, and (for non-crossers) near-road
    geometry where a small tracking error could flip the label.
    """
    t_c = (
        parse_zod_ts(action["crossing_frame_timestamp"])
        if action["crosses_ego_road"] else None
    )
    labels = {
        f"{h:.1f}": (
            "crossing"
            if t_c is not None and (w.end - _EPS_S) <= t_c <= (w.end + h + _EPS_S)
            else "not_crossing"
        )
        for h in cfg.prediction_horizons_s
    }
    # The scalar label reads at the window's own anchor TTE (crossers) — by construction
    # `crossing` there; comparison windows are `not_crossing`. labels_by_horizon carries the
    # per-horizon variation; `uncertain` is reserved for the behavioral labeler.
    label = labels[f"{w.anchor_tte_s:.1f}"] if w.anchor_tte_s is not None else "not_crossing"

    confidence = observed_frac
    if t_c is not None:
        if action.get("diagnostics", {}).get("crossing_observed") is False:
            confidence *= 0.6   # t_c timestamped on a coasted (Kalman-only) frame
    elif dist_road_m < 1.0:
        confidence *= 0.7       # non-crosser skimming the road edge: label is fragile

    return {
        "label": label,
        "labels_by_horizon": labels,
        "crossing_definition": "t_c in [window_end, window_end + horizon]",
        "method": "derived_from_action",
        "confidence": round(float(confidence), 3),
        "soft_label": {
            "p_crossing": 1.0 if label == "crossing" else 0.0,
            "p_not_crossing": 1.0 if label == "not_crossing" else 0.0,
            "p_uncertain": 0.0,
        },
    }


def _trajectory_block(timeline: TrackTimeline, w: WindowSpec, cfg: AssemblyConfig, em: dict) -> dict:
    """Trajectory frames in [window_start, window_end + max horizon], + position_ego_rel."""
    lo, hi = w.start - _EPS_S, w.end + cfg.max_horizon_s + _EPS_S
    idx = np.where((timeline.ts >= lo) & (timeline.ts <= hi))[0]

    T_ego_world = np.linalg.inv(interpolate_pose(em, w.start))
    rel = transform_points(T_ego_world, timeline.pos[idx])

    frames = []
    for row, i in enumerate(idx):
        f = timeline.frames[i]
        frames.append({
            "timestamp": f["timestamp"],
            "position_world": f["position_world"],
            "position_ego_rel": [round(float(v), 4) for v in rel[row]],
            "in_observation": f["in_observation"],
            "num_lidar_points": f["num_lidar_points"],
            "kalman_confidence": f["kalman_confidence"],
            "tracking_method": f["tracking_method"],
            "box": f["box"],
        })
    return {"frames": frames}


def project_box_xyxy(
    ctx: SequenceContext, timeline: TrackTimeline, t: float
) -> Tuple[Optional[List[float]], int]:
    """The tracked 3D box at time t, projected to an image xyxy bbox (clipped to the frame).

    Returns (bbox, n_corners_visible); bbox is None when no corner lands in the image (behind
    the camera / out of the fisheye FOV). Partially visible boxes yield the bbox of the visible
    corners — truncated exactly like an image-border crop would be.
    """
    center = timeline.position_at(t)
    box = timeline.nearest_box(t)
    l, wdt, h = box["size_lwh"]
    yaw = box["yaw_world"]

    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([l, l, -l, -l]) / 2.0
    dy = np.array([wdt, -wdt, wdt, -wdt]) / 2.0
    gx, gy = c * dx - s * dy, s * dx + c * dy
    ground = np.stack([center[0] + gx, center[1] + gy, np.full(4, center[2])], axis=1)
    corners = np.vstack([ground + [0, 0, h / 2.0], ground - [0, 0, h / 2.0]])

    T_world_lidar = get_T_world_lidar(ctx.em, ctx.lidar_ext, t)
    uv, valid = project_world_to_image(corners, ctx.calib, T_world_lidar)
    if not valid.any():
        return None, 0

    img_w, img_h = ctx.calib["FC"]["image_dimensions"]
    u, v = uv[valid][:, 0], uv[valid][:, 1]
    bbox = [
        round(float(np.clip(u.min(), 0, img_w - 1)), 1),
        round(float(np.clip(v.min(), 0, img_h - 1)), 1),
        round(float(np.clip(u.max(), 0, img_w - 1)), 1),
        round(float(np.clip(v.max(), 0, img_h - 1)), 1),
    ]
    return bbox, int(valid.sum())


def _multimodal_block(
    ctx: SequenceContext,
    timeline: TrackTimeline,
    cams: List[Tuple[float, str]],
    lidars: List[Tuple[float, str]],
) -> dict:
    camera_frames = []
    for ts, rel_path in cams:
        bbox, n_vis = project_box_xyxy(ctx, timeline, ts)
        camera_frames.append({
            "timestamp": _iso(ts),
            "path": rel_path,
            "bbox_xyxy": bbox,
            "bbox_source": "projected_3d_box",
            "n_box_corners_visible": n_vis,
        })
    return {
        "camera_frames": camera_frames,
        "lidar_scans": [{"timestamp": _iso(ts), "path": p} for ts, p in lidars],
        "radar_path": ctx.radar_path,
    }


# ---------------------------------------------------------------------------
# Track → samples
# ---------------------------------------------------------------------------

def assemble_track_samples(
    trajectory: dict,
    action: dict,
    timeline: TrackTimeline,
    ctx: SequenceContext,
    cfg: AssemblyConfig,
    num_pedestrians_in_scene: int,
    is_key_pedestrian: bool,
) -> Tuple[List[dict], Counter]:
    """Assemble every surviving sample for one track. Returns (samples, skip_counts).

    Skip reasons (all counted, nothing silent):
      track-level : undetermined_action, unlabeled_no_ego_road, no_road_surface_points
      window-level: window_outside_track, comparison_window_unobserved, distance_to_ego,
                    distance_to_road, lidar_availability, camera_availability, no_track_frames
    """
    skips: Counter = Counter()

    if action["status"] != "determined":
        skips["undetermined_action"] += 1
        return [], skips
    if action["crosses_ego_road"] is None:
        skips["unlabeled_no_ego_road"] += 1      # sequence has no ego_road annotation
        return [], skips
    if ctx.road_pts_world is None:
        skips["no_road_surface_points"] += 1     # road polygon caught no LiDAR returns
        return [], skips

    windows, skips = propose_windows(action, timeline, ctx.road_pts_world, cfg)

    samples: List[dict] = []
    for w in windows:
        # --- per-window filters, evaluated at the window midpoint (schema v0.2) ---
        mid = (w.start + w.end) / 2.0
        ped_mid = timeline.position_at(mid)
        ego_x, ego_y = ego_xy_at(ctx.em, mid)
        dist_ego = float(np.hypot(ped_mid[0] - ego_x, ped_mid[1] - ego_y))
        if dist_ego > cfg.max_distance_to_ego_m:
            skips["distance_to_ego"] += 1
            continue
        dist_road = float(_min_road_distance(ped_mid[None, :2], ctx.road_pts_world)[0])
        if dist_road > cfg.max_distance_to_road_m:
            skips["distance_to_road"] += 1
            continue

        lidars = [x for x in ctx.lidar_scans if w.start - _EPS_S <= x[0] <= w.end + _EPS_S]
        if len(lidars) < cfg.min_lidar_frames_in_window:
            skips["lidar_availability"] += 1
            continue
        cams = [x for x in ctx.camera_frames if w.start - _EPS_S <= x[0] <= w.end + _EPS_S]
        if not cams:
            skips["camera_availability"] += 1
            continue

        observed_frac = timeline.observed_fraction(w.start, w.end)
        if observed_frac is None:
            skips["no_track_frames"] += 1
            continue

        # --- assemble ---
        sample_id = f"{ctx.seq_id}_{trajectory['pedestrian_id']}_{int(round(w.start * 1000))}"
        samples.append({
            "schema": SAMPLE_SCHEMA,
            "sequence_id": ctx.seq_id,
            "pedestrian_id": trajectory["pedestrian_id"],
            "sample_id": sample_id,
            "window_start_timestamp": _iso(w.start),
            "window_end_timestamp": _iso(w.end),
            "keyframe_timestamp": ctx.keyframe_iso,
            "window_kind": w.kind,
            "anchor_tte_s": w.anchor_tte_s,
            "action": {   # track-level Step-2 record, denormalised into each sample
                "crosses_ego_road": action["crosses_ego_road"],
                "crossing_frame_timestamp": action["crossing_frame_timestamp"],
                "status": action["status"],
                "method": action["method"],
            },
            "intent": _intent_block(action, w, cfg, observed_frac, dist_road),
            "trajectory": _trajectory_block(timeline, w, cfg, ctx.em),
            "multimodal": _multimodal_block(ctx, timeline, cams, lidars),
            "context": {
                "ego_speed_ms": ctx.vehicle.speed_at(w.start),
                "ego_heading_deg": ctx.vehicle.heading_at(w.start),
                "turn_indicator": ctx.vehicle.turn_indicator_at(w.start),
                "distance_to_ego_m": round(dist_ego, 2),
                "distance_to_road_m": round(dist_road, 2),
                "occlusion": (trajectory.get("anchor") or {}).get("occlusion"),
                "observed_fraction_window": round(observed_frac, 3),
            },
            "metadata": {
                "location": ctx.location,
                "collection_date": ctx.collection_date,
                "split": None,   # stamped from the frozen Step-4a sequence mapping by the caller
                "label_confidence_tier": trajectory.get("label_confidence_tier", "high"),
                "is_in_gold_standard": trajectory.get("is_in_gold_standard", True),
                "num_pedestrians_in_scene": num_pedestrians_in_scene,
                "is_key_pedestrian": is_key_pedestrian,
            },
        })

    return samples, skips


def key_pedestrian(timelines: Dict[str, TrackTimeline], em: dict) -> Optional[str]:
    """The sequence's key pedestrian = the track that comes closest to the ego vehicle.

    Deterministic (ties broken by sorted pedestrian id); purely kinematic on purpose — "key"
    marks scene salience, not label quality.
    """
    best_id, best_d = None, np.inf
    for ped_id, tl in sorted(timelines.items()):
        ex, ey = ego_xy_at(em, tl.ts)
        d = float(np.min(np.hypot(tl.pos[:, 0] - ex, tl.pos[:, 1] - ey)))
        if d < best_d:
            best_id, best_d = ped_id, d
    return best_id
