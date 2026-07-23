"""Model-consensus committee harness (Step 2 v2 — docs/PIPELINE.md "Action label source").

Turns a Step-1 trajectory into the observation windows a JAAD/PIE-family member consumes,
and aggregates the member's per-window crossing scores into a track-level ACTION verdict
that the GOLD gate compares against the geometric feet-on-road anchor.

Design facts this module encodes (validated 2026-07-16, see PIPELINE.md):
  * Members are SCORERS, never argmax voters: the released PV-LSTM checkpoint tops out at
    p(cross)≈0.42, so any fixed 0.5 rule is degenerate; thresholds are calibrated per member
    on the frozen train split.
  * Members were trained at 30 fps; ZOD's camera runs at ~10 Hz. The RTS-smoothed track is
    continuous in time, so we sample it on a VIRTUAL 30 Hz timeline and project the 3D box
    per tick — native-rate windows without frame interpolation.
  * ZOD's 8 MP fisheye pixels are mapped to the members' JAAD-like pixel space by a uniform
    scale (3848x2168 → x0.5 ≈ 1920x1080). Fisheye-vs-rectilinear distortion remains part of
    the measured domain gap.

The member itself (model load, batching, GPU) lives in the runner script — third-party repos
need sys.path surgery that does not belong in the zodped package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from zodped.labeling.samples import TrackTimeline, project_box_xyxy
from zodped.utils.ego_motion import get_T_world_lidar, load_ego_motion
from zodped.utils.projection import load_calibration, project_world_to_image


@dataclass(frozen=True)
class SweepConfig:
    """Virtual-timeline sweep parameters (defaults = PV-LSTM's released training protocol)."""

    rate_hz: float = 30.0    # members' native frame rate
    obs_len: int = 16        # observation frames per window
    stride: int = 5          # virtual frames between window ends (~0.17 s)
    scale: float = 0.5       # ZOD pixels → member pixel space (3848x2168 → ~1920x1080)

    @property
    def horizon_s(self) -> float:
        """What a window score predicts: crossing within this span after the window ends
        (= the member's future-sequence length, 16 frames at the sweep rate)."""
        return self.obs_len / self.rate_hz

    def as_dict(self) -> dict:
        return {**asdict(self), "horizon_s": round(self.horizon_s, 4)}


class ProjectionContext:
    """The three per-sequence assets box projection needs (calib, ego motion, extrinsics).

    Duck-types the samples.SequenceContext attributes used by project_box_xyxy without paying
    for road surfaces / vehicle signals / sensor listings the sweep never touches.
    """

    def __init__(self, seq_dir: Path) -> None:
        self.calib = load_calibration(seq_dir / "calibration.json")
        self.lidar_ext = np.array(self.calib["FC"]["lidar_extrinsics"])
        self.em = load_ego_motion(seq_dir / "ego_motion.json")


@dataclass
class TrackWindows:
    """All scoreable windows of one track: boxes in member pixel space, cxcywh."""

    t_ends: np.ndarray   # (N,) unix seconds — end of each observation window
    boxes: np.ndarray    # (N, obs_len, 4) [center_x, center_y, w, h], scaled pixels
    n_ticks: int         # virtual ticks the track spans (windows possible before FOV gating)


def _cut_windows(ticks: np.ndarray, boxes: np.ndarray, cfg: SweepConfig) -> TrackWindows:
    """Cut member windows from a per-tick box array: one window per run of `obs_len`
    consecutive valid (non-NaN) ticks, stride `cfg.stride`. Invalid ticks break the run —
    a member must never see a fabricated box."""
    valid = ~np.isnan(boxes[:, 0])
    t_ends: List[float] = []
    windows: List[np.ndarray] = []
    for end in range(cfg.obs_len - 1, len(ticks), cfg.stride):
        start = end - cfg.obs_len + 1
        if valid[start:end + 1].all():
            t_ends.append(float(ticks[end]))
            windows.append(boxes[start:end + 1])
    return TrackWindows(
        t_ends=np.array(t_ends),
        boxes=np.array(windows) if windows else np.empty((0, cfg.obs_len, 4)),
        n_ticks=len(ticks),
    )


def build_track_windows(
    timeline: TrackTimeline, ctx: ProjectionContext, cfg: SweepConfig
) -> TrackWindows:
    """Sweep the virtual 30 Hz timeline over the track and cut member observation windows.

    Every tick projects the tracked 3D box into the image; out-of-FOV ticks break windows.
    Coasted stretches are fine — the projected box is defined there too.
    """
    ticks = np.arange(timeline.t0, timeline.t1, 1.0 / cfg.rate_hz)
    boxes = np.full((len(ticks), 4), np.nan)
    for i, t in enumerate(ticks):
        xyxy, _ = project_box_xyxy(ctx, timeline, float(t))
        if xyxy is not None:
            x1, y1, x2, y2 = xyxy
            boxes[i] = [(x1 + x2) / 2 * cfg.scale, (y1 + y2) / 2 * cfg.scale,
                        (x2 - x1) * cfg.scale, (y2 - y1) * cfg.scale]
    return _cut_windows(ticks, boxes, cfg)


def match_detections_to_track(
    timeline: TrackTimeline,
    ctx: ProjectionContext,
    det_frames: List[tuple],
    min_conf: float = 0.3,
    pad_frac: float = 0.25,
) -> tuple:
    """Assign one cached Step-0 detector box per camera frame to this track.

    For each camera frame inside the track's span, the track position is projected into the
    image and the highest-overlap candidate is chosen among detections (conf ≥ min_conf) whose
    padded box contains that point; ties go to the nearest box centre. Frames with no
    containing detection yield no match (a detector miss or an occlusion — never fabricated).

    `det_frames`: [(unix_ts, (N,5) ndarray of [x1,y1,x2,y2,conf])], sorted by time.
    Returns (matched_ts (M,), matched_cxcywh (M,4) full-resolution pixels).
    """
    m_ts: List[float] = []
    m_boxes: List[List[float]] = []
    for t, dets in det_frames:
        if not (timeline.t0 <= t <= timeline.t1) or len(dets) == 0:
            continue
        uv, valid = project_world_to_image(
            timeline.position_at(t)[None, :], ctx.calib,
            get_T_world_lidar(ctx.em, ctx.lidar_ext, t))
        if not valid[0]:
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])

        best: Optional[List[float]] = None
        best_d2 = np.inf
        for x1, y1, x2, y2, conf in dets:
            if conf < min_conf:
                continue
            pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
            if not (x1 - pw <= u <= x2 + pw and y1 - ph <= v <= y2 + ph):
                continue
            d2 = ((x1 + x2) / 2 - u) ** 2 + ((y1 + y2) / 2 - v) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        if best is not None:
            m_ts.append(t)
            m_boxes.append(best)
    return np.array(m_ts), np.array(m_boxes)


def build_track_windows_from_detections(
    timeline: TrackTimeline,
    ctx: ProjectionContext,
    det_frames: List[tuple],
    cfg: SweepConfig,
    max_bridge_s: float = 0.35,
    min_conf: float = 0.3,
) -> TrackWindows:
    """Windows from Step-0 DETECTOR boxes instead of projected 3D boxes.

    Detector boxes are the same species the members trained on (tight, image-evidence-driven),
    so this isolates the box-style share of the domain gap (EXPERIMENTS_LOG "Committee gate
    v1"). Matched ~10 Hz boxes are lerped onto the 30 Hz timeline; a tick is valid only when
    its bracketing matches are ≤ `max_bridge_s` apart (one skipped camera frame) — longer
    detector gaps break windows rather than inventing boxes.
    """
    m_ts, m_boxes = match_detections_to_track(timeline, ctx, det_frames, min_conf=min_conf)
    ticks = np.arange(timeline.t0, timeline.t1, 1.0 / cfg.rate_hz)
    boxes = np.full((len(ticks), 4), np.nan)
    if len(m_ts) >= 2:
        right = np.searchsorted(m_ts, ticks)
        for i, t in enumerate(ticks):
            r = right[i]
            if r == 0 or r == len(m_ts):
                continue
            t0, t1 = m_ts[r - 1], m_ts[r]
            if t1 - t0 > max_bridge_s:
                continue
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            boxes[i] = ((1 - w) * m_boxes[r - 1] + w * m_boxes[r]) * cfg.scale
    return _cut_windows(ticks, boxes, cfg)


@dataclass
class TrackVerdict:
    """A member's track-level ACTION verdict, aggregated from calibrated window scores."""

    crosses: bool
    score: float                  # track score = max window score (the sweep's strongest claim)
    t_c_estimate: Optional[float]  # unix s: end of FIRST window above threshold (event lies
                                   # within (t_end, t_end + horizon]; this is its lower bound)
    n_windows: int


def aggregate_track(t_ends: np.ndarray, scores: np.ndarray, threshold: float) -> TrackVerdict:
    """Window scores → track verdict: did the member ever see a crossing coming, and when first.

    Track score is the MAX window score: one confident window is a crossing claim, and a track
    that never crosses should never produce one. t_c is estimated from the FIRST above-threshold
    window (not the max — the strongest response often comes mid-crossing, after onset).
    """
    if len(scores) == 0:
        return TrackVerdict(crosses=False, score=float("nan"), t_c_estimate=None, n_windows=0)
    above = scores >= threshold
    return TrackVerdict(
        crosses=bool(above.any()),
        score=float(scores.max()),
        t_c_estimate=float(t_ends[np.argmax(above)]) if above.any() else None,
        n_windows=len(scores),
    )
