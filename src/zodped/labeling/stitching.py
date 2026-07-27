"""Tracklet stitching — propose merges of fragmented tracks that are one pedestrian.

The KF/RTS linker (``tracker.py``) terminates a track after ``max_consecutive_misses`` coasts
(~0.44 s), so a pedestrian who is missed for longer — occluded, or dropped by the detector — is
re-birthed as a fresh track. One walker then lives on disk as several fragments (a GOLD anchor
track + SILVER stubs). This module re-links those fragments *after* tracking.

It is deliberately PROPOSE-ONLY: it never rewrites a trajectory file. It reads the per-pedestrian
trajectory JSONs of one sequence, and for each ordered pair (A ends, B begins later) it tests
whether B is the continuation of A:

  1. time    — B's first observed frame is within [-overlap, max_gap_s] of A's last observed frame.
  2. motion  — coast A's last observed position forward at its own velocity across the gap; the
               residual to B's first observed position must be <= max_residual_m (a gap-scaled gate).
  3. height  — box heights agree within max_height_diff_m (loose; SILVER sizes are mostly a prior).

Surviving pairs are chained greedily (each track keeps its single cheapest successor) into groups.
Each group is one pedestrian; the PRIMARY id is a GOLD member if present (most observed frames wins),
else the longest SILVER fragment. Every other member is a fragment that should merge into the primary.

Applying the merge (concatenating frames, re-smoothing, re-running Steps 2-4) is a separate step; on
the curated batch the human confirms each proposal first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts


@dataclass
class StitchParams:
    """Gates for a proposed continuation A -> B (all CLI-tunable)."""
    max_gap_s: float = 2.0          # longest time gap between A's end and B's start to bridge
    max_overlap_s: float = 0.3      # allow B to start slightly before A ends (jitter)
    base_residual_m: float = 1.0    # motion-prediction residual allowed at zero gap
    residual_per_s: float = 1.0     # extra residual allowed per second of gap (velocity error grows)
    max_height_diff_m: float = 0.6  # box-height agreement
    vel_window: int = 5             # observed frames used to estimate endpoint velocity


@dataclass
class TrackEnd:
    """Endpoint summary of one track (observed frames only)."""
    ped_id: str
    is_gold: bool
    n_obs: int
    start_ts: float
    end_ts: float
    start_pos: np.ndarray           # xy, first observed
    end_pos: np.ndarray             # xy, last observed
    end_vel: np.ndarray             # xy m/s, over the last vel_window observed frames
    height: float


def summarize_track(doc: dict, vel_window: int) -> Optional[TrackEnd]:
    """Reduce a trajectory doc to its observed endpoints; None if it has no real measurement."""
    obs = [f for f in doc["frames"] if f["in_observation"] and f.get("num_lidar_points", -1) >= 0]
    if not obs:
        return None
    ts = np.array([parse_zod_ts(f["timestamp"]) for f in obs])
    pos = np.array([f["position_world"][:2] for f in obs], dtype=np.float64)
    heights = [f["box"]["size_lwh"][2] for f in obs]

    # endpoint velocity from the last vel_window observed frames (fall back to 0 if too short)
    k = min(vel_window, len(obs))
    if k >= 2 and ts[-1] > ts[-k]:
        end_vel = (pos[-1] - pos[-k]) / (ts[-1] - ts[-k])
    else:
        end_vel = np.zeros(2)

    return TrackEnd(
        ped_id=doc["pedestrian_id"],
        is_gold=bool(doc.get("is_in_gold_standard", False)),
        n_obs=len(obs),
        start_ts=float(ts[0]),
        end_ts=float(ts[-1]),
        start_pos=pos[0],
        end_pos=pos[-1],
        end_vel=end_vel,
        height=float(np.median(heights)),
    )


def edge_cost(a: TrackEnd, b: TrackEnd, p: StitchParams) -> Optional[dict]:
    """Cost of continuing A -> B, or None if a gate fails. Cost = motion residual (metres)."""
    gap = b.start_ts - a.end_ts
    if gap < -p.max_overlap_s or gap > p.max_gap_s:
        return None
    if abs(a.height - b.height) > p.max_height_diff_m:
        return None
    predicted = a.end_pos + a.end_vel * max(gap, 0.0)
    residual = float(np.linalg.norm(predicted - b.start_pos))
    allowed = p.base_residual_m + p.residual_per_s * max(gap, 0.0)
    if residual > allowed:
        return None
    return {"gap_s": round(gap, 3), "residual_m": round(residual, 3), "allowed_m": round(allowed, 3)}


def _pick_primary(members: List[TrackEnd]) -> TrackEnd:
    """GOLD wins over SILVER; within a tier the most-observed track is the identity to keep."""
    return max(members, key=lambda t: (t.is_gold, t.n_obs))


def stitch_sequence(docs: List[dict], p: StitchParams) -> List[dict]:
    """Propose merge groups for one sequence. Returns groups with >= 2 members only.

    Each group: {primary_id, is_gold, member_ids, links:[{from,to,...evidence}]}.
    """
    ends = [e for e in (summarize_track(d, p.vel_window) for d in docs) if e is not None]
    by_id = {e.ped_id: e for e in ends}
    order = sorted(ends, key=lambda e: e.end_ts)

    # Greedy successor assignment: each track claims its single cheapest valid follower, and each
    # follower is claimed at most once (nearest-in-time chaining, high precision).
    successor: Dict[str, str] = {}
    claimed: set[str] = set()
    edges: Dict[tuple, dict] = {}
    for a in order:
        best: Optional[tuple] = None
        for b in ends:
            if b.ped_id == a.ped_id or b.ped_id in claimed or b.start_ts <= a.start_ts:
                continue
            ev = edge_cost(a, b, p)
            if ev is None:
                continue
            key = (ev["residual_m"], b.start_ts)
            if best is None or key < best[0]:
                best = (key, b, ev)
        if best is not None:
            _, b, ev = best
            successor[a.ped_id] = b.ped_id
            claimed.add(b.ped_id)
            edges[(a.ped_id, b.ped_id)] = ev

    # Follow successor links into chains (heads = tracks nobody points to).
    targets = set(successor.values())
    groups: List[dict] = []
    for head in order:
        if head.ped_id in targets or head.ped_id not in successor:
            if head.ped_id in targets:
                continue  # not a chain head
        # walk the chain from this head
        chain_ids: List[str] = []
        cur: Optional[str] = head.ped_id
        while cur is not None:
            chain_ids.append(cur)
            cur = successor.get(cur)
        if len(chain_ids) < 2:
            continue
        members = [by_id[i] for i in chain_ids]
        primary = _pick_primary(members)
        links = [{"from": a, "to": b, **edges[(a, b)]}
                 for a, b in zip(chain_ids, chain_ids[1:])]
        groups.append({
            "primary_id": primary.ped_id,
            "primary_is_gold": primary.is_gold,
            "member_ids": chain_ids,
            "fragment_ids": [i for i in chain_ids if i != primary.ped_id],
            "links": links,
        })
    return groups


def stitch_from_dir(traj_dir: Path, seq_id: str, p: StitchParams) -> List[dict]:
    """Load a sequence's trajectory JSONs and propose merge groups."""
    import json
    docs = []
    for f in sorted(traj_dir.glob(f"{seq_id}_*.json")):
        if f.name.startswith("_"):
            continue
        docs.append(json.loads(f.read_text()))
    return stitch_sequence(docs, p)
