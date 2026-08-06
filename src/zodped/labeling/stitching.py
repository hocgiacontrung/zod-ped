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

Applying a finalized proposal set is `resolve_sequence_tracks` at the bottom of this module: it cuts
junk, folds fragments into their primary IN MEMORY, and hands Steps 2-3 one doc per pedestrian. No
trajectory file is ever rewritten, so Step 1's output stays the reproducible source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


# ---------------------------------------------------------------------------
# Applying a finalized proposal set: cut + merge, in memory
# ---------------------------------------------------------------------------

def load_cut(path: Path) -> Set[Tuple[str, str]]:
    """(seq, ped) keys the SILVER free cut removed as junk. Empty set if the manifest is absent."""
    if not path.exists():
        return set()
    return {(x["seq"], x["ped"]) for x in json.loads(path.read_text())["dropped"]}


def load_groups(path: Path) -> Dict[str, List[dict]]:
    """Finalized stitch proposals, indexed by sequence. Empty dict if the manifest is absent."""
    if not path.exists():
        return {}
    groups: Dict[str, List[dict]] = {}
    for g in json.loads(path.read_text())["groups"]:
        groups.setdefault(g["seq_id"], []).append(g)
    return groups


def merge_docs(primary: dict, fragments: List[dict]) -> dict:
    """Fold `fragments` into `primary`, returning ONE trajectory doc for the whole pedestrian.

    The merged doc keeps the primary's identity and tier — the fragments are the same person, so
    nothing about *who* this is changes. Frames are unioned and re-sorted by timestamp; where two
    fragments claim the same instant the OBSERVED frame wins over a coasted one (a coast is the
    linker's guess, a detection is a measurement), and ties break on more LiDAR support.

    The gap between fragments is left as a genuine hole rather than interpolated: Step 2 tests each
    frame independently, and Step 3 measures `observed_fraction` per window, so an invented bridge
    would inflate both. `stats` is recomputed; `config` gains a `stitched_from` provenance list.
    """
    if not fragments:
        return primary

    best: Dict[str, dict] = {}
    for frame in [f for doc in [primary, *fragments] for f in doc["frames"]]:
        key = frame["timestamp"]
        rank = (bool(frame.get("in_observation")), frame.get("num_lidar_points") or 0)
        if key not in best or rank > (bool(best[key].get("in_observation")),
                                      best[key].get("num_lidar_points") or 0):
            best[key] = frame

    frames = sorted(best.values(), key=lambda f: f["timestamp"])
    observed = sum(bool(f.get("in_observation")) for f in frames)
    merged = dict(primary)
    merged["frames"] = frames
    merged["stats"] = {"n_frames": len(frames), "n_observed": observed,
                       "n_coasted": len(frames) - observed,
                       "observed_fraction": round(observed / len(frames), 4) if frames else 0.0}
    merged["config"] = {**primary.get("config", {}),
                        "stitched_from": [d["pedestrian_id"] for d in fragments]}
    return merged


def resolve_sequence_tracks(paths: List[Path], groups: List[dict],
                            cut: Set[Tuple[str, str]]) -> List[Tuple[str, dict]]:
    """One sequence's trajectory files -> the PEDESTRIANS they represent, cut and stitched.

    Returns (output_filename, doc) pairs so a caller writes one record per pedestrian under the
    primary's name. Three things happen, in order:

      1. cut       — junk tracks are dropped outright, and never merged into anything.
      2. GOLD-primary groups — the SILVER fragments are dropped, NOT folded in. They belong to a
         pedestrian that is already tracked, labeled and human-verified under its GOLD identity;
         rewriting that track here would silently invalidate the verified label keyed to it.
         Healing GOLD with SILVER fragments is a separate, deliberate decision.
      3. SILVER-primary groups — fragments are folded into the primary (merge_docs).

    With no manifests this is just "load every file", so callers can pass empty inputs safely.
    """
    docs = {}
    for path in paths:
        seq, ped = path.name.split("_", 1)[0], path.name.split("_", 1)[1][:-5]
        if (seq, ped) in cut:
            continue
        docs[ped] = (path.name, json.loads(path.read_text()))

    absorbed: Set[str] = set()
    merged_out: List[Tuple[str, dict]] = []
    for g in groups:
        primary_id = g["primary_id"]
        # A GOLD member is NEVER absorbed, whatever the proposal says. Each GOLD track is anchored on
        # its own verified ZOD keyframe box and its human label is keyed to its id, so folding one
        # away would silently delete an annotated pedestrian. A GOLD-GOLD proposal is therefore a
        # false merge (or two ZOD annotations of one person) — either way, not ours to resolve here.
        absorbable = [m for m in g["member_ids"]
                      if m != primary_id and m in docs and not docs[m][1].get("is_in_gold_standard", True)]
        if g["primary_is_gold"]:
            absorbed.update(absorbable)          # silver fragments of a gold ped: dropped, not folded
            continue
        if primary_id not in docs:               # the cut removed the primary; fragments stand alone
            continue
        name, primary = docs[primary_id]
        merged_out.append((name, merge_docs(primary, [docs[m][1] for m in absorbable])))
        absorbed.update(absorbable)
        absorbed.add(primary_id)

    singles = [v for k, v in docs.items() if k not in absorbed]
    return singles + merged_out
