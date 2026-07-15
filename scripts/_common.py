"""Shared CLI plumbing for the pipeline scripts: repo paths, the frustum-pool parameter
group, and the GOLD/SILVER tier filter.

The frustum-pool + linker parameters MUST stay identical between Step 1a
(scripts/01_generate_trajectories.py) and Step 1b (scripts/01b_generate_silver.py): SILVER claims
GOLD's candidates out of the SAME pool, so if the two scripts built their pools with different
parameters the residual would silently contain (or miss) candidates GOLD never saw. Defining the
group once makes that desync impossible.

SILVER tracks land in the same trajectories dir as GOLD (distinguished by
`is_in_gold_standard=false`), so every consumer that globs that dir chooses its population
explicitly via `--tier {gold,silver,all}` (add_tier_arg / tier_of / tier_matches) instead of
silently processing whatever is on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from zodped.dataset.keyframe import MAX_LIDAR_GAP_S

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_DET_DIR = ROOT / "data" / "processed" / "detections"       # Step 0 2D-box cache
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"    # Step 1 output (GOLD + SILVER)

TIERS = ("gold", "silver", "all")


def add_pool_args(ap: argparse.ArgumentParser) -> None:
    """Frustum candidate-pool + KF/RTS-linker parameters shared verbatim by Steps 1a and 1b."""
    ap.add_argument("--camera", default="camera_front_blur")
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    ap.add_argument("--max-gap", type=float, default=MAX_LIDAR_GAP_S, help="max image↔scan gap (s)")
    ap.add_argument("--box-shrink", type=float, default=0.6, help="kept central width fraction of each 2D box")
    ap.add_argument("--slab", type=float, default=1.5, help="nearest-depth slab thickness (m)")
    ap.add_argument("--min-pts", type=int, default=3, help="min in-frustum LiDAR points to lift a box")
    ap.add_argument("--gate-mahal2", type=float, default=9.0, help="squared-Mahalanobis association gate")
    ap.add_argument("--max-misses", type=int, default=5, help="consecutive coasted frames before a pass/track ends")
    ap.add_argument("--meas-sigma", type=float, default=0.3, help="frustum measurement noise std (m)")
    ap.add_argument("--min-speed", type=float, default=0.3, help="speed below which box yaw is filled, not from velocity (m/s)")
    ap.add_argument("--det-dir", type=Path, default=DEFAULT_DET_DIR, help="Step 0 2D-detection cache dir")


def pool_kwargs(args: argparse.Namespace) -> dict:
    """Keyword args for zodped.labeling.frustum.build_candidate_pool, from the shared arg group."""
    return {"camera": args.camera, "max_gap": args.max_gap, "box_shrink": args.box_shrink,
            "slab": args.slab, "min_pts": args.min_pts}


def pool_config(args: argparse.Namespace) -> dict:
    """The shared pool/linker parameters as recorded in run reports and output docs."""
    return {"detector": "frustum(2d+lidar)", "conf": args.conf, "max_gap_s": args.max_gap,
            "box_shrink": args.box_shrink, "slab_m": args.slab, "min_pts": args.min_pts,
            "gate_mahal2": args.gate_mahal2, "max_consecutive_misses": args.max_misses,
            "meas_sigma": args.meas_sigma, "min_speed": args.min_speed}


def load_seq_ids(max_seqs: int | None = None) -> List[str]:
    """Sequence ids of the pedestrian working set, optionally capped for a smoke test."""
    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    return seq_ids[:max_seqs] if max_seqs else seq_ids


def add_tier_arg(ap: argparse.ArgumentParser, default: str = "gold") -> None:
    """`--tier {gold,silver,all}` — which trajectory population a run covers (explicit, not implicit)."""
    ap.add_argument("--tier", choices=TIERS, default=default,
                    help=f"which trajectory tier to process (default: {default})")


def tier_of(doc: dict) -> str:
    """A trajectory/action doc's tier, from its is_in_gold_standard flag."""
    return "gold" if doc.get("is_in_gold_standard", True) else "silver"


def tier_matches(doc: dict, tier: str) -> bool:
    """Whether a doc belongs to the requested tier selection."""
    return tier == "all" or tier_of(doc) == tier
