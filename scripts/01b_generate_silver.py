"""Step 1b — SILVER-tier trajectory generation (detector-birth tracks).

GOLD (scripts/01_generate_trajectories.py) tracks only the pedestrians ZOD annotated at the keyframe.
The 2D detector sees many more people than that (the GOLD 2D-detection coverage is ~24%). SILVER
recovers them: it births tracks from the frustum candidates GOLD left UNCLAIMED, with no verified
anchor — so every SILVER track is flagged `is_in_gold_standard=false` and carries a lower confidence
tier and a pedestrian size PRIOR (no keyframe extent exists). Steps 2–4 then run over GOLD and SILVER
with the same code (the tier only rides along as a flag). See docs/PIPELINE.md "Two-tier labels".

Pipeline (per sequence):
  1. Build the candidate pool once (same frustum lift as GOLD; zodped.labeling.frustum).
  2. Load the sequence's finished GOLD tracks and CLAIM the pool candidates within `--claim-radius`
     of a GOLD observed position at the same scan — the residual is "everyone GOLD didn't track".
  3. Birth SILVER tracks from the residual pool (online MOT; zodped.labeling.tracker
     .birth_tracks_from_residual_pool), assemble per-frame boxes from the size prior, and write one
     JSON per track next to GOLD (data/processed/trajectories/{seq}_silver{n}.json).

Requires the Step-0 detection cache (scripts/00_detect.py) and a finished GOLD run (Step 1a).

Usage:
    python scripts/01b_generate_silver.py --max-seqs 3      # smoke test
    python scripts/01b_generate_silver.py                   # full SILVER pass
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (
    DEFAULT_TRAJ_DIR, ROOT, SEQ_DIR, add_pool_args, load_seq_ids, pool_config, pool_kwargs,
)
from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.boxes import assemble_track_boxes
from zodped.labeling.detection_cache import cache_path, cached_detector, load_detections
from zodped.labeling.frustum import FrameCandidates, build_candidate_pool
from zodped.labeling.tracker import birth_tracks_from_residual_pool
from zodped.utils.ego_motion import load_ego_motion
from zodped.utils.projection import load_calibration

DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "silver_run_report.json"


def _gold_positions_by_ts(traj_dir: Path, seq_id: str) -> Dict[float, List[np.ndarray]]:
    """Scan-timestamp -> GOLD world positions, over the sequence's finished GOLD tracks.

    Includes coasted frames as well as observed ones: the GOLD pedestrian is still at that estimated
    position while the linker coasts, so claiming there stops SILVER re-birthing it in its gaps.
    """
    by_ts: Dict[float, List[np.ndarray]] = defaultdict(list)
    for path in sorted(traj_dir.glob(f"{seq_id}_*.json")):
        doc = json.loads(path.read_text())
        if not doc.get("is_in_gold_standard", True):        # skip prior SILVER output
            continue
        for f in doc["frames"]:
            by_ts[round(parse_zod_ts(f["timestamp"]), 6)].append(np.asarray(f["position_world"]))
    return by_ts


def _residual_pool(pool: List[FrameCandidates], gold_by_ts: Dict[float, List[np.ndarray]],
                   claim_radius: float) -> List[FrameCandidates]:
    """Drop candidates within `claim_radius` (horizontal) of any GOLD position at the same scan."""
    residual: List[FrameCandidates] = []
    for ts, cand_world, cand_n in pool:
        gold = gold_by_ts.get(round(ts, 6))
        if not len(cand_world) or not gold:
            residual.append((ts, cand_world, cand_n))
            continue
        gold_xy = np.array(gold)[:, :2]
        keep = np.array([
            bool((np.linalg.norm(gold_xy - c[:2], axis=1) > claim_radius).all())
            for c in cand_world
        ])
        residual.append((ts, cand_world[keep], cand_n[keep]))
    return residual


def _silver_doc(seq_id: str, ped_id: str, keyframe_iso: str, frames: List[dict],
                prior_size: List[float], config: dict) -> dict:
    """Assemble one SILVER per-pedestrian document (detector-born; no verified anchor)."""
    n_obs = sum(f["in_observation"] for f in frames)
    return {
        "schema": "trajectory/v0.2",
        "sequence_id": seq_id,
        "pedestrian_id": ped_id,
        "label_confidence_tier": "medium",
        "is_in_gold_standard": False,
        "keyframe_timestamp": keyframe_iso,
        "anchor": {"occlusion": "n/a", "source": "detector_birth", "size_prior_lwh": prior_size},
        "stats": {
            "n_frames": len(frames),
            "n_observed": int(n_obs),
            "n_coasted": len(frames) - int(n_obs),
            "observed_fraction": round(n_obs / len(frames), 4) if frames else None,
        },
        "config": config,
        "frames": frames,
    }


def process_sequence(seq_id: str, args: argparse.Namespace, config: dict) -> dict:
    """Birth + write SILVER tracks for one sequence. Returns a summary dict."""
    det_path = cache_path(args.det_dir, seq_id)
    if not det_path.exists():
        return {"seq_id": seq_id, "n_silver": 0, "n_pool_frames": 0, "error": "no detection cache"}

    seq_dir = SEQ_DIR / seq_id
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    keyframe_iso = json.loads((seq_dir / "info.json").read_text())["keyframe_time"]

    detector = cached_detector(load_detections(det_path, min_conf=args.conf))
    pool = build_candidate_pool(seq_dir, detector, em, lidar_ext, calib, **pool_kwargs(args))
    if not pool:
        return {"seq_id": seq_id, "n_silver": 0, "n_pool_frames": 0}

    residual = _residual_pool(pool, _gold_positions_by_ts(args.traj_dir, seq_id), args.claim_radius)
    tracks = birth_tracks_from_residual_pool(
        residual, gate_mahal2=args.gate_mahal2, max_consecutive_misses=args.max_misses,
        meas_sigma=args.meas_sigma, min_support=args.min_support, min_duration_s=args.min_duration,
    )

    # Idempotent re-runs: clear this sequence's prior SILVER output before writing.
    for stale in args.traj_dir.glob(f"{seq_id}_silver*.json"):
        stale.unlink()

    prior_size = list(args.prior_size)
    for n, frames in enumerate(tracks):
        assemble_track_boxes(frames, np.asarray(prior_size), min_speed=args.min_speed)
        ped_id = f"silver{n:03d}"
        doc = _silver_doc(seq_id, ped_id, keyframe_iso, frames, prior_size, config)
        (args.traj_dir / f"{seq_id}_{ped_id}.json").write_text(json.dumps(doc))

    return {"seq_id": seq_id, "n_silver": len(tracks), "n_pool_frames": len(pool)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="GOLD in / SILVER out (same dir)")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    add_pool_args(ap)                              # pool/linker params shared verbatim with Step 1a
    ap.add_argument("--claim-radius", type=float, default=1.5, help="GOLD claims candidates within this radius (m)")
    ap.add_argument("--min-support", type=int, default=4, help="min real observations to confirm a SILVER track")
    ap.add_argument("--min-duration", type=float, default=1.0, help="min real-observation span to confirm (s)")
    ap.add_argument("--prior-size", type=float, nargs=3, default=[0.6, 0.6, 1.7],
                    metavar=("L", "W", "H"), help="pedestrian size prior (m), since SILVER has no keyframe extent")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    args = ap.parse_args()

    args.traj_dir.mkdir(parents=True, exist_ok=True)
    seq_ids = load_seq_ids(args.max_seqs)

    config = {**pool_config(args), "tier": "silver",
              "claim_radius_m": args.claim_radius, "min_support": args.min_support,
              "min_duration_s": args.min_duration, "prior_size_lwh": list(args.prior_size)}
    print(f"Step 1b (SILVER): birthing detector tracks over {len(seq_ids)} sequences → {args.traj_dir}")

    summaries: List[dict] = []
    failures: List[dict] = []
    total = 0
    for i, seq_id in enumerate(seq_ids):
        try:
            s = process_sequence(seq_id, args, config)
            summaries.append(s)
            total += s["n_silver"]
        except Exception as exc:                 # continue-on-error, like the other steps
            failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences  (SILVER tracks: {total})")

    report = {"config": config, "n_sequences": len(seq_ids), "n_silver_tracks": total,
              "per_sequence": summaries, "failures": failures}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDone. {total} SILVER tracks from {len(seq_ids)} sequences "
          f"({len(failures)} failed). Report → {args.report_path}")


if __name__ == "__main__":
    main()
