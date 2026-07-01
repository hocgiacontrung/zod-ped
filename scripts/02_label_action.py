"""Step 2 — track-level ACTION labeling

Reads every Step-1 GOLD trajectory (data/processed/trajectories/{seq}_{ped}.json) and writes one
action record per track (data/processed/actions/{seq}_{ped}.json) with the crossing ACTION:

  * crosses_ego_corridor (primary) + crossing_frame_timestamp (t_c) + ego_distance_at_crossing_m
  * crosses_ego_road (complement, via the keyframe camera)
  * status = determined | undetermined  (EMPTY tracks → undetermined; kept + flagged, never forced)

This is pure geometry over the smoothed world-frame trajectory. The crossing onset t_c drives Step-3 TTE anchoring. 
Runs on GOLD now, and on SILVER tracks unchanged once Step 1b exists. See docs/PIPELINE.md and the
geometry in src/zodped/labeling/actions.py.

Usage:
    python scripts/02_label_action.py --max-seqs 2     # smoke test
    python scripts/02_label_action.py                  # full GOLD set
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.actions import (
    CORRIDOR_HALF_WIDTH_M, CORRIDOR_LOOKAHEAD_M, CORRIDOR_MIN_FORWARD_M,
    label_track_action, load_ego_road_polygons,
)
from zodped.utils.ego_motion import get_T_world_lidar, load_ego_motion
from zodped.utils.projection import load_calibration

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "actions"
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "actions_run_report.json"


def _group_by_sequence(traj_dir: Path) -> Dict[str, List[Path]]:
    """Map seq_id -> sorted trajectory JSON paths (filename is {seq}_{ped}.json)."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(traj_dir.glob("*.json")):
        if path.name.startswith("_"):           # skip _step2_report.json and friends
            continue
        groups[path.name.split("_", 1)[0]].append(path)
    return groups


def process_sequence(seq_id: str, traj_paths: List[Path], args: argparse.Namespace) -> List[dict]:
    """Label every track in one sequence; write a JSON per pedestrian. Returns the action records."""
    seq_dir = SEQ_DIR / seq_id
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])
    T_world_lidar_kf = get_T_world_lidar(em, lidar_ext, keyframe_ts)

    road_path = seq_dir / "annotations" / "ego_road.json"
    polygons = load_ego_road_polygons(json.loads(road_path.read_text())) if road_path.exists() else None

    records: List[dict] = []
    for path in traj_paths:
        trajectory = json.loads(path.read_text())
        record = label_track_action(
            trajectory, em, calib, polygons, T_world_lidar_kf,
            half_width_m=args.corridor_half_width, lookahead_m=args.corridor_lookahead,
            min_forward_m=args.corridor_min_forward,
        )
        (args.out_dir / path.name).write_text(json.dumps(record))
        records.append(record)
    return records


def _summarise(records: List[dict]) -> dict:
    """Aggregate counts + the crossing rate over DETERMINED tracks (the labelable set)."""
    determined = [r for r in records if r["status"] == "determined"]
    corr = sum(bool(r["crosses_ego_corridor"]) for r in determined)
    road = sum(bool(r["crosses_ego_road"]) for r in determined)
    return {
        "n_tracks": len(records),
        "n_determined": len(determined),
        "n_undetermined": len(records) - len(determined),
        "n_crosses_ego_corridor": corr,
        "n_crosses_ego_road": road,
        "corridor_crossing_rate": round(corr / len(determined), 4) if determined else None,
        "road_crossing_rate": round(road / len(determined), 4) if determined else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="Step 1 trajectory JSONs")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-track action JSONs")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="run report (out of the data dir)")
    ap.add_argument("--corridor-half-width", type=float, default=CORRIDOR_HALF_WIDTH_M, help="ego-corridor half-width (m)")
    ap.add_argument("--corridor-lookahead", type=float, default=CORRIDOR_LOOKAHEAD_M, help="max forward range counted as in-front (m)")
    ap.add_argument("--corridor-min-forward", type=float, default=CORRIDOR_MIN_FORWARD_M, help="near edge of the corridor strip (m)")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    args = ap.parse_args()

    groups = _group_by_sequence(args.traj_dir)
    seq_ids = sorted(groups)
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Step 2 (action labeling): {sum(len(groups[s]) for s in seq_ids)} tracks over "
          f"{len(seq_ids)} sequences → {args.out_dir}")

    all_records: List[dict] = []
    failures: List[dict] = []
    for i, seq_id in enumerate(seq_ids):
        try:
            all_records.extend(process_sequence(seq_id, groups[seq_id], args))
        except Exception as exc:                 # continue-on-error, like Step 1
            failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences")

    summary = _summarise(all_records)
    report = {
        "config": {
            "corridor_half_width_m": args.corridor_half_width,
            "corridor_lookahead_m": args.corridor_lookahead,
            "corridor_min_forward_m": args.corridor_min_forward,
        },
        "n_sequences": len(seq_ids),
        "summary": summary,
        "failures": failures,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print(f"\nDone. {summary['n_tracks']} tracks "
          f"({summary['n_determined']} determined / {summary['n_undetermined']} undetermined, "
          f"{len(failures)} seq failures).")
    print(f"  corridor crossings: {summary['n_crosses_ego_corridor']} "
          f"(rate {summary['corridor_crossing_rate']} over determined)")
    print(f"  ego_road crossings: {summary['n_crosses_ego_road']} "
          f"(rate {summary['road_crossing_rate']} over determined)")
    print(f"  report → {args.report_path}")


if __name__ == "__main__":
    main()
