"""Step 3 — sample assembly + v1 ROUGH intent labeling

Materialises the dataset's training unit — one sample per (pedestrian, observation window) —
from the Step-1 trajectories and Step-2 action records of the selected tier:

  * TTE-anchored windows from the action's t_c (crossers: one window per prediction horizon;
    determined non-crossers: one comparison window at closest approach to the ego road)
  * per-window filters: proximity (≤50 m ego / ≤15 m road at the window midpoint) +
    data availability (≥3 LiDAR scans, ≥1 camera frame on disk)
  * v1 ROUGH intent per horizon (t_c ∈ [window_end, window_end + h]) — forward-looking,
    derived straight from the action; behavioral intent is a separate later step
  * per-window geometry (position_ego_rel), image-plane bbox sequences (projected 3D box),
    ego context (speed / heading / turn indicator from vehicle_data.hdf5)

Outputs one JSON per sample (data/annotations/{sample_id}.json), a Parquet index of all scalar
fields (data/annotations/dataset_index.parquet), and a run report.

Usage:
    python scripts/03_assemble_samples.py --max-seqs 2     # smoke test
    python scripts/03_assemble_samples.py                  # full GOLD set
    python scripts/03_assemble_samples.py --tier all       # GOLD + SILVER (after SILVER QC)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from _common import DEFAULT_TRAJ_DIR, ROOT, SEQ_DIR, add_tier_arg, tier_matches
from zodped.labeling.samples import (
    AssemblyConfig,
    SequenceContext,
    TrackTimeline,
    assemble_track_samples,
    key_pedestrian,
)

DEFAULT_ACTIONS_DIR = ROOT / "data" / "processed" / "actions"
DEFAULT_OUT_DIR = ROOT / "data" / "annotations"
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "samples_run_report.json"


def _group_by_sequence(traj_dir: Path) -> Dict[str, List[Path]]:
    """Map seq_id -> sorted trajectory JSON paths (filename is {seq}_{ped}.json)."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(traj_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        groups[path.name.split("_", 1)[0]].append(path)
    return groups


def process_sequence(
    seq_id: str, traj_paths: List[Path], cfg: AssemblyConfig, args: argparse.Namespace
) -> tuple[List[dict], Counter]:
    """Assemble and write every sample of one sequence. Returns (samples, skip counts)."""
    skips: Counter = Counter()

    # Load the tier's tracks + their Step-2 action records first (cheap), so a sequence whose
    # tracks are all skippable still pays for context setup only once it has work to do.
    tracks: List[tuple[dict, dict]] = []
    for path in traj_paths:
        trajectory = json.loads(path.read_text())
        if not tier_matches(trajectory, args.tier):
            continue
        action_path = args.actions_dir / path.name
        if not action_path.exists():
            skips["no_action_record"] += 1       # Step 2 has not labeled this track yet
            continue
        tracks.append((trajectory, json.loads(action_path.read_text())))
    if not tracks:
        return [], skips

    ctx = SequenceContext(seq_id, SEQ_DIR / seq_id)
    timelines = {t["pedestrian_id"]: TrackTimeline(t) for t, _ in tracks if t["frames"]}
    key_ped = key_pedestrian(timelines, ctx.em)

    samples: List[dict] = []
    for trajectory, action in tracks:
        ped_id = trajectory["pedestrian_id"]
        if ped_id not in timelines:              # frameless track: undetermined by definition
            skips["undetermined_action"] += 1
            continue
        track_samples, track_skips = assemble_track_samples(
            trajectory, action, timelines[ped_id], ctx, cfg,
            num_pedestrians_in_scene=len(timelines),
            is_key_pedestrian=(ped_id == key_ped),
        )
        skips.update(track_skips)
        for sample in track_samples:
            (args.out_dir / f"{sample['sample_id']}.json").write_text(json.dumps(sample))
        samples.extend(track_samples)
    return samples, skips


def _index_row(s: dict) -> dict:
    """Flatten one sample's scalar fields into a Parquet index row."""
    row = {
        "sample_id": s["sample_id"],
        "sequence_id": s["sequence_id"],
        "pedestrian_id": s["pedestrian_id"],
        "window_start_timestamp": s["window_start_timestamp"],
        "window_end_timestamp": s["window_end_timestamp"],
        "window_kind": s["window_kind"],
        "anchor_tte_s": s["anchor_tte_s"],
        "intent_label": s["intent"]["label"],
        "intent_confidence": s["intent"]["confidence"],
        "crosses_ego_road": s["action"]["crosses_ego_road"],
        "crossing_frame_timestamp": s["action"]["crossing_frame_timestamp"],
        "n_camera_frames": len(s["multimodal"]["camera_frames"]),
        "n_lidar_scans": len(s["multimodal"]["lidar_scans"]),
        "split": s["metadata"]["split"],
        "label_confidence_tier": s["metadata"]["label_confidence_tier"],
        "is_in_gold_standard": s["metadata"]["is_in_gold_standard"],
        "num_pedestrians_in_scene": s["metadata"]["num_pedestrians_in_scene"],
        "is_key_pedestrian": s["metadata"]["is_key_pedestrian"],
        "location": s["metadata"]["location"],
        "collection_date": s["metadata"]["collection_date"],
    }
    row.update({f"intent_h{h.replace('.', '_')}s": lbl
                for h, lbl in s["intent"]["labels_by_horizon"].items()})
    row.update({k: s["context"][k] for k in (
        "ego_speed_ms", "ego_heading_deg", "turn_indicator", "distance_to_ego_m",
        "distance_to_road_m", "occlusion", "observed_fraction_window")})
    return row


def _summarise(samples: List[dict], cfg: AssemblyConfig) -> dict:
    per_horizon = {}
    for h in cfg.prediction_horizons_s:
        key = f"{h:.1f}"
        crossing = sum(s["intent"]["labels_by_horizon"][key] == "crossing" for s in samples)
        per_horizon[key] = {
            "crossing": crossing,
            "not_crossing": len(samples) - crossing,
            "crossing_ratio": round(crossing / len(samples), 4) if samples else None,
        }
    return {
        "n_samples": len(samples),
        "n_tte_anchored": sum(s["window_kind"] == "tte_anchored" for s in samples),
        "n_closest_approach": sum(s["window_kind"] == "closest_approach" for s in samples),
        "n_pedestrians": len({(s["sequence_id"], s["pedestrian_id"]) for s in samples}),
        "per_horizon": per_horizon,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="Step 1 trajectory JSONs")
    ap.add_argument("--actions-dir", type=Path, default=DEFAULT_ACTIONS_DIR, help="Step 2 action records")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-sample JSONs + Parquet index")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="run report")
    add_tier_arg(ap, default="gold")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    args = ap.parse_args()
    cfg = AssemblyConfig()   # schema v0.2 CONFIRMED values; change the schema first, then here

    groups = _group_by_sequence(args.traj_dir)
    seq_ids = sorted(groups)
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Step 3 (sample assembly): {sum(len(groups[s]) for s in seq_ids)} candidate tracks over "
          f"{len(seq_ids)} sequences (tier: {args.tier}) → {args.out_dir}")

    all_samples: List[dict] = []
    skips: Counter = Counter()
    failures: List[dict] = []
    for i, seq_id in enumerate(seq_ids):
        try:
            samples, seq_skips = process_sequence(seq_id, groups[seq_id], cfg, args)
            all_samples.extend(samples)
            skips.update(seq_skips)
        except Exception as exc:                 # continue-on-error, like Steps 1-2
            failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences")

    index_path = args.out_dir / "dataset_index.parquet"
    pd.DataFrame([_index_row(s) for s in all_samples]).to_parquet(index_path, index=False)

    summary = _summarise(all_samples, cfg)
    report = {
        "tier": args.tier,
        "config": cfg.as_dict(),
        "n_sequences": len(seq_ids),
        "summary": summary,
        "skip_counts": dict(sorted(skips.items())),
        "failures": failures,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print(f"\nDone. {summary['n_samples']} samples from {summary['n_pedestrians']} pedestrians "
          f"({summary['n_tte_anchored']} TTE-anchored / {summary['n_closest_approach']} comparison, "
          f"{len(failures)} seq failures).")
    for h, stats in summary["per_horizon"].items():
        print(f"  horizon {h}s: {stats['crossing']} crossing / {stats['not_crossing']} not "
              f"(ratio {stats['crossing_ratio']})")
    print(f"  skips: {dict(sorted(skips.items()))}")
    print(f"  index → {index_path}\n  report → {args.report_path}")


if __name__ == "__main__":
    main()
