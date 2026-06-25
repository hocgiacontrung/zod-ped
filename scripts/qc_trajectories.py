"""Step 1 QC — score every GOLD trajectory and rank a manual-review queue.

Turns the qualitative "watch the demo video" pass into a scalable, quantitative gate. Reads the
per-pedestrian trajectory JSONs (data/processed/trajectories/) plus the Step 1 run report (for each
sequence's pool-frame count) and, per track, derives quality signals that are computable WITHOUT
re-running the detector:

  - span_fraction   : n_frames / n_pool_frames — did the track cover its clip, or did the linker
                      lose it early? (A pass terminates after `max_consecutive_misses` coasts, so a
                      short span is the strongest "the track died" signal.)
  - real_frac       : fraction of frames with a REAL frustum measurement (num_lidar_points >= 0).
                      Excludes the keyframe anchor seed, so a zero-measurement stub reads 0.0 (flagged
                      EMPTY) rather than a misleading ~0.09. (observed_frac, kept for continuity,
                      still counts the anchor.)
  - longest_coast   : longest consecutive coast run, and how many distinct coast runs there are.
  - median_lidar_pts: median in-frustum LiDAR points on observed frames — low => far/sparse/noisy lift.
  - median_conf     : median Kalman confidence on observed frames.
  - max_speed       : fastest inter-frame world-frame speed — implausible spikes flag mis-association.

Each track gets a list of FLAGS (thresholds are CLI-tunable) and the corpus is summarised by flag
counts and by anchor occlusion — which is the data-driven read on "is YOLO11 good enough":

  - low coverage concentrated on None/Light-occlusion anchors  -> detector misses (improve detector)
  - low coverage concentrated on Heavy/VeryHeavy anchors        -> genuine occlusion (expected; filter)
  - low median_lidar_pts on the flagged tracks                  -> frustum starvation / range, not 2D

True per-coast attribution (detector-miss vs gate-reject) needs the candidate pool rebuilt
(re-run the detector); drill into a specific bad track with scripts/viz_render_video.py.

Usage:
    python scripts/qc_trajectories.py                       # score all tracks, print queue + summary
    python scripts/qc_trajectories.py --seq 000007          # one sequence
    python scripts/qc_trajectories.py --top 40 --csv data/processed/review/qc.csv
    python scripts/qc_trajectories.py --min-span 0.6 --min-observed 0.5 --max-speed 3.5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "trajectories_run_report.json"


def _pool_frames_by_seq(report_path: Path) -> Dict[str, int]:
    """Map seq_id -> n_pool_frames from the Step 1 run report (for span_fraction). Empty if absent."""
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    return {s["seq_id"]: s["n_pool_frames"] for s in report.get("per_sequence", [])}


def _coast_runs(observed: List[bool]) -> tuple[int, int]:
    """Return (longest consecutive coast run, number of distinct coast runs)."""
    longest = run = n_runs = 0
    for obs in observed:
        if obs:
            run = 0
        else:
            run += 1
            if run == 1:
                n_runs += 1
            longest = max(longest, run)
    return longest, n_runs


def score_track(doc: dict, n_pool_frames: Optional[int]) -> dict:
    """Derive QC metrics for one trajectory document (no detector re-run)."""
    frames = doc["frames"]
    observed = [bool(f["in_observation"]) for f in frames]
    ts = np.array([parse_zod_ts(f["timestamp"]) for f in frames])
    pos = np.array([f["position_world"] for f in frames], dtype=np.float64)

    n_frames = len(frames)
    n_obs = int(sum(observed))
    # A REAL measurement is an observed frame backed by frustum points (num_lidar_points >= 0).
    # The keyframe anchor seed is marked in_observation but carries num_lidar_points == -1, so it is
    # NOT a real measurement — counting it inflates coverage and hides zero-measurement stubs.
    obs_pts = [f["num_lidar_points"] for f, o in zip(frames, observed) if o and f["num_lidar_points"] >= 0]
    obs_conf = [f["kalman_confidence"] for f, o in zip(frames, observed) if o]
    n_real = len(obs_pts)

    # Inter-frame world-frame speed (m/s); guards against zero/negative dt.
    if n_frames >= 2:
        dt = np.diff(ts)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        speed = step[dt > 0] / dt[dt > 0]
        max_speed = float(speed.max()) if speed.size else 0.0
    else:
        max_speed = 0.0

    longest_coast, n_coast_runs = _coast_runs(observed)
    return {
        "seq": doc["sequence_id"],
        "ped": doc["pedestrian_id"][:8],
        "occlusion": doc.get("anchor", {}).get("occlusion", "?"),
        "n_frames": n_frames,
        "duration_s": round(float(ts[-1] - ts[0]), 2) if n_frames >= 2 else 0.0,
        "span_frac": round(n_frames / n_pool_frames, 3) if n_pool_frames else None,
        "observed_frac": round(n_obs / n_frames, 3) if n_frames else 0.0,
        "real_frac": round(n_real / n_frames, 3) if n_frames else 0.0,  # excludes the anchor seed
        "n_real": n_real,
        "longest_coast": longest_coast,
        "n_coast_runs": n_coast_runs,
        "median_lidar_pts": int(np.median(obs_pts)) if obs_pts else 0,
        "median_conf": round(float(np.median(obs_conf)), 3) if obs_conf else 0.0,
        "max_speed": round(max_speed, 2),
    }


def flag_track(m: dict, args: argparse.Namespace) -> List[str]:
    """Return the QC flags a track trips, given the CLI thresholds."""
    flags = []
    # EMPTY supersedes LOW_COVERAGE: a track with zero real measurements is just the GT anchor
    # coasted out — there is nothing to review, it should be dropped (or rescued by a better detector).
    if m["n_real"] == 0:
        flags.append("EMPTY")
    elif m["real_frac"] < args.min_observed:
        flags.append("LOW_COVERAGE")
    if m["duration_s"] < args.min_duration:
        flags.append("SHORT")
    if m["median_lidar_pts"] < args.min_lidar_pts:
        flags.append("SPARSE_LIFT")
    if m["median_conf"] < args.min_conf:
        flags.append("LOW_CONF")
    if m["max_speed"] > args.max_speed:
        flags.append("JITTER")
    return flags


def _print_summary(rows: List[dict]) -> None:
    """Corpus-level QC summary: flag counts and coverage by anchor occlusion."""
    n = len(rows)
    print(f"\n=== QC summary over {n} tracks ===")

    def q(key: str) -> str:
        vals = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        if not vals.size:
            return "n/a"
        p = np.percentile(vals, [10, 50, 90])
        return f"p10={p[0]:.2f}  median={p[1]:.2f}  p90={p[2]:.2f}"

    n_empty = sum(1 for r in rows if r["n_real"] == 0)
    print(f"  EMPTY tracks    : {n_empty}/{n} ({100 * n_empty / n:.0f}%)  (zero real measurements — anchor-only stubs; drop)")
    print(f"  duration_s      : {q('duration_s')}")
    print(f"  span_fraction   : {q('span_frac')}  (context: peds aren't present the whole clip)")
    print(f"  real_frac       : {q('real_frac')}  (measured frames / total; excludes the anchor seed)")
    print(f"  median_lidar_pts: {q('median_lidar_pts')}")

    flag_counts: Dict[str, int] = {}
    n_flagged = 0
    for r in rows:
        if r["flags"]:
            n_flagged += 1
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    print(f"  flagged tracks  : {n_flagged}/{n} ({100 * n_flagged / n:.0f}%)" if n else "  flagged: 0")
    for f, c in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
        print(f"      {f:<13} {c}")

    print("  coverage by anchor occlusion (the 'is YOLO good enough' read):")
    buckets: Dict[str, List[float]] = {}
    for r in rows:
        buckets.setdefault(str(r["occlusion"]), []).append(r["real_frac"])
    for occ, vals in sorted(buckets.items()):
        print(f"      {occ:<12} n={len(vals):<4} mean_real_frac={np.mean(vals):.3f}")


def _print_queue(rows: List[dict], top: int) -> None:
    """Print the worst `top` tracks (most flags, then lowest span) as a review queue."""
    ranked = sorted(rows, key=lambda r: (-len(r["flags"]), r["real_frac"]))
    print(f"\n=== review queue: worst {min(top, len(ranked))} tracks ===")
    hdr = f"{'seq':<8}{'ped':<10}{'occ':<11}{'dur_s':>6}{'span':>6}{'real':>6}{'lidar':>7}{'conf':>6}{'maxv':>7}  flags"
    print(hdr)
    print("-" * len(hdr))
    for r in ranked[:top]:
        span = f"{r['span_frac']:.2f}" if r["span_frac"] is not None else "  - "
        print(f"{r['seq']:<8}{r['ped']:<10}{str(r['occlusion']):<11}{r['duration_s']:>6.1f}{span:>6}"
              f"{r['real_frac']:>6.2f}{r['median_lidar_pts']:>7}{r['median_conf']:>6.2f}"
              f"{r['max_speed']:>7.2f}  {','.join(r['flags'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Step 1 run report (for span_fraction)")
    ap.add_argument("--seq", default=None, help="restrict to one sequence id, e.g. 000007")
    ap.add_argument("--top", type=int, default=30, help="how many worst tracks to print")
    ap.add_argument("--csv", type=Path, default=None, help="optional path to write the full per-track table")
    ap.add_argument("--min-duration", type=float, default=3.0, help="flag SHORT below this tracked duration (s)")
    ap.add_argument("--min-observed", type=float, default=0.5, help="flag LOW_COVERAGE below this observed fraction")
    ap.add_argument("--min-lidar-pts", type=int, default=5, help="flag SPARSE_LIFT below this median in-frustum point count")
    ap.add_argument("--min-conf", type=float, default=0.3, help="flag LOW_CONF below this median Kalman confidence")
    ap.add_argument("--max-speed", type=float, default=3.5, help="flag JITTER above this inter-frame speed (m/s)")
    args = ap.parse_args()

    pool_frames = _pool_frames_by_seq(args.report_path)
    pattern = f"{args.seq}_*.json" if args.seq else "*.json"
    files = [f for f in sorted(args.traj_dir.glob(pattern)) if not f.name.startswith("_")]
    if not files:
        raise SystemExit(f"No trajectory JSONs matching {pattern} in {args.traj_dir}")

    rows: List[dict] = []
    for f in files:
        doc = json.loads(f.read_text())
        m = score_track(doc, pool_frames.get(doc["sequence_id"]))
        m["flags"] = flag_track(m, args)
        rows.append(m)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        cols = ["seq", "ped", "occlusion", "n_frames", "duration_s", "span_frac", "observed_frac",
                "real_frac", "n_real", "longest_coast", "n_coast_runs", "median_lidar_pts",
                "median_conf", "max_speed", "flags"]
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "flags": "|".join(r["flags"])})
        print(f"wrote per-track table ({len(rows)} rows) → {args.csv}")

    _print_queue(rows, args.top)
    _print_summary(rows)


if __name__ == "__main__":
    main()
