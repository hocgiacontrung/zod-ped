"""Find the best pedestrians to feature in a demo — especially "tracked through occlusion".

Scans the Step 1 trajectories and ranks pedestrians whose track BRIDGES an occlusion: an internal
run of coasted frames (detector matched nothing) with observed frames on BOTH sides, i.e. the
constant-velocity linker carried the track across a gap and re-acquired the same person. These make
the most convincing demo figures (render the chosen sequence with scripts/viz_render_sequence_video.py).

A candidate must clear plausibility/quality gates so we don't showcase a broken track:
  * longest bridged gap >= --min-gap frames (the occlusion actually bridged),
  * observed_fraction >= --min-obs (otherwise it's mostly guesswork, not a real track),
  * path length >= --min-path m (the pedestrian actually moves — interesting),
  * max observed inter-frame step <= --max-step m (a big jump flags a likely ID swap).

Ranked by bridged-gap duration, then path length. Prints the exact render command per candidate.

Usage:
    python scripts/viz_find_demo_pedestrians.py                 # top occlusion-bridged tracks
    python scripts/viz_find_demo_pedestrians.py --top 15 --min-gap 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"


def _longest_bridged_gap(obs: np.ndarray, ts: np.ndarray) -> tuple:
    """Longest INTERNAL coast run (observed on both sides). Returns (n_frames, seconds)."""
    best_n, best_s = 0, 0.0
    i, n = 0, len(obs)
    while i < n:
        if obs[i]:
            i += 1
            continue
        j = i
        while j < n and not obs[j]:
            j += 1
        # [i, j) is a coast run; bridged iff it has observed frames before AND after
        if i > 0 and j < n and (j - i) > best_n:
            best_n = j - i
            best_s = float(ts[j - 1] - ts[i])
        i = j
    return best_n, best_s


def _summarize(doc: dict) -> dict:
    frames = doc["frames"]
    obs = np.array([f["in_observation"] for f in frames])
    pos = np.array([f["position_world"] for f in frames])
    ts = np.array([_parse(f["timestamp"]) for f in frames])

    observed = pos[obs]
    steps = np.linalg.norm(np.diff(observed, axis=0), axis=1) if len(observed) > 1 else np.zeros(0)
    gap_n, gap_s = _longest_bridged_gap(obs, ts)
    return {
        "seq": doc["sequence_id"],
        "ped": doc["pedestrian_id"][:8],
        "n": len(frames),
        "obs_frac": doc["stats"]["observed_fraction"] or 0.0,
        "gap_frames": gap_n,
        "gap_s": round(gap_s, 2),
        "path_m": round(float(steps.sum()), 1),
        "max_step": round(float(steps.max()), 2) if steps.size else 0.0,
    }


def _parse(iso: str) -> float:
    import datetime
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--top", type=int, default=20, help="how many candidates to print")
    ap.add_argument("--min-gap", type=int, default=2, help="min bridged-coast run (frames)")
    ap.add_argument("--min-obs", type=float, default=0.6, help="min observed_fraction")
    ap.add_argument("--min-path", type=float, default=2.0, help="min path length (m)")
    ap.add_argument("--max-step", type=float, default=1.5, help="max observed inter-frame step (m)")
    args = ap.parse_args()

    files = sorted(f for f in args.traj_dir.glob("*.json"))
    rows = [_summarize(json.loads(f.read_text())) for f in files]

    cands = [r for r in rows
             if r["gap_frames"] >= args.min_gap and r["obs_frac"] >= args.min_obs
             and r["path_m"] >= args.min_path and r["max_step"] <= args.max_step]
    cands.sort(key=lambda r: (r["gap_frames"], r["path_m"]), reverse=True)

    print(f"Scanned {len(rows)} pedestrians; {len(cands)} occlusion-bridged demo candidates "
          f"(gap>={args.min_gap}f, obs>={args.min_obs}, path>={args.min_path}m, max_step<={args.max_step}m)\n")
    print(f"{'seq':>7} {'ped':>9} {'n':>4} {'obs%':>5} {'gap':>4} {'gap_s':>6} {'path_m':>7} {'maxstep':>8}")
    print("-" * 60)
    for r in cands[: args.top]:
        print(f"{r['seq']:>7} {r['ped']:>9} {r['n']:>4} {r['obs_frac']*100:>4.0f}% "
              f"{r['gap_frames']:>4} {r['gap_s']:>6} {r['path_m']:>7} {r['max_step']:>8}")

    if cands:
        print("\nRender the top candidate's sequence (full-frame video, all peds boxed + marked):")
        top = cands[0]
        print(f"  python scripts/viz_render_sequence_video.py --seq {top['seq']}")
        print(f"  python scripts/viz_trajectories.py --seq {top['seq']}   # BEV + QC table for the same seq")


if __name__ == "__main__":
    main()
