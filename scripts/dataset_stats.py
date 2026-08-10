"""The dataset's numbers, derived from the artifacts themselves — the single source of truth.

Counts drift across docs because the pipeline is re-run and prose is not. This script never quotes a
number: it reads what is on disk right now and prints the canonical view. Anything reported to a
supervisor, written into a doc, or put in a paper should come from here, and any doc that disagrees
with this output is stale.

The counting itself lives in `zodped.dataset.stats`, so the release bundle's manifest and dataset
card are generated from the same computation this prints — a shipped bundle cannot disagree with
the console.

What each layer is authoritative for:
  * data/processed/trajectories/  — tracks Step 1 produced (FILES, not pedestrians)
  * data/processed/actions/       — pedestrians after cut + stitch (one record each)
  * data/processed/actions_verified/ — the shipped label layer; `human_review` marks a human verdict
  * data/annotations/dataset_index.parquet — THE shipped dataset, one row per sample

Usage:
    python scripts/dataset_stats.py                # the canonical table
    python scripts/dataset_stats.py --horizon 1.0  # a different prediction horizon
"""
from __future__ import annotations

import argparse

import pandas as pd

from _common import DEFAULT_TRAJ_DIR, ROOT
from zodped.dataset.stats import (DEFAULT_HORIZON, HORIZONS, funnel_counts, index_summary,
                                  summary_table)

ACTIONS = ROOT / "data" / "processed" / "actions"
VERIFIED = ROOT / "data" / "processed" / "actions_verified"
INDEX = ROOT / "data" / "annotations" / "dataset_index.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", default=DEFAULT_HORIZON, choices=HORIZONS,
                    help="prediction horizon the crossing ratios are reported at")
    args = ap.parse_args()

    funnel = funnel_counts(DEFAULT_TRAJ_DIR, ACTIONS, VERIFIED)
    step1, step2, step2e = (funnel["step1_trajectory_files"], funnel["step2_pedestrians"],
                            funnel["step2e_labeled_records"])

    print("PIPELINE FUNNEL")
    print(f"  Step 1  trajectory files      {step1['total']:>6}   "
          f"gold {step1.get('gold', 0):>5}   silver {step1.get('silver', 0):>5}")
    print(f"  Step 2  pedestrians (cut+stitch){step2['total']:>5}   "
          f"gold {step2.get('gold', 0):>5}   silver {step2.get('silver', 0):>5}")
    print(f"  Step 2e labeled records       {step2e['total']:>6}   "
          f"of which HUMAN-VERIFIED: {step2e['human_verified']}")

    if not INDEX.exists():
        print("\n(no dataset_index.parquet — Step 3 has not run)")
        return
    df = pd.read_parquet(INDEX)
    summary = index_summary(df, args.horizon)
    totals = summary["totals"]

    print(f"\nSHIPPED DATASET  (data/annotations, horizon {args.horizon}s)")
    print(f"  {totals['samples']} samples / {totals['pedestrians']} pedestrians / "
          f"{totals['sequences']} sequences")
    print(summary_table(df, args.horizon).to_string())

    for tier, block in summary["by_tier"].items():
        print(f"  {tier:6} total: {block['samples']:>5} samples, {block['crossers']:>4} "
              f"crossers (ratio {block['crossing_ratio']:.3f})")
    print("\n  Report the tiers SEPARATELY. GOLD is the human-verified evaluation set; SILVER is")
    print("  weak-labeled training bulk. A combined crossing ratio describes neither.")


if __name__ == "__main__":
    main()
