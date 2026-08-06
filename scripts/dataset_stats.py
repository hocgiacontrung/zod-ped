"""The dataset's numbers, derived from the artifacts themselves — the single source of truth.

Counts drift across docs because the pipeline is re-run and prose is not. This script never quotes a
number: it reads what is on disk right now and prints the canonical view. Anything reported to a
supervisor, written into a doc, or put in a paper should come from here, and any doc that disagrees
with this output is stale.

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
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from _common import DEFAULT_TRAJ_DIR, ROOT

ACTIONS = ROOT / "data" / "processed" / "actions"
VERIFIED = ROOT / "data" / "processed" / "actions_verified"
INDEX = ROOT / "data" / "annotations" / "dataset_index.parquet"


def _tier_counts(paths, key="is_in_gold_standard") -> Counter:
    """GOLD/SILVER split of a directory of docs, read from the tier flag on each."""
    c: Counter = Counter()
    for p in paths:
        c["gold" if json.loads(p.read_text()).get(key, True) else "silver"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", default="2.0", choices=("1.0", "1.5", "2.0"),
                    help="prediction horizon the crossing ratios are reported at")
    args = ap.parse_args()
    h = f"intent_h{args.horizon.replace('.', '_')}s"

    traj = _tier_counts(p for p in DEFAULT_TRAJ_DIR.glob("*.json") if not p.name.startswith("_"))
    acts = _tier_counts(ACTIONS.glob("*.json"))
    verified = list(VERIFIED.glob("*.json"))
    human = sum(bool(json.loads(p.read_text()).get("human_review")) for p in verified)

    print("PIPELINE FUNNEL")
    print(f"  Step 1  trajectory files      {sum(traj.values()):>6}   "
          f"gold {traj['gold']:>5}   silver {traj['silver']:>5}")
    print(f"  Step 2  pedestrians (cut+stitch){sum(acts.values()):>5}   "
          f"gold {acts['gold']:>5}   silver {acts['silver']:>5}")
    print(f"  Step 2e labeled records       {len(verified):>6}   "
          f"of which HUMAN-VERIFIED: {human}")

    if not INDEX.exists():
        print("\n(no dataset_index.parquet — Step 3 has not run)")
        return
    df = pd.read_parquet(INDEX)
    df["tier"] = df.is_in_gold_standard.map({True: "GOLD", False: "SILVER"})
    df["ped_key"] = df.sequence_id + "_" + df.pedestrian_id      # silver ids repeat across sequences
    df["crossing"] = df[h] == "crossing"

    print(f"\nSHIPPED DATASET  (data/annotations, horizon {args.horizon}s)")
    print(f"  {len(df)} samples / {df.ped_key.nunique()} pedestrians / "
          f"{df.sequence_id.nunique()} sequences")
    tbl = df.groupby(["tier", "split"]).agg(samples=("crossing", "size"),
                                            peds=("ped_key", "nunique"),
                                            crossers=("crossing", "sum"))
    tbl["ratio"] = (tbl.crossers / tbl.samples).round(3)
    print(tbl.to_string())

    for tier in ("GOLD", "SILVER"):
        d = df[df.tier == tier]
        print(f"  {tier:6} total: {len(d):>5} samples, {int(d.crossing.sum()):>4} crossers "
              f"(ratio {d.crossing.mean():.3f})")
    print("\n  Report the tiers SEPARATELY. GOLD is the human-verified evaluation set; SILVER is")
    print("  weak-labeled training bulk. A combined crossing ratio describes neither.")


if __name__ == "__main__":
    main()
