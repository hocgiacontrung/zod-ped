"""Step 4a — sequence-level train/val/test split assignment.

Deals whole SEQUENCES into train/val/test (sample-level splitting would leak overlapping
windows of one pedestrian/scene across the split boundary — see zodped.dataset.splits),
stratified so the scarce crosser windows spread proportionally, then stamps the result into
the Parquet index and every per-sample JSON.

The {seq_id: split} mapping is FROZEN once written: re-running re-APPLIES the existing
mapping (so Step-3 re-runs with upgraded labels keep comparable test sequences) and refuses
to re-deal unless --force is given. Re-dealing after any result has been reported on the
old split invalidates that result — force only while nothing downstream depends on it.

Usage:
    python scripts/04_assign_splits.py               # deal once, then stamp index + samples
    python scripts/04_assign_splits.py --force       # RE-DEAL (breaks comparability!)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from _common import DEFAULT_SPLITS_PATH, ROOT, load_seq_ids
from zodped.dataset.splits import (
    DEFAULT_RATIOS,
    SPLITS,
    assign_splits,
    balance_score,
    split_summary,
    stats_from_index,
)

DEFAULT_ANNOTATIONS_DIR = ROOT / "data" / "annotations"
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "splits_report.json"


def make_mapping(index: pd.DataFrame, ratios: tuple, seed: int, n_seeds: int) -> dict:
    """Deal the full working set into splits and wrap it in the frozen-mapping document.

    Sequences are lumpy, so one greedy deal lands a few percent off target by shuffle luck;
    we score `n_seeds` consecutive seeds and freeze the best-balanced deal (deterministic:
    the candidate range and winner are recorded in the mapping).
    """
    stats = stats_from_index(index, load_seq_ids())
    deals = ((s, assign_splits(stats, ratios=ratios, seed=s))
             for s in range(seed, seed + n_seeds))
    best_seed, assignments = min(deals, key=lambda d: balance_score(stats, d[1], ratios))
    return {
        "schema": "splits/v0.1",
        "created": dt.date.today().isoformat(),
        "config": {
            "ratios": dict(zip(SPLITS, ratios)),
            "seed": best_seed,
            "candidate_seeds": [seed, seed + n_seeds - 1],
            "unit": "sequence",
            "strata": "crosser-window buckets (2+, 1, 0), greedy weighted deal",
        },
        "n_sequences": len(assignments),
        "assignments": assignments,
    }


def apply_mapping(mapping: dict, index_path: Path, annotations_dir: Path) -> pd.DataFrame:
    """Stamp the mapping into the Parquet index and every per-sample JSON."""
    index = pd.read_parquet(index_path)
    assignments = mapping["assignments"]
    unknown = sorted(set(index["sequence_id"]) - set(assignments))
    if unknown:   # a sample from a sequence the frozen deal never saw — never guess a split
        raise SystemExit(f"{len(unknown)} sequence(s) missing from the frozen mapping "
                         f"(e.g. {unknown[:5]}); re-deal deliberately with --force")
    index["split"] = index["sequence_id"].map(assignments)
    index.to_parquet(index_path, index=False)

    for sample_id, split in zip(index["sample_id"], index["split"]):
        path = annotations_dir / f"{sample_id}.json"
        sample = json.loads(path.read_text())
        sample["metadata"]["split"] = split
        path.write_text(json.dumps(sample))
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR,
                    help="Step 3 output: per-sample JSONs + Parquet index")
    ap.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS_PATH,
                    help="the frozen {seq_id: split} mapping")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="run report")
    ap.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                    metavar=("TRAIN", "VAL", "TEST"), help="target sample fractions")
    ap.add_argument("--seed", type=int, default=42, help="first candidate shuffle seed")
    ap.add_argument("--n-seeds", type=int, default=64,
                    help="candidate seeds scored; the best-balanced deal is frozen")
    ap.add_argument("--force", action="store_true",
                    help="re-deal even if a frozen mapping exists (breaks comparability)")
    args = ap.parse_args()
    index_path = args.annotations_dir / "dataset_index.parquet"

    if args.splits_path.exists() and not args.force:
        mapping = json.loads(args.splits_path.read_text())
        print(f"Frozen mapping found ({mapping['created']}, {mapping['n_sequences']} sequences) "
              f"→ re-applying. Use --force to re-deal.")
    else:
        mapping = make_mapping(pd.read_parquet(index_path), tuple(args.ratios),
                               args.seed, args.n_seeds)
        args.splits_path.parent.mkdir(parents=True, exist_ok=True)
        args.splits_path.write_text(json.dumps(mapping, indent=2))
        print(f"Dealt {mapping['n_sequences']} sequences (best of {args.n_seeds} seeds: "
              f"{mapping['config']['seed']}, ratios {args.ratios}) → {args.splits_path}")

    index = apply_mapping(mapping, index_path, args.annotations_dir)
    summary = split_summary(index)

    report = {"splits_path": str(args.splits_path), "config": mapping["config"],
              "n_samples": len(index), "summary": summary}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print(f"\nStamped {len(index)} samples in {args.annotations_dir}")
    for split in SPLITS:
        s = summary[split]
        ratios = {h.removeprefix("intent_h").removesuffix("s"): v["crossing_ratio"]
                  for h, v in s["per_horizon"].items()}
        print(f"  {split:5s}: {s['n_samples']:4d} samples ({s['sample_fraction']:.0%}) over "
              f"{s['n_sequences']} seqs, {s['n_tte_anchored']} TTE-anchored, "
              f"crossing ratio by horizon {ratios}")
    print(f"  report → {args.report_path}")


if __name__ == "__main__":
    main()
