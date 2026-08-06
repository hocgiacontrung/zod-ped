"""SILVER free cut — drop obvious junk before pouring SILVER through Steps 2-3.

Uses ONLY signals already computed in the Step-1 QC report (all_tracks_quality.csv), so it costs
no compute. The three drop rules are each backed by the hand-curated worksheet (110 SILVER verdicts):

  - JITTER flag        -> 80% of curated JITTER tracks were dropped (ID-swaps / box bounce)
  - quality_tier=bad   -> curated 'bad' tier was 16/20 drop-or-keepless, 1 merge (ghost/too-short)
  - max_speed > 5 m/s  -> catches people-in-cars (5.9, 6.4 m/s in curation); a walker tops ~4 m/s

NOT dropped here (deliberately): SHORT alone (it is the fragment signature -> stitch, don't drop),
SPARSE_LIFT (0% curated drop), and slow cyclists / reflections (speed can't see them -> left for
human review; build a bicycle/car re-detect pass only if they visibly pollute the survivors).

Writes a keep/drop manifest that Step 2/3 can honor. Does NOT touch trajectory files.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QC_CSV = REPO / "data/processed/review/all_tracks_quality.csv"
OUT = REPO / "data/processed/review/silver_cut.json"


def drop_reasons(row: dict, max_speed_thresh: float) -> list[str]:
    """Return the list of cut rules this SILVER track trips (empty = keep)."""
    reasons = []
    if "JITTER" in (row.get("flags") or ""):
        reasons.append("jitter")
    if (row.get("quality_tier") or "").strip() == "bad":
        reasons.append("bad_tier")
    ms = row.get("max_speed") or ""
    if ms not in ("", "None") and float(ms) > max_speed_thresh:
        reasons.append(f"fast>{max_speed_thresh:g}")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-speed", type=float, default=5.0, help="drop tracks faster than this (m/s)")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.qc_csv)) if (r.get("tier") or "").strip() == "silver"]
    kept, dropped = [], []
    reason_counter: Counter = Counter()
    for r in rows:
        reasons = drop_reasons(r, args.max_speed)
        key = {"seq": r["seq"].strip(), "ped": r["ped"].strip()}
        if reasons:
            dropped.append({**key, "reasons": reasons})
            for rz in reasons:
                reason_counter[rz] += 1
        else:
            kept.append(key)

    manifest = {
        "source": str(args.qc_csv.relative_to(REPO)),
        "rules": {"jitter_flag": True, "bad_quality_tier": True, "max_speed_m_s": args.max_speed},
        "n_silver_total": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reason_counts": dict(reason_counter),
        "kept": kept,
        "dropped": dropped,
    }
    args.out.write_text(json.dumps(manifest, indent=1))

    pct = 100 * len(kept) / max(len(rows), 1)
    print(f"SILVER tracks:   {len(rows)}")
    print(f"  kept:          {len(kept)}  ({pct:.0f}%)")
    print(f"  dropped:       {len(dropped)}  ({100 - pct:.0f}%)")
    print(f"  by rule (a track can trip several): {dict(reason_counter)}")
    print(f"  -> {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
