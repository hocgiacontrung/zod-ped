"""Propose tracklet merges (fragmentation fix) and pre-fill the curation worksheet.

PROPOSE-ONLY: reads trajectory JSONs, proposes which fragments are one pedestrian (see
zodped.labeling.stitching), writes the proposals to JSON, and — for the curated batch — fills the
worksheet's `merge_into` column with the suggested primary id so the human only confirms/rejects.
No trajectory file is modified.

Usage:
    python scripts/stitch_tracks.py                        # run over the curated-batch-20 seqs
    python scripts/stitch_tracks.py --seq 000940           # one sequence, print only
    python scripts/stitch_tracks.py --max-gap-s 2.5 --base-residual-m 1.2   # tune the gates
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from _common import DEFAULT_TRAJ_DIR, ROOT
from zodped.labeling.stitching import StitchParams, stitch_from_dir

DEFAULT_BATCH = ROOT / "data" / "processed" / "review" / "curated_batch_20.json"
DEFAULT_WORKSHEET = ROOT / "data" / "processed" / "review" / "curation_worksheet.csv"
DEFAULT_PROPOSALS = ROOT / "data" / "processed" / "review" / "stitch_proposals.json"


def _prefill_worksheet(worksheet: Path, frag_to_primary: Dict[tuple, str], evidence: Dict[tuple, str]) -> int:
    """Write suggested merge_into (+ evidence note) for fragment rows. Returns rows updated.

    Keys are (seq_id, pedestrian_id): SILVER ids ("silver000", ...) are unique only WITHIN a
    sequence, so a bare-id match would touch every sequence that reused the id.
    """
    if not worksheet.exists():
        print(f"  (worksheet {worksheet} not found — skipping pre-fill)")
        return 0
    rows = list(csv.DictReader(worksheet.open()))
    fields = rows[0].keys() if rows else []
    if "merge_into" not in fields:
        print("  (worksheet has no merge_into column — skipping pre-fill)")
        return 0
    n = 0
    for r in rows:
        key = (r["seq"], r["pedestrian_id"])
        if key in frag_to_primary:
            r["merge_into"] = frag_to_primary[key]
            note = evidence.get(key, "")
            r["notes"] = (r.get("notes") or "")
            r["notes"] = f"auto-merge: {note}" if not r["notes"] else f"{r['notes']}; auto-merge: {note}"
            n += 1
    with worksheet.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="batch manifest (seq list) to run over")
    ap.add_argument("--seq", default=None, help="one sequence id (overrides --batch; prints, no worksheet write)")
    ap.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    ap.add_argument("--proposals-out", type=Path, default=DEFAULT_PROPOSALS)
    ap.add_argument("--no-prefill", action="store_true", help="do not touch the worksheet")
    ap.add_argument("--max-gap-s", type=float, default=StitchParams.max_gap_s)
    ap.add_argument("--max-overlap-s", type=float, default=StitchParams.max_overlap_s)
    ap.add_argument("--base-residual-m", type=float, default=StitchParams.base_residual_m)
    ap.add_argument("--residual-per-s", type=float, default=StitchParams.residual_per_s)
    ap.add_argument("--max-height-diff-m", type=float, default=StitchParams.max_height_diff_m)
    ap.add_argument("--vel-window", type=int, default=StitchParams.vel_window)
    args = ap.parse_args()

    p = StitchParams(max_gap_s=args.max_gap_s, max_overlap_s=args.max_overlap_s,
                     base_residual_m=args.base_residual_m, residual_per_s=args.residual_per_s,
                     max_height_diff_m=args.max_height_diff_m, vel_window=args.vel_window)

    if args.seq:
        seqs = [args.seq]
    else:
        batch = json.loads(args.batch.read_text())
        seqs = [s["seq_id"] for s in batch["sequences"]]

    all_groups: List[dict] = []
    frag_to_primary: Dict[tuple, str] = {}      # (seq, ped_id) -> primary_id
    evidence: Dict[tuple, str] = {}
    for seq in seqs:
        groups = stitch_from_dir(args.traj_dir, seq, p)
        for g in groups:
            g["seq_id"] = seq
            all_groups.append(g)
            for fid in g["fragment_ids"]:
                frag_to_primary[(seq, fid)] = g["primary_id"]
                # note each fragment's STRONGEST adjacent link (smallest residual) as the evidence
                adj = [l for l in g["links"] if fid in (l["from"], l["to"])]
                best = min(adj, key=lambda l: l["residual_m"], default=None)
                evidence[(seq, fid)] = (f"gap={best['gap_s']}s dist={best['residual_m']}m"
                                        if best else "")

    n_frag = sum(len(g["fragment_ids"]) for g in all_groups)
    print(f"proposed {len(all_groups)} merge groups over {len(seqs)} seq(s): "
          f"{n_frag} fragments fold into {len(all_groups)} pedestrians")
    for g in all_groups:
        tier = "GOLD" if g["primary_is_gold"] else "SILVER"
        chain = " -> ".join(f"{l['from'][:12]}~{l['to'][:12]}(gap{l['gap_s']}/d{l['residual_m']})"
                            for l in g["links"])
        print(f"  {g['seq_id']}: keep {g['primary_id'][:14]} [{tier}] <= {chain}")

    args.proposals_out.parent.mkdir(parents=True, exist_ok=True)
    args.proposals_out.write_text(json.dumps(
        {"params": vars(p) if hasattr(p, "__dict__") else p.__dict__,
         "n_groups": len(all_groups), "n_fragments": n_frag, "groups": all_groups}, indent=2))
    print(f"\nwrote proposals -> {args.proposals_out}")

    if not args.seq and not args.no_prefill:
        n = _prefill_worksheet(args.worksheet, frag_to_primary, evidence)
        print(f"pre-filled merge_into for {n} fragment rows -> {args.worksheet}")


if __name__ == "__main__":
    main()
