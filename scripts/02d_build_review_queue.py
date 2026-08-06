"""Step 2b — build the SILVER action-label review queue (geometry + PV-LSTM, merge-aware).

Combines the two independent crossing signals into ONE per-pedestrian verdict so the human reviews
only where they disagree (the 62%/92% agreement result — docs / [[silver-cut-stitch]]):

  * geometry  = Step-2 `crosses_ego_road` (feet-on-road), per track
  * PV-LSTM   = p(cross) at threshold, per track  (this script runs the scorer over ALL silver)

Then it resolves TRACKS into PEDESTRIANS using the finalized stitch mapping (a merged pedestrian
crosses if ANY surviving fragment crosses; pv_score = max over fragments; t_c = earliest), applies the
free cut (drops junk tracks), and EXCLUDES anything already hand-labeled in the worksheet — the
curated batch stays untouched, both file and content.

Output (NEW files, never the worksheet):
  * data/processed/review/silver_review_queue.csv   — disagreements first = what to review
  * data/processed/review/silver_auto_labels.json   — the agreements = auto-accepted labels

A silver fragment that merges into a GOLD primary is OUT of scope here (it belongs to that gold
pedestrian, already labeled/reviewed on the gold side) and is skipped.

Usage:
    python scripts/02d_build_review_queue.py --max-seqs 5     # smoke test
    python scripts/02d_build_review_queue.py                  # full silver (background; ~PV inference)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from _common import DEFAULT_DET_DIR, DEFAULT_TRAJ_DIR, ROOT, tier_matches
from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.committee import ProjectionContext, SweepConfig, build_track_windows_from_detections
from zodped.labeling.detection_cache import read_doc
from zodped.labeling.samples import TrackTimeline

# reuse the validated scorer + labels loaders from the gate (same checkpoint, same protocol)
import importlib.util
_spec = importlib.util.spec_from_file_location("gate02b", ROOT / "scripts" / "02b_committee_gate.py")
gate02b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate02b)

REVIEW_DIR = ROOT / "data" / "processed" / "review"
CUT_PATH = REVIEW_DIR / "silver_cut.json"
STITCH_PATH = REVIEW_DIR / "stitch_proposals_final.json"
# PV-LSTM decision threshold: the max-F1 point from the human-anchor gate (committee_gate_pvlstm_human)
PV_THRESHOLD = 0.170


def load_cut_dropped(path: Path) -> set:
    """(seq, ped) tracks the free cut removed as junk."""
    d = json.loads(path.read_text())
    return {(x["seq"], x["ped"]) for x in d["dropped"]}


def load_merge_map(path: Path) -> tuple[dict, dict]:
    """Return (member -> primary_key, primary_key -> {members, primary_is_gold}).

    Keys are (seq, ped). Only groups are represented; singleton tracks are handled by their absence.
    """
    d = json.loads(path.read_text())
    member_to_primary, groups = {}, {}
    for g in d["groups"]:
        seq = g["seq_id"]
        primary = (seq, g["primary_id"])
        groups[primary] = {"members": [(seq, m) for m in g["member_ids"]],
                           "primary_is_gold": g["primary_is_gold"]}
        for m in g["member_ids"]:
            member_to_primary[(seq, m)] = primary
    return member_to_primary, groups


def load_curated(worksheet: Path) -> set:
    """(seq, ped) already hand-judged — excluded from the new queue so the curated batch stays clean."""
    done = set()
    for r in csv.DictReader(worksheet.open()):
        if (r["crossed_yes_no"] or "").strip() or (r["verdict_keep_drop_fix_merge"] or "").strip():
            done.add((r["seq"].zfill(6), r["pedestrian_id"]))
    return done


def score_all_silver(args, cfg, scorer, dropped: set) -> dict:
    """Score every kept SILVER track. Returns (seq, ped) -> {score, geo_cross, t_c, n_windows}."""
    out, skips = {}, {"no_windows": 0, "frameless": 0, "seq_failures": 0}
    by_seq = defaultdict(list)
    for path in sorted(args.traj_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        by_seq[path.name.split("_", 1)[0]].append(path)

    seqs = sorted(by_seq)[: args.max_seqs] if args.max_seqs else sorted(by_seq)
    for i, seq in enumerate(seqs, 1):
        try:
            ctx = ProjectionContext(ROOT / "data" / "raw" / "sequences" / seq)
            doc = read_doc(DEFAULT_DET_DIR / f"{seq}.json")
            det_frames = sorted((parse_zod_ts(Path(fn).stem.rsplit("_", 1)[-1]), np.asarray(dets))
                                for fn, dets in doc["images"].items() if dets)
            for path in by_seq[seq]:
                ped = path.name.split("_", 1)[1][:-5]
                if (seq, ped) in dropped:
                    continue
                trajectory = json.loads(path.read_text())
                if not tier_matches(trajectory, args.tier) or not trajectory["frames"]:
                    continue
                timeline = TrackTimeline(trajectory)
                windows = build_track_windows_from_detections(timeline, ctx, det_frames, cfg)
                score = float(scorer(windows.boxes).max()) if windows.boxes.shape[0] else None
                action_path = args.actions_dir / path.name
                geo_cross, t_c = None, None
                if action_path.exists():
                    a = json.loads(action_path.read_text())
                    if a["status"] == "determined":
                        geo_cross = bool(a["crosses_ego_road"])
                        t_c = a.get("crossing_frame_timestamp")
                out[(seq, ped)] = {"score": score, "geo_cross": geo_cross, "t_c": t_c,
                                   "n_windows": int(windows.boxes.shape[0])}
        except Exception as exc:
            skips["seq_failures"] += 1
            print(f"  ! {seq}: {exc!r}"[:160])
        if i % 25 == 0:
            print(f"  ...{i}/{len(seqs)} sequences scored")
    return out, skips


def build_pedestrians(scored: dict, member_to_primary: dict, groups: dict, curated: set) -> list:
    """Fold tracks into pedestrians via the merge map; aggregate the two signals per pedestrian."""
    peds = []
    handled = set()
    # merged groups whose primary is SILVER
    for primary, g in groups.items():
        if g["primary_is_gold"]:
            handled.update(g["members"])          # silver fragments belong to a gold ped — out of scope
            continue
        members = [m for m in g["members"] if m in scored]
        handled.update(g["members"])
        if not members or primary in curated:
            continue
        peds.append(_aggregate(primary, members, scored))
    # singletons (not in any group)
    for key, rec in scored.items():
        if key in handled or key in curated:
            continue
        peds.append(_aggregate(key, [key], scored))
    return peds


def _aggregate(primary: tuple, members: list, scored: dict) -> dict:
    seq, ped = primary
    scores = [scored[m]["score"] for m in members if scored[m]["score"] is not None]
    geos = [scored[m]["geo_cross"] for m in members if scored[m]["geo_cross"] is not None]
    tcs = [scored[m]["t_c"] for m in members if scored[m]["geo_cross"] and scored[m]["t_c"]]
    pv = max(scores) if scores else None
    geo = (any(geos) if geos else None)
    pv_cross = (pv >= PV_THRESHOLD) if pv is not None else None
    agree = (pv_cross == geo) if (pv_cross is not None and geo is not None) else None
    return {
        "seq": seq, "primary_ped": ped, "n_fragments": len(members),
        "geo_cross": geo, "pv_score": round(pv, 4) if pv is not None else None,
        "pv_cross": pv_cross, "agree": agree,
        "t_c": min(tcs) if tcs else None,
        # needs_review when the two disagree OR a signal is missing (can't auto-accept blind)
        "needs_review": (agree is not True),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--actions-dir", type=Path, default=ROOT / "data" / "processed" / "actions")
    ap.add_argument("--worksheet", type=Path, default=REVIEW_DIR / "curation_worksheet.csv")
    ap.add_argument("--ckpt", type=Path, default=gate02b.DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--tier", choices=("gold", "silver"), default="silver",
                    help="gold skips cut+stitch (its tracks are verified + one-per-person)")
    ap.add_argument("--queue-out", type=Path, default=None)
    ap.add_argument("--auto-out", type=Path, default=None)
    args = ap.parse_args()
    args.queue_out = args.queue_out or REVIEW_DIR / f"{args.tier}_review_queue.csv"
    args.auto_out = args.auto_out or REVIEW_DIR / f"{args.tier}_auto_labels.json"

    cfg = SweepConfig()
    scorer = gate02b.load_pvlstm_scorer(args.ckpt, cfg, args.device)
    # GOLD needs neither: no junk cut (verified tracks) and no fragment stitching (one track per ped)
    dropped = load_cut_dropped(CUT_PATH) if args.tier == "silver" else set()
    member_to_primary, groups = load_merge_map(STITCH_PATH) if args.tier == "silver" else ({}, {})
    curated = load_curated(args.worksheet)
    print(f"cut-dropped: {len(dropped)} | merge groups: {len(groups)} | already curated (excluded): {len(curated)}")

    scored, skips = score_all_silver(args, cfg, scorer, dropped)
    peds = build_pedestrians(scored, member_to_primary, groups, curated)

    review = [p for p in peds if p["needs_review"]]
    auto = [p for p in peds if not p["needs_review"]]
    # queue: disagreements first, then by pv_score desc so likely-crossers surface early
    review.sort(key=lambda p: (p["agree"] is not None, -(p["pv_score"] or 0)))

    with args.queue_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "primary_ped", "n_fragments", "geo_cross",
                                          "pv_score", "pv_cross", "agree", "t_c", "human_cross", "notes"])
        w.writeheader()
        for p in review:
            w.writerow({**{k: p[k] for k in ("seq", "primary_ped", "n_fragments", "geo_cross",
                                             "pv_score", "pv_cross", "agree", "t_c")},
                        "human_cross": "", "notes": ""})
    args.auto_out.write_text(json.dumps(
        {"pv_threshold": PV_THRESHOLD, "n_auto": len(auto),
         "auto_labels": [{"seq": p["seq"], "primary_ped": p["primary_ped"],
                          "cross": bool(p["geo_cross"]), "t_c": p["t_c"]} for p in auto]}, indent=1))

    n = len(peds)
    print(f"\n{args.tier.upper()} pedestrians resolved: {n}  (skips {skips})")
    print(f"  AUTO-accepted (agree):   {len(auto)}  ({len(auto)/max(n,1):.0%})")
    print(f"  NEEDS REVIEW:            {len(review)}  ({len(review)/max(n,1):.0%})")
    ac = sum(bool(p['geo_cross']) for p in auto)
    print(f"    of auto: {ac} cross / {len(auto)-ac} no-cross")
    print(f"  queue → {args.queue_out}")
    print(f"  auto  → {args.auto_out}")


if __name__ == "__main__":
    main()
