"""Step 2e — merge the human crossing verdicts into a VERIFIED GOLD action-label set.

Step 2 labels every track by geometry alone. Measured against a human, that geometry is ~28% wrong on
the crossings it declares, and its crossing TIME is often seconds off — and `crossing_frame_timestamp`
is what Step 3 anchors every prediction window to, so a wrong t_c mislabels the window CONTENTS while
the window COUNT still looks healthy. This script writes the human answer over both fields.
Precedence: human > geometry. PV-LSTM only ranked which tracks a human should watch; where nobody
looked and it dissents from geometry, geometry wins and the record is flagged `pv_disputed`.

INPUTS (two disjoint human passes; overlap is a hard error)
  * review/gold_crosser_worksheet.csv — the 2026-08 crosser review, reviewed independently by two
    people and settled into this one file (86% raw agreement — docs/EXPERIMENTS_LOG.md).
  * review/curation_worksheet.csv     — the earlier hand-curated batch, `tier == gold` rows only.
    Its silver rows are out of scope (SILVER keeps geometry labels).

Offsets in both are SECONDS RELATIVE TO THE KEYFRAME (`info.json: keyframe_time`). `pre_obs` means the
crossing had already begun before the track was observed, so no t_c exists inside the clip.

OUTPUT — a NEW directory, never a mutation of Step 2 or of the worksheets, so Step 2 stays
reproducible from geometry alone and the human layer stays independently auditable:
  * data/processed/actions_verified/{seq}_{ped}.json — the FULL gold population, overridden where a
    human looked and copied through (flagged) where nobody did.
  * data/processed/reports/human_merge_report.json — includes `crossers_without_a_crossing_time`,
    the rows still needing an offset or a `pre_obs` before Step 3 can anchor them.

Step 3 consumes it with no code change:
    python scripts/03_assemble_samples.py --actions-dir data/processed/actions_verified

Usage:
    python scripts/02e_merge_human_labels.py --dry-run   # report only, writes nothing
    python scripts/02e_merge_human_labels.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from _common import DEFAULT_TRAJ_DIR, ROOT, SEQ_DIR, add_tier_arg, tier_matches
from zodped.dataset.keyframe import parse_zod_ts

REVIEW_DIR = ROOT / "data" / "processed" / "review"
CROSSER_WORKSHEET = REVIEW_DIR / "gold_crosser_worksheet.csv"
CURATION_WORKSHEET = REVIEW_DIR / "curation_worksheet.csv"
REVIEW_QUEUE = REVIEW_DIR / "gold_review_queue.csv"
DEFAULT_ACTIONS_DIR = ROOT / "data" / "processed" / "actions"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "actions_verified"
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "human_merge_report.json"

# Worksheet sentinel: the pedestrian was already crossing when the track began, so the crossing
# INSTANT is not observable. A true crosser with no anchorable t_c.
PRE_OBSERVATION = "pre_obs"

# Free text reviewers actually wrote, mapped to the three answers the schema has. Anything outside
# this table is refused rather than silently read as "uncertain" — a typo'd "yes" must not become a
# shrug. Extend the table when a new phrasing appears; do not loosen the matching.
ANSWERS: Dict[str, Optional[bool]] = {
    "yes": True,
    "no": False,
    "uncertain": None, "hard to tell": None, "unsure": None, "undetermined": None,
}

Key = Tuple[str, str]  # (seq_id zero-padded, pedestrian_id)


class Verdict:
    """One human judgement of one GOLD pedestrian, normalised across the two worksheet formats."""

    __slots__ = ("crosses", "offset_s", "pre_obs", "notes", "source")

    def __init__(self, crosses: Optional[bool], offset_s: Optional[float],
                 pre_obs: bool, notes: str, source: str):
        self.crosses = crosses          # True / False / None (= human could not tell)
        self.offset_s = offset_s        # seconds from keyframe, or None
        self.pre_obs = pre_obs          # crossing began before observation
        self.notes = notes
        self.source = source

    @property
    def needs_crossing_time(self) -> bool:
        """A crosser whose crossing instant is unusable — Step 3 has nothing to anchor a window to."""
        return self.crosses is True and self.offset_s is None and not self.pre_obs


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _parse_answer(raw: str, where: str) -> Tuple[bool, Optional[bool]]:
    """Verdict cell -> (was_answered, crosses). Unknown text is a hard error, not a shrug."""
    text = _clean(raw).lower()
    if not text:
        return False, None
    if text not in ANSWERS:
        raise SystemExit(f"{where}: unrecognised verdict {text!r}; "
                         f"use one of {sorted(ANSWERS)} or extend ANSWERS in this script")
    return True, ANSWERS[text]


def _parse_offset(raw: str) -> Tuple[Optional[float], bool]:
    """Worksheet offset cell -> (seconds_from_keyframe, is_pre_observation)."""
    text = _clean(raw)
    if not text:
        return None, False
    if text.lower() == PRE_OBSERVATION:
        return None, True
    try:
        return float(text.replace("+", "")), False
    except ValueError:                      # a free-text scribble is not a time; treat as absent
        return None, False


def read_crosser_worksheet(path: Path) -> Dict[Key, Verdict]:
    """The 2026-08 crosser review. Verdict column is yes / no / uncertain.

    Read with utf-8-sig: these worksheets come back from a spreadsheet editor, which may prepend a
    BOM that would otherwise corrupt the first column name.
    """
    verdicts: Dict[Key, Verdict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        key = (_clean(row["seq"]).zfill(6), _clean(row["pedestrian_id"]))
        answered, crosses = _parse_answer(row["human_cross(yes/no/uncertain)"],
                                          f"{path.name} {key[0]}_{key[1][:8]}")
        if not answered:
            continue
        offset_s, pre_obs = _parse_offset(row["human_crossing_offset_s(e.g.+4.5)"])
        verdicts[key] = Verdict(crosses, offset_s, pre_obs, _clean(row["notes"]), path.stem)
    return verdicts


def read_curation_worksheet(path: Path) -> Dict[Key, Verdict]:
    """The earlier hand-curated batch; GOLD rows only (silver rows keep their committee labels)."""
    verdicts: Dict[Key, Verdict] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if _clean(row.get("tier")).lower() != "gold":
            continue
        key = (_clean(row["seq"]).zfill(6), _clean(row["pedestrian_id"]))
        answered, crosses = _parse_answer(row["crossed_yes_no"], f"{path.name} {key[0]}")
        if not answered:
            continue
        offset_s, pre_obs = _parse_offset(row["crossing_start_frame"])
        verdicts[key] = Verdict(crosses, offset_s, pre_obs, _clean(row.get("notes") or ""),
                                "curation_worksheet")
    return verdicts


def load_disputed(path: Path) -> set:
    """(seq, ped) where geometry says no-cross but PV-LSTM says cross, and no human ever looked.

    These keep geometry's label — in a rare-event dataset, a false 'cross' poisons the scarce
    positive class while a false 'no' dissolves into the large negative one — but they carry a flag
    so the residual contamination stays a counted quantity instead of an invisible one.
    """
    if not path.exists():
        return set()
    disputed = set()
    for row in csv.DictReader(path.open()):
        if _clean(row["geo_cross"]) == "False" and _clean(row["pv_cross"]) == "True":
            disputed.add((_clean(row["seq"]).zfill(6), _clean(row["primary_ped"])))
    return disputed


def keyframe_times(seq_ids: Iterable[str]) -> Dict[str, float]:
    """seq_id -> keyframe epoch seconds, the reference both worksheets' offsets are measured from."""
    out = {}
    for seq_id in seq_ids:
        info = json.loads((SEQ_DIR / seq_id / "info.json").read_text())
        out[seq_id] = parse_zod_ts(info["keyframe_time"])
    return out


def _iso(epoch_s: float) -> str:
    """Epoch seconds -> the ISO-8601 UTC form Step 2 records (and parse_zod_ts) already use."""
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()


def apply_verdict(record: dict, verdict: Verdict, keyframe_ts: float,
                  track_span: Optional[Tuple[float, float]]) -> dict:
    """Return `record` with the human answer written over the geometric one.

    Three shapes of outcome:
      * decided + timed   -> determined, crosses set, t_c = keyframe + offset
      * crosser, no time  -> UNDETERMINED. The action is known but no window can be anchored to it,
                             and Step 3 dereferences crossing_frame_timestamp unguarded whenever
                             crosses_ego_road is true (samples.py:248). The truth is preserved in
                             human_review so the record never claims the human was unsure.
      * human unsure      -> undetermined, matching the pipeline rule for empty tracks:
                             kept and flagged, never forced.
    """
    geometry = {"crosses_ego_road": record.get("crosses_ego_road"),
                "crossing_frame_timestamp": record.get("crossing_frame_timestamp"),
                "status": record.get("status")}

    record["method"] = "human_verified"
    record["label_confidence_tier"] = "verified"
    record["human_review"] = {
        "source": verdict.source,
        "crosses_ego_road": verdict.crosses,
        "offset_from_keyframe_s": verdict.offset_s,
        "pre_observation_crossing": verdict.pre_obs,
        "notes": verdict.notes,
        "overrode_geometry": geometry,
    }

    if verdict.crosses is None:                                  # human could not tell
        record.update(status="undetermined", crosses_ego_road=None,
                      crossing_frame_timestamp=None,
                      undetermined_reason="human_uncertain")
        return record

    record["crosses_ego_road"] = verdict.crosses
    if not verdict.crosses:
        record.update(status="determined", crossing_frame_timestamp=None)
        return record

    if verdict.offset_s is None:                                 # true crosser, unanchorable
        record.update(status="undetermined", crossing_frame_timestamp=None,
                      undetermined_reason="crossing_started_before_observation"
                      if verdict.pre_obs else "human_gave_no_crossing_time")
        return record

    t_c = keyframe_ts + verdict.offset_s
    record.update(status="determined", crossing_frame_timestamp=_iso(t_c))
    # A human can legitimately place the crossing outside the tracked span (that is exactly the
    # "not tracked when start crossing" note). Flagged, not clamped: snapping would fabricate a
    # crossing time, and Step 3 drops such tracks on its own when no window can precede t_c.
    if track_span is not None:
        first, last = track_span
        if not (first <= t_c <= last):
            record["t_c_outside_tracked_span"] = True
    return record


def build(args: argparse.Namespace) -> dict:
    verdicts = read_crosser_worksheet(CROSSER_WORKSHEET)
    curated = read_curation_worksheet(CURATION_WORKSHEET)
    overlap = sorted(set(verdicts) & set(curated))
    if overlap:                                  # the two batches are meant to be disjoint populations
        raise SystemExit(f"the crosser review and the curated batch both cover {len(overlap)} "
                         f"pedestrians, e.g. {overlap[:3]}; resolve by hand before merging")
    verdicts.update(curated)
    # Crossers the worksheets leave untimed: Step 3 cannot anchor a window to them, so they are named
    # in the report rather than merely counted — each needs an offset or the `pre_obs` sentinel.
    untimed = sorted(f"{seq}_{ped}" for (seq, ped), v in verdicts.items() if v.needs_crossing_time)
    disputed = load_disputed(REVIEW_QUEUE)

    traj_paths = [p for p in sorted(args.traj_dir.glob("*.json")) if not p.name.startswith("_")]
    seq_ids = sorted({p.name.split("_", 1)[0] for p in traj_paths})
    keyframes = keyframe_times(seq_ids)

    counts: Counter = Counter()
    unmatched = set(verdicts)
    outside_span: List[str] = []
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in traj_paths:
        seq_id, ped_id = path.name.split("_", 1)[0], path.name.split("_", 1)[1][:-5]
        trajectory = json.loads(path.read_text())
        if not tier_matches(trajectory, args.tier):
            continue
        action_path = args.actions_dir / path.name
        if not action_path.exists():
            counts["no_action_record"] += 1
            continue

        record = json.loads(action_path.read_text())
        key = (seq_id, ped_id)
        verdict = verdicts.get(key)

        if verdict is not None:
            unmatched.discard(key)
            frames = trajectory.get("frames") or []
            span = ((parse_zod_ts(frames[0]["timestamp"]), parse_zod_ts(frames[-1]["timestamp"]))
                    if frames else None)
            before = bool(record.get("crosses_ego_road"))
            record = apply_verdict(record, verdict, keyframes[seq_id], span)
            counts["human_verified"] += 1
            if verdict.crosses is None:
                counts["human_uncertain"] += 1
            elif verdict.crosses:
                counts["human_cross"] += 1
                counts["flipped_no_to_cross"] += int(not before)
                if record["status"] == "undetermined":
                    counts["cross_without_anchorable_t_c"] += 1
                if record.get("t_c_outside_tracked_span"):
                    outside_span.append(f"{seq_id}_{ped_id}")
            else:
                counts["human_no_cross"] += 1
                counts["flipped_cross_to_no"] += int(before)
        else:
            record["method"] = record.get("method", "rule_based_geometry")
            if key in disputed:
                record["pv_disputed"] = True     # geometry kept; PV dissent recorded, not applied
                counts["pv_disputed_unreviewed"] += 1
            counts["geometry_kept"] += 1

        counts["written"] += 1
        if not args.dry_run:
            (args.out_dir / path.name).write_text(json.dumps(record))

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "tier": args.tier,
        "inputs": {"crosser_worksheet": str(CROSSER_WORKSHEET.relative_to(ROOT)),
                   "curation_worksheet": str(CURATION_WORKSHEET.relative_to(ROOT)),
                   "review_queue": str(REVIEW_QUEUE.relative_to(ROOT))},
        "out_dir": str(args.out_dir.relative_to(ROOT)),
        "precedence": "human > geometry; PV-LSTM ranks review only (dissent flagged, never applied)",
        "counts": dict(sorted(counts.items())),
        "crossers_without_a_crossing_time": untimed,
        "t_c_outside_tracked_span": outside_span,
        "verdicts_without_a_track": sorted(f"{s}_{p}" for s, p in unmatched),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="Step 1 trajectory JSONs")
    ap.add_argument("--actions-dir", type=Path, default=DEFAULT_ACTIONS_DIR, help="Step 2 action records")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="verified action records")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="run report")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    add_tier_arg(ap)
    args = ap.parse_args()

    report = build(args)
    if not args.dry_run:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
