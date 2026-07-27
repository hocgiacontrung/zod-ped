"""Step 2b — committee gate: PV-LSTM crossing scores vs the HUMAN curated labels.

The committee's ACTION labeler must be validated against ground truth before it can label at scale
(docs/PIPELINE.md "Action label source"). Ground truth is the HUMAN `crossed_yes_no` in the
curation worksheet — NOT the geometric feet-on-road label, which the manual anchor batch showed is
itself ~30% wrong vs a human (docs/EXPERIMENTS_LOG.md "Manual anchor batch"). An earlier version of
this gate graded against geometry and mistook geometry's error for model error; that is retired.

Protocol (forced by the small hand-labeled set):
  * Truth = worksheet `crossed_yes_no` (yes/no); undetermined / blank / dropped / merged excluded.
  * Both tiers scored (silver has no geometric action record but does have a box track + human label).
  * Metrics POOLED; AUC is the headline. AUC needs no threshold, so it is the one honest number at
    ~20 positives; the max-F1 threshold + its F1/precision/recall are reported but flagged OPTIMISTIC
    (calibrated and evaluated on the same pool — no held-out split this small).
  * Members are SCORERS: score = p(cross) at the checkpoint's last future frame; track score = max
    over the track's windows.

Caveat carried in the report: merges are propose-only, so a crosser whose crossing lived in a silver
fragment (worksheet note "good after merge") is scored on its INCOMPLETE gold track here.

Usage:
    python scripts/02b_committee_gate.py                    # detector boxes (default)
    python scripts/02b_committee_gate.py --boxes projected3d
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from _common import DEFAULT_DET_DIR, DEFAULT_TRAJ_DIR, ROOT, tier_matches
from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.committee import (
    ProjectionContext,
    SweepConfig,
    build_track_windows,
    build_track_windows_from_detections,
)
from zodped.labeling.detection_cache import read_doc
from zodped.labeling.samples import TrackTimeline

PVLSTM_REPO = ROOT / "third_party" / "bounding-box-prediction"
DEFAULT_CKPT = PVLSTM_REPO / "weights" / "multitask_pv_lstm_trained.pkl"
DEFAULT_WORKSHEET = ROOT / "data" / "processed" / "review" / "curation_worksheet.csv"
DEFAULT_ACTIONS_DIR = ROOT / "data" / "processed" / "actions"
KEEP_VERDICTS = {"", "keep", "fix"}


def load_pvlstm_scorer(ckpt: Path, cfg: SweepConfig, device: str,
                       batch_size: int = 512) -> Callable[[np.ndarray], np.ndarray]:
    """The PV-LSTM member as a pure scorer: (N, obs_len, 4) cxcywh boxes → (N,) p(cross).

    Score = softmax p(cross) at the LAST future frame (the checkpoint's intention frame). The
    third-party repo is a flat script collection — its root goes on sys.path here so zodped stays
    import-clean (torch + the gitignored checkpoint never leak into the library).
    """
    sys.path.insert(0, str(PVLSTM_REPO))
    import network  # noqa: E402

    net_cfg = SimpleNamespace(input=cfg.obs_len, output=cfg.obs_len, skip=1, is_3D=False,
                              task="2D_bounding_box-intention", hidden_size=512,
                              hardtanh_limit=100, device=device)
    net = network.PV_LSTM(net_cfg).to(device)
    result = net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False),
                                 strict=False)
    bad = [k for k in result.missing_keys if not k.startswith(("attrib_decoder", "fc_attrib"))]
    if bad or result.unexpected_keys:  # only the unused nuScenes attribute head may be absent
        raise SystemExit(f"checkpoint mismatch: missing {bad}, unexpected {result.unexpected_keys}")
    net.eval()

    def score(boxes: np.ndarray) -> np.ndarray:
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(boxes), batch_size):
                pos = torch.tensor(boxes[i:i + batch_size], dtype=torch.float32, device=device)
                speed = pos[:, 1:] - pos[:, :-1]
                _, crossing_preds, _ = net(speed=speed, pos=pos, average=True)
                out.append(crossing_preds[:, -1, 1].cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    return score


def load_human_labels(worksheet: Path) -> dict:
    """(seq_id, pedestrian_id) -> {y, tier, note} for kept tracks with a yes/no crossing label."""
    labels = {}
    for r in csv.DictReader(worksheet.open()):
        verdict = (r["verdict_keep_drop_fix_merge"] or "").strip()
        crossed = (r["crossed_yes_no"] or "").strip().lower()
        if verdict not in KEEP_VERDICTS or crossed not in ("yes", "no"):
            continue
        labels[(r["seq"].zfill(6), r["pedestrian_id"])] = {
            "y": int(crossed == "yes"), "tier": r["tier"], "note": (r["notes"] or "").strip(),
        }
    return labels


def score_curated(args, cfg: SweepConfig, scorer, labels: dict) -> tuple[list, dict]:
    """Sweep PV-LSTM over every labeled curated track. Returns (rows, skip counts)."""
    rows, skips = [], {"missing_trajectory": 0, "frameless": 0, "no_windows": 0, "seq_failures": 0}
    by_seq: dict[str, list] = {}
    for (seq, ped) in labels:
        by_seq.setdefault(seq, []).append(ped)

    for seq in sorted(by_seq):
        try:
            ctx = ProjectionContext(ROOT / "data" / "raw" / "sequences" / seq)
            det_frames = None
            if args.boxes == "detector":
                doc = read_doc(DEFAULT_DET_DIR / f"{seq}.json")
                det_frames = sorted(
                    (parse_zod_ts(Path(fn).stem.rsplit("_", 1)[-1]), np.asarray(dets))
                    for fn, dets in doc["images"].items() if dets
                )
            for ped in by_seq[seq]:
                path = args.traj_dir / f"{seq}_{ped}.json"
                if not path.exists():
                    skips["missing_trajectory"] += 1
                    continue
                trajectory = json.loads(path.read_text())
                if not tier_matches(trajectory, args.tier) or not trajectory["frames"]:
                    skips["frameless"] += 1
                    continue
                timeline = TrackTimeline(trajectory)
                windows = (build_track_windows_from_detections(timeline, ctx, det_frames, cfg)
                           if args.boxes == "detector" else build_track_windows(timeline, ctx, cfg))
                if windows.boxes.shape[0] == 0:
                    skips["no_windows"] += 1
                    continue
                scores = scorer(windows.boxes)
                meta = labels[(seq, ped)]
                action_path = args.actions_dir / path.name
                anchor = json.loads(action_path.read_text()) if action_path.exists() else None
                rows.append({
                    "seq_id": seq, "pedestrian_id": ped, "tier": meta["tier"],
                    "y_human": meta["y"], "note": meta["note"],
                    "anchor_crosses": (bool(anchor["crosses_ego_road"])
                                       if anchor and anchor["status"] == "determined" else None),
                    "n_windows": int(len(scores)),
                    "track_score": round(float(scores.max()), 4),
                })
        except Exception as exc:                       # continue-on-error, like Steps 1-3
            skips["seq_failures"] += 1
            print(f"  ! {seq}: {exc!r}"[:160])
    return rows, skips


def metrics(rows: list) -> dict:
    """Pooled AUC (headline) + optimistic-threshold F1/p/r + geometry-vs-human on shared golds."""
    y = np.array([r["y_human"] for r in rows])
    s = np.array([r["track_score"] for r in rows])
    grid = np.unique(np.round(s, 3))
    thr = float(grid[int(np.argmax([f1_score(y, s >= t, zero_division=0) for t in grid]))])
    pred = s >= thr
    crosser_s, non_s = s[y == 1], s[y == 0]
    tri = [r for r in rows if r["anchor_crosses"] is not None]   # gold determined tracks
    return {
        "n": len(rows), "n_crossers": int(y.sum()), "n_non": int((y == 0).sum()),
        "auc": round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None,
        "mean_score_crosser": round(float(crosser_s.mean()), 4) if len(crosser_s) else None,
        "mean_score_non": round(float(non_s.mean()), 4) if len(non_s) else None,
        "optimistic_threshold": round(thr, 3),
        "optimistic_f1": round(f1_score(y, pred, zero_division=0), 4),
        "optimistic_precision": round(precision_score(y, pred, zero_division=0), 4),
        "optimistic_recall": round(recall_score(y, pred, zero_division=0), 4),
        "geometric_vs_human_on_shared": {
            "n": len(tri),
            "agreement": round(float(np.mean([r["anchor_crosses"] == bool(r["y_human"])
                                              for r in tri])), 4) if tri else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--actions-dir", type=Path, default=DEFAULT_ACTIONS_DIR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--boxes", choices=("detector", "projected3d"), default="detector",
                    help="committee input boxes: Step-0 detector boxes (default, best AUC) or the "
                         "shipped projected-3D box (wider coverage, incl. coasted frames)")
    ap.add_argument("--tier", choices=("gold", "silver", "all"), default="all")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-path", type=Path,
                    default=ROOT / "data" / "processed" / "reports" / "committee_gate_pvlstm_human.json")
    args = ap.parse_args()

    cfg = SweepConfig(stride=args.stride)
    scorer = load_pvlstm_scorer(args.ckpt, cfg, args.device)
    labels = load_human_labels(args.worksheet)
    print(f"human-anchor gate: {len(labels)} kept+labeled tracks (tier {args.tier}, boxes {args.boxes})")

    rows, skips = score_curated(args, cfg, scorer, labels)
    m = metrics(rows)
    report = {"member": "pvlstm", "checkpoint": str(args.ckpt.relative_to(ROOT)),
              "box_source": args.boxes, "tier": args.tier, "sweep": cfg.as_dict(),
              "truth": "human curation_worksheet crossed_yes_no", "skips": skips, "metrics": m,
              "caveat": "merges are propose-only; crossers whose crossing lived in a merged silver "
                        "fragment are scored on their incomplete gold track",
              "tracks": rows}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print(f"\nscored {m['n']} tracks ({m['n_crossers']} crossers / {m['n_non']} non); skips {skips}")
    print(f"  AUC vs HUMAN            : {m['auc']}   (JAAD ref 0.77 | old ZOD-geometric gate 0.55)")
    print(f"  mean score crosser/non  : {m['mean_score_crosser']} / {m['mean_score_non']}")
    print(f"  optimistic thr {m['optimistic_threshold']}: F1 {m['optimistic_f1']} "
          f"p {m['optimistic_precision']} r {m['optimistic_recall']}")
    g = m["geometric_vs_human_on_shared"]
    print(f"  (geometry vs human on shared {g['n']} gold tracks: agree {g['agreement']})")
    print(f"  report → {args.report_path}")


if __name__ == "__main__":
    main()
