"""Step 2b — committee GOLD gate: member crossing scores vs the geometric anchor.

Sweeps a committee member (v1: the released PV-LSTM checkpoint) over every GOLD track on a
virtual 30 Hz timeline (zodped.labeling.committee), aggregates its per-window crossing scores
into a track-level ACTION verdict, and measures agreement with the Step-2 geometric
feet-on-road anchor. This is the gate the consensus labeler must pass before it may label
anything at scale (docs/PIPELINE.md "Action label source").

Protocol facts:
  * Members are SCORERS: the decision threshold is calibrated (max F1 vs the anchor) on the
    FROZEN train split only; agreement is reported per split, so val/test stay honest.
  * Tracks with an `undetermined` anchor (EMPTY tracks) are excluded from calibration and
    agreement, but still scored — a model verdict on anchor-blind tracks is the labeler's
    whole value-add; we count how many it can reach.
  * t_c comparison: the member's t_c estimate is the END of its first above-threshold window,
    a LOWER bound (the score claims a crossing within the next `horizon_s`); the report keeps
    the raw delta distribution rather than pretending onset precision.

Usage:
    python scripts/02b_committee_gate.py --max-seqs 5      # smoke test
    python scripts/02b_committee_gate.py                   # full GOLD gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from _common import DEFAULT_DET_DIR, DEFAULT_SPLITS_PATH, DEFAULT_TRAJ_DIR, ROOT, tier_matches
from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.committee import (
    ProjectionContext,
    SweepConfig,
    TrackWindows,
    aggregate_track,
    build_track_windows,
    build_track_windows_from_detections,
)
from zodped.labeling.detection_cache import read_doc
from zodped.labeling.samples import TrackTimeline

PVLSTM_REPO = ROOT / "third_party" / "bounding-box-prediction"
DEFAULT_CKPT = PVLSTM_REPO / "weights" / "multitask_pv_lstm_trained.pkl"
DEFAULT_ACTIONS_DIR = ROOT / "data" / "processed" / "actions"


def load_pvlstm_scorer(ckpt: Path, cfg: SweepConfig, device: str,
                       batch_size: int = 512) -> Callable[[np.ndarray], np.ndarray]:
    """The PV-LSTM member as a pure scorer: (N, obs_len, 4) cxcywh boxes → (N,) p(cross).

    Score = softmax p(cross) at the LAST future frame (the checkpoint's intention frame).
    The third-party repo is a flat script collection — its root must go on sys.path here;
    keeping that surgery in the runner keeps zodped import-clean.
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


def sweep_tracks(args, cfg: SweepConfig, scorer) -> tuple[List[dict], dict]:
    """Score every selected track; returns (per-track rows, skip counts)."""
    splits = json.loads(DEFAULT_SPLITS_PATH.read_text())["assignments"]
    rows: List[dict] = []
    skips = {"no_action_record": 0, "frameless_track": 0, "no_windows": 0, "seq_failures": 0}

    traj_paths = sorted(args.traj_dir.glob("*.json"))
    by_seq: dict[str, List[Path]] = {}
    for p in traj_paths:
        if not p.name.startswith("_"):
            by_seq.setdefault(p.name.split("_", 1)[0], []).append(p)
    seq_ids = sorted(by_seq)[: args.max_seqs] if args.max_seqs else sorted(by_seq)

    for n_done, seq_id in enumerate(seq_ids):
        try:
            ctx = ProjectionContext(ROOT / "data" / "raw" / "sequences" / seq_id)
            det_frames = None
            if args.boxes == "detector":
                doc = read_doc(DEFAULT_DET_DIR / f"{seq_id}.json")
                det_frames = sorted(
                    (parse_zod_ts(Path(fn).stem.rsplit("_", 1)[-1]), np.asarray(rows))
                    for fn, rows in doc["images"].items() if rows
                )
            for path in by_seq[seq_id]:
                trajectory = json.loads(path.read_text())
                if not tier_matches(trajectory, args.tier):
                    continue
                action_path = args.actions_dir / path.name
                if not action_path.exists():
                    skips["no_action_record"] += 1
                    continue
                action = json.loads(action_path.read_text())
                if not trajectory["frames"]:
                    skips["frameless_track"] += 1
                    continue

                timeline = TrackTimeline(trajectory)
                if args.boxes == "detector":
                    windows = build_track_windows_from_detections(timeline, ctx, det_frames, cfg)
                else:
                    windows = build_track_windows(timeline, ctx, cfg)
                if windows.boxes.shape[0] == 0:
                    skips["no_windows"] += 1
                    continue
                scores = scorer(windows.boxes)
                t_c = action.get("crossing_frame_timestamp")
                rows.append({
                    "seq_id": seq_id,
                    "pedestrian_id": trajectory["pedestrian_id"],
                    "split": splits.get(seq_id),
                    "anchor_status": action["status"],
                    "anchor_crosses": bool(action["crosses_ego_road"]),
                    "anchor_t_c": parse_zod_ts(t_c) if t_c else None,
                    "n_windows": int(len(scores)),
                    "n_ticks": windows.n_ticks,
                    "track_score": round(float(scores.max()), 4),
                    "window_t_ends": [round(t, 3) for t in windows.t_ends.tolist()],
                    "window_scores": [round(float(s), 4) for s in scores.tolist()],
                })
        except Exception as exc:                       # continue-on-error, like Steps 1-3
            skips["seq_failures"] += 1
            print(f"  ! {seq_id}: {exc!r}"[:160])
        if (n_done + 1) % 25 == 0:
            print(f"  ...{n_done + 1}/{len(seq_ids)} sequences, {len(rows)} tracks scored")
    return rows, skips


def calibrate(rows: List[dict]) -> float:
    """Max-F1 track-score threshold on the frozen TRAIN split, determined anchors only."""
    train = [r for r in rows if r["split"] == "train" and r["anchor_status"] == "determined"]
    y = np.array([r["anchor_crosses"] for r in train])
    s = np.array([r["track_score"] for r in train])
    if y.sum() == 0:   # a positive-free calibration set makes every threshold F1=0 (smoke runs)
        raise SystemExit(f"cannot calibrate: 0 anchor crossers among {len(train)} train tracks "
                         "— run over more sequences")
    grid = np.unique(np.round(s, 3))
    f1s = [f1_score(y, s >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def split_agreement(rows: List[dict], threshold: float, horizon_s: float) -> dict:
    """Verdict-vs-anchor agreement + t_c deltas, per frozen split."""
    out: dict = {}
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split and r["anchor_status"] == "determined"]
        if not part:
            continue
        y = np.array([r["anchor_crosses"] for r in part])
        s = np.array([r["track_score"] for r in part])
        pred = s >= threshold

        t_c_deltas = []
        for r in part:
            if r["anchor_crosses"] and r["anchor_t_c"]:
                v = aggregate_track(np.array(r["window_t_ends"]),
                                    np.array(r["window_scores"]), threshold)
                if v.t_c_estimate is not None:
                    t_c_deltas.append(r["anchor_t_c"] - v.t_c_estimate)
        deltas = np.array(t_c_deltas)
        out[split] = {
            "n_tracks": len(part),
            "anchor_crossing_ratio": round(float(y.mean()), 4),
            "agreement": round(float((pred == y).mean()), 4),
            "auc": round(roc_auc_score(y, s), 4) if 0 < y.sum() < len(y) else None,
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "t_c_delta_s": {   # anchor t_c minus member estimate; +[0, horizon] = ideal band
                "n": len(deltas),
                "median": round(float(np.median(deltas)), 2) if len(deltas) else None,
                "p10": round(float(np.percentile(deltas, 10)), 2) if len(deltas) else None,
                "p90": round(float(np.percentile(deltas, 90)), 2) if len(deltas) else None,
                "within_horizon_frac": round(float(((deltas >= 0) & (deltas <= horizon_s)).mean()), 3)
                if len(deltas) else None,
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--actions-dir", type=Path, default=DEFAULT_ACTIONS_DIR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--report-path", type=Path, default=None,
                    help="default: reports/committee_gate_pvlstm_{boxes}.json")
    ap.add_argument("--boxes", choices=("projected3d", "detector"), default="projected3d",
                    help="window box source: projected 3D track box, or matched Step-0 "
                         "detector boxes (isolates the box-style share of the domain gap)")
    ap.add_argument("--tier", choices=("gold", "silver", "all"), default="gold")
    ap.add_argument("--stride", type=int, default=5, help="virtual frames between window ends")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.report_path is None:
        args.report_path = ROOT / "data" / "processed" / "reports" / \
            f"committee_gate_pvlstm_{args.boxes}.json"

    cfg = SweepConfig(stride=args.stride)
    scorer = load_pvlstm_scorer(args.ckpt, cfg, args.device)
    print(f"Committee gate (PV-LSTM, tier {args.tier}, boxes: {args.boxes}): sweeping at "
          f"{cfg.rate_hz:.0f} Hz, obs {cfg.obs_len}, stride {cfg.stride} "
          f"(~{cfg.stride / cfg.rate_hz:.2f}s)")

    rows, skips = sweep_tracks(args, cfg, scorer)
    threshold = calibrate(rows)
    agreement = split_agreement(rows, threshold, cfg.horizon_s)

    n_undet = sum(r["anchor_status"] != "determined" for r in rows)
    undet_verdicts = sum(r["anchor_status"] != "determined" and r["track_score"] >= threshold
                         for r in rows)
    report = {
        "member": "pvlstm", "checkpoint": str(args.ckpt.relative_to(ROOT)),
        "box_source": args.boxes,
        "tier": args.tier, "sweep": cfg.as_dict(), "threshold": round(threshold, 3),
        "n_tracks_scored": len(rows), "skips": skips,
        "agreement_by_split": agreement,
        "anchor_undetermined": {"n_scored": n_undet, "n_model_crossing_verdicts": undet_verdicts},
        "tracks": rows,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report))

    print(f"\n{len(rows)} tracks scored (skips {skips}); threshold {threshold:.3f} (train max-F1)")
    for split, m in agreement.items():
        print(f"  {split:5s}: n={m['n_tracks']:4d} agree={m['agreement']} auc={m['auc']} "
              f"f1={m['f1']} p={m['precision']} r={m['recall']} "
              f"t_c within-horizon {m['t_c_delta_s']['within_horizon_frac']}")
    print(f"  anchor-undetermined tracks scored: {n_undet} "
          f"({undet_verdicts} get a model crossing verdict)")
    print(f"  report → {args.report_path}")


if __name__ == "__main__":
    main()
