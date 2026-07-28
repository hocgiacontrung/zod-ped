"""Step 2b — committee gate, member #2: PedGraph+ crossing scores vs the HUMAN curated labels.

Sibling of scripts/02b_committee_gate.py (the PV-LSTM gate); same protocol, same ground truth, same
report shape — only the member changes. PedGraph+ is a GCN over a 19-node skeleton, so it consumes
POSE windows (Step 0b cache, data/processed/pose) instead of box windows, matched to each tracked
pedestrian by IoU. Everything else is identical so the two members are graded apples-to-apples:

  * Truth = worksheet `crossed_yes_no` (yes/no); undetermined / blank / dropped / merged excluded.
  * Metrics POOLED; AUC is the headline (no threshold needed at ~20 positives).
  * Members are SCORERS: score = p(cross) class; track score = max over the track's windows.

CAVEAT — the 19-joint node ORDER is a convention (COCO-17 + neck + mid-hip), NOT read from the repo
(training keypoints ship as opaque pickles). It is UNVALIDATED until a JAAD reference run reproduces
the paper's AUC (~0.77). Until then the ZOD AUC printed here is PROVISIONAL: a low number may mean a
wrong node order rather than a weak member. The report carries this flag.

Usage:
    python scripts/02c_committee_gate_pedgraph.py
    python scripts/02c_committee_gate_pedgraph.py --tier gold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

# Reuse the PV-LSTM gate's truth loader, per-track scoring loop and metrics verbatim — only the
# window builder and the member differ, so the grading protocol stays provably identical.
from _common import DEFAULT_TRAJ_DIR, ROOT, tier_matches
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_gate = import_module("02b_committee_gate")
load_human_labels = _gate.load_human_labels
metrics = _gate.metrics
KEEP_VERDICTS = _gate.KEEP_VERDICTS

from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.committee import ProjectionContext, build_pose_windows
from zodped.labeling.pose_cache import cache_path as pose_cache_path
from zodped.labeling.pose_cache import load_poses
from zodped.labeling.samples import TrackTimeline

PEDGRAPH_REPO = ROOT / "third_party" / "Pedestrian_graph_plus"
DEFAULT_CKPT = PEDGRAPH_REPO / "weigths" / "jaad-23-h2d" / "best.pth"
DEFAULT_WORKSHEET = ROOT / "data" / "processed" / "review" / "curation_worksheet.csv"
DEFAULT_POSE_DIR = ROOT / "data" / "processed" / "pose"
DEFAULT_ACTIONS_DIR = ROOT / "data" / "processed" / "actions"
CROSS_CLASS = 1  # PedGraph 3-class head: 0=not-cross, 1=cross, 2=irrelevant


def load_pedgraph_scorer(ckpt: Path, device: str,
                         batch_size: int = 256) -> Callable[[np.ndarray], np.ndarray]:
    """The PedGraph+ member as a pure scorer: (N, T, 19, 3) skeletons → (N,) p(cross).

    Score = softmax probability of the crossing class. The 2D pose-only checkpoint (jaad-23-h2d)
    fixes frames/vel/seg off and h3d off (3 input channels). The third-party repo is a flat script
    collection, so its root goes on sys.path here to keep zodped import-clean.
    """
    sys.path.insert(0, str(PEDGRAPH_REPO))
    from models.ped_graph23 import pedMondel  # noqa: E402

    net = pedMondel(frames=False, vel=False, seg=False, h3d=False, n_clss=3).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    def score(poses: np.ndarray) -> np.ndarray:
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(poses), batch_size):
                # (B, T, V, C) -> (B, C, T, V) as pedMondel expects
                kp = torch.tensor(poses[i:i + batch_size], dtype=torch.float32, device=device)
                kp = kp.permute(0, 3, 1, 2).contiguous()
                probs = net(kp).softmax(1)
                out.append(probs[:, CROSS_CLASS].cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    return score


def score_curated(args, scorer, labels: dict) -> tuple[list, dict]:
    """Sweep PedGraph over every labeled curated track. Returns (rows, skip counts).

    Mirrors the PV-LSTM gate's loop but builds POSE windows from the Step-0b cache instead of box
    windows. Same skip taxonomy and continue-on-error so the two members' coverage is comparable.
    """
    rows, skips = [], {"missing_trajectory": 0, "missing_pose": 0, "frameless": 0,
                       "no_windows": 0, "seq_failures": 0}
    by_seq: dict[str, list] = {}
    for (seq, ped) in labels:
        by_seq.setdefault(seq, []).append(ped)

    for seq in sorted(by_seq):
        try:
            ctx = ProjectionContext(ROOT / "data" / "raw" / "sequences" / seq)
            pose_path = pose_cache_path(args.pose_dir, seq)
            if not pose_path.exists():
                skips["missing_pose"] += len(by_seq[seq])
                continue
            poses_by_img = load_poses(pose_path, min_conf=args.min_conf)
            pose_frames = sorted(
                (parse_zod_ts(Path(fn).stem.rsplit("_", 1)[-1]), plist)
                for fn, plist in poses_by_img.items()
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
                windows = build_pose_windows(timeline, ctx, pose_frames,
                                             obs_len=args.obs_len, stride=args.stride)
                if windows.poses.shape[0] == 0:
                    skips["no_windows"] += 1
                    continue
                scores = scorer(windows.poses)
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
        except Exception as exc:                       # continue-on-error, like the PV-LSTM gate
            skips["seq_failures"] += 1
            print(f"  ! {seq}: {exc!r}"[:160])
    return rows, skips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--pose-dir", type=Path, default=DEFAULT_POSE_DIR)
    ap.add_argument("--actions-dir", type=Path, default=DEFAULT_ACTIONS_DIR)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--tier", choices=("gold", "silver", "all"), default="all")
    # obs_len is in NATIVE ~10 Hz camera frames (skeletons are never interpolated, unlike the
    # PV-LSTM box gate's 30 Hz virtual timeline); 8 frames ~0.8 s keeps coverage without fabricating.
    ap.add_argument("--obs-len", type=int, default=8, help="camera frames per pose window")
    ap.add_argument("--stride", type=int, default=2, help="camera frames between window ends")
    ap.add_argument("--min-conf", type=float, default=0.3, help="person-confidence floor at cache load")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-path", type=Path,
                    default=ROOT / "data" / "processed" / "reports" / "committee_gate_pedgraph_human.json")
    args = ap.parse_args()

    scorer = load_pedgraph_scorer(args.ckpt, args.device)
    labels = load_human_labels(args.worksheet)
    print(f"human-anchor gate: {len(labels)} kept+labeled tracks (tier {args.tier}, PedGraph pose)")

    rows, skips = score_curated(args, scorer, labels)
    m = metrics(rows)
    report = {"member": "pedgraph+", "checkpoint": str(args.ckpt.relative_to(ROOT)),
              "box_source": "pose(coco17->19)", "tier": args.tier,
              "sweep": {"obs_len": args.obs_len, "stride": args.stride, "min_conf": args.min_conf},
              "truth": "human curation_worksheet crossed_yes_no", "skips": skips, "metrics": m,
              "caveat": "PROVISIONAL: 19-joint node order (COCO-17 + neck + mid-hip) is a convention, "
                        "not read from the repo; UNVALIDATED until a JAAD reference run reproduces the "
                        "paper AUC (~0.77). A low AUC here may be a wrong node order, not a weak member.",
              "tracks": rows}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    print(f"\nscored {m['n']} tracks ({m['n_crossers']} crossers / {m['n_non']} non); skips {skips}")
    print(f"  AUC vs HUMAN (PROVISIONAL): {m['auc']}   (PV-LSTM member #1 = 0.74 | JAAD ref ~0.77)")
    print(f"  mean score crosser/non   : {m['mean_score_crosser']} / {m['mean_score_non']}")
    print(f"  !! node order UNVALIDATED — run the JAAD reference gate before trusting this number")
    print(f"  report → {args.report_path}")


if __name__ == "__main__":
    main()
