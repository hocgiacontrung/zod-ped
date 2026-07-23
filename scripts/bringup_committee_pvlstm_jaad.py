"""Committee bring-up gate 1 — PV-LSTM checkpoint sanity on JAAD's own test split.

Runs the released multitask PV-LSTM checkpoint (vita-epfl/bounding-box-prediction v0.1.0;
JAAD-trained, 16-frame obs/pred @30fps) on JAAD test windows built by the authors' own
preprocessing + dataset code, and reports their metrics. If these land in the published
ballpark, our input formatting is proven and any later degradation on ZOD is attributable
to the domain gap — not to plumbing.

Differences from the repo's test.py (whose __main__ is broken for the 2D-intention task:
it routes every dataset to test_3d and evaluates dtype='val'): we evaluate the real 'test'
split, without shuffling, without dropping the last partial batch.

Prerequisites (see CLAUDE.md "external"):
  * third_party/bounding-box-prediction clone + weights/multitask_pv_lstm_trained.pkl
  * JAAD root at data/external/JAAD, preprocessed once:
      python preprocess/jaad_preprocessor.py --data_path=.../data/external/JAAD \
          --train_ratio=0.7 --val_ratio=0.2 --test_ratio=0.1

Usage:
    python scripts/bringup_committee_pvlstm_jaad.py            # full test split
    python scripts/bringup_committee_pvlstm_jaad.py --dtype val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from _common import ROOT

PVLSTM_REPO = ROOT / "third_party" / "bounding-box-prediction"
DEFAULT_CKPT = PVLSTM_REPO / "weights" / "multitask_pv_lstm_trained.pkl"
DEFAULT_DATA = ROOT / "data" / "external" / "JAAD" / "processed_annotations"
DEFAULT_REPORT = ROOT / "data" / "processed" / "reports" / "committee_pvlstm_jaad_eval.json"

# The repo is a flat script collection, not a package: its modules import each other by bare
# name (datasets/jaad.py does `import utils`), so its root must be on sys.path to use it at all.
sys.path.insert(0, str(PVLSTM_REPO))

import network                # noqa: E402  (PV_LSTM)
import utils as pvlstm_utils  # noqa: E402  (speed2pos, ADE/FDE/AIOU/FIOU)
from datasets.jaad import JAAD  # noqa: E402


def predict_split(net: torch.nn.Module, loader, device: str) -> dict:
    """Run the multitask head over one split; return raw scores + trajectory metrics.

    The committee consumes PV-LSTM as a SCORER: the released checkpoint is conservative
    (p(cross) tops out ≈0.42, so the repo's 0.5-argmax `intentions` output never predicts
    crossing at all — degenerate recall 0), while the underlying probability RANKS windows
    well. The window score is p(cross) at the LAST future frame (p_last, their intention
    frame); the decision threshold is calibrated on val, never assumed to be 0.5.
    """
    scores, labels, state_preds, state_targets = [], [], [], []
    ade = fde = aiou = fiou = 0.0
    n_batches = 0

    net.eval()
    with torch.no_grad():
        for obs_s, target_s, obs_p, target_p, target_c, label_c in loader:
            obs_s, obs_p = obs_s.to(device), obs_p.to(device)
            target_p = target_p.to(device)
            speed_preds, crossing_preds, _ = net(speed=obs_s, pos=obs_p, average=True)

            preds_p = pvlstm_utils.speed2pos(speed_preds, obs_p)
            ade += float(pvlstm_utils.ADE(preds_p, target_p))
            fde += float(pvlstm_utils.FDE(preds_p, target_p))
            aiou += float(pvlstm_utils.AIOU(preds_p, target_p))
            fiou += float(pvlstm_utils.FIOU(preds_p, target_p))
            n_batches += 1

            scores.extend(crossing_preds[:, -1, 1].cpu().numpy())      # p_last
            labels.extend(label_c.view(-1).numpy())
            state_preds.extend(np.argmax(crossing_preds.view(-1, 2).cpu().numpy(), axis=1))
            state_targets.extend(target_c[:, :, 1].reshape(-1).numpy())

    return {
        "scores": np.asarray(scores), "labels": np.asarray(labels),
        "state_acc": round(accuracy_score(state_targets, state_preds), 4),
        "ade_px": round(ade / n_batches, 2), "fde_px": round(fde / n_batches, 2),
        "aiou": round(aiou / n_batches, 4), "fiou": round(fiou / n_batches, 4),
    }


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Score threshold maximizing F1 on a calibration split (sweep the observed scores)."""
    grid = np.unique(np.round(scores, 3))
    f1s = [f1_score(labels, scores >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def intent_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    binary = (scores >= threshold).astype(int)
    return {
        "n_windows": int(len(labels)),
        "crossing_ratio": round(float(labels.mean()), 4),
        "threshold": round(threshold, 3),
        "intent_auc": round(roc_auc_score(labels, scores), 4),
        "intent_acc": round(accuracy_score(labels, binary), 4),
        "intent_f1": round(f1_score(labels, binary), 4),
        "intent_precision": round(precision_score(labels, binary), 4),
        "intent_recall": round(recall_score(labels, binary), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="preprocessed_annotations root")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # The released checkpoint's training configuration (release notes v0.1.0): 16 in / 16 out,
    # stride 16 — the dataset windowing MUST match or the eval is meaningless.
    cfg = SimpleNamespace(input=16, output=16, stride=16, skip=1, is_3D=False,
                          task="2D_bounding_box-intention", hidden_size=512,
                          hardtanh_limit=100, device=args.device)

    net = network.PV_LSTM(cfg).to(args.device)
    # The released checkpoint predates the repo's attribute head (nuScenes-only branch, never
    # used by the JAAD intention task), so those keys are legitimately absent. Load non-strictly
    # but verify that NOTHING ELSE is missing/unexpected — a real mismatch must still fail loud.
    result = net.load_state_dict(
        torch.load(args.ckpt, map_location=args.device, weights_only=False), strict=False)
    bad = [k for k in result.missing_keys if not k.startswith(("attrib_decoder", "fc_attrib"))]
    if bad or result.unexpected_keys:
        raise SystemExit(f"checkpoint mismatch: missing {bad}, unexpected {result.unexpected_keys}")

    cache_dir = PVLSTM_REPO / "outputs"    # their window cache (jaad_{dtype}_16_16_16.csv)
    cache_dir.mkdir(exist_ok=True)
    splits = {}
    for dtype in ("val", "test"):          # val calibrates the threshold, test reports
        dataset = JAAD(data_dir=str(args.data_dir), out_dir=str(cache_dir), dtype=dtype,
                       input=cfg.input, output=cfg.output, stride=cfg.stride, task=cfg.task,
                       from_file=(cache_dir / f"jaad_{dtype}_16_16_16.csv").exists())
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, drop_last=False)
        splits[dtype] = predict_split(net, loader, args.device)

    threshold = calibrate_threshold(splits["val"]["scores"], splits["val"]["labels"])
    report = {"checkpoint": str(args.ckpt.relative_to(ROOT)),
              "window": {"input": cfg.input, "output": cfg.output, "stride": cfg.stride},
              "score": "p(cross) at last future frame; threshold = max-F1 on val",
              "splits": {}}
    for dtype, res in splits.items():
        report["splits"][dtype] = {
            **intent_metrics(res["scores"], res["labels"], threshold),
            **{k: res[k] for k in ("state_acc", "ade_px", "fde_px", "aiou", "fiou")},
        }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))

    for dtype, m in report["splits"].items():
        print(f"\nPV-LSTM on JAAD {dtype} ({m['n_windows']} windows, "
              f"crossing ratio {m['crossing_ratio']}, threshold {m['threshold']}):")
        for k, v in m.items():
            if k not in ("n_windows", "crossing_ratio", "threshold"):
                print(f"  {k:16s} {v}")
    print(f"\n  report → {args.report_path}")


if __name__ == "__main__":
    main()
