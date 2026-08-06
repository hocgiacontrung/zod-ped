"""Step 4b — reference baseline: a PV-LSTM-style crossing predictor trained ON OUR DATA.

This is dataset QA, not a model deliverable. It answers two questions:

  1. Are the labels learnable at all? A model that cannot beat chance means the labels are noise or
     the features do not carry the signal — a dataset bug, found before anyone builds on it.
  2. Does SILVER earn its place? Run twice (`--train-tiers gold` vs `all`) against the SAME
     evaluation set. If the weak-labeled tier helps, that is the evidence justifying it; if it
     hurts, that is worth knowing before it ships as training bulk.

Zero-shot PV-LSTM scored AUC 0.52 on our verified tracks, but that judged a JAAD-trained checkpoint
on ZOD windows — a domain-gap result, not a verdict on the architecture. This trains the same
position+velocity LSTM at OUR protocol (5-frame windows at ~10 Hz rather than 16 at 30 Hz), which is
the matched-protocol retrain the gate work kept naming as the honest next step.

PROTOCOL — the parts that keep the number meaningful:
  * Train on the FROZEN train split only. Model selection on GOLD val. Test touched once.
  * EVALUATE ON GOLD TEST ONLY, always, whatever it trained on. SILVER is 2.6% positives with ~28%
    label noise, so scoring on it would measure agreement with geometry, not crossing.
  * AUC and average precision, never accuracy: predicting never-crossing already scores ~82% on GOLD.
  * Ceiling reminder: two humans agreed on only 86% of these labels, so ~14% of the truth is itself
    contested. That does not translate to a precise AUC bound, but it does mean near-1.0 is not
    available and a few points of difference between runs are not meaningful.

Usage:
    python scripts/04b_train_baseline.py --train-tiers gold     # verified labels only
    python scripts/04b_train_baseline.py --train-tiers all      # + SILVER weak labels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import ROOT

ANNOTATIONS = ROOT / "data" / "annotations"
INDEX = ANNOTATIONS / "dataset_index.parquet"
DEFAULT_REPORT = ROOT / "data" / "processed" / "reports" / "baseline_pvlstm_trained.json"
IMAGE_W, IMAGE_H = 3848.0, 2168.0        # ZOD front camera, for normalising pixel boxes


def load_split(df: pd.DataFrame, horizon: str, obs_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Samples -> (N, obs_len, 8) inputs and (N,) labels.

    Per frame the model sees the normalised box [cx, cy, w, h] and its velocity (frame-to-frame
    delta) — the "PV" in PV-LSTM. Velocity is what carries approach-to-the-kerb; position alone is
    mostly "where in the image", which is scene layout rather than behaviour.

    A window whose boxes are missing (out of the fisheye FOV) is dropped rather than zero-filled: a
    zero box is a confident claim about a pedestrian we could not see.
    """
    xs, ys = [], []
    label_col = f"intent_h{horizon.replace('.', '_')}s"
    for sample_id, label in zip(df.sample_id, df[label_col]):
        doc = json.loads((ANNOTATIONS / f"{sample_id}.json").read_text())
        boxes = [f["bbox_xyxy"] for f in doc["multimodal"]["camera_frames"]]
        boxes = [b for b in boxes if b is not None]
        if len(boxes) < obs_len:
            continue
        b = np.asarray(boxes[-obs_len:], dtype=np.float32)             # last obs_len frames
        cx = (b[:, 0] + b[:, 2]) / 2.0 / IMAGE_W
        cy = (b[:, 1] + b[:, 3]) / 2.0 / IMAGE_H
        w = (b[:, 2] - b[:, 0]) / IMAGE_W
        h = (b[:, 3] - b[:, 1]) / IMAGE_H
        pos = np.stack([cx, cy, w, h], axis=1)
        vel = np.vstack([np.zeros((1, 4), np.float32), np.diff(pos, axis=0)])
        xs.append(np.concatenate([pos, vel], axis=1))
        ys.append(int(label == "crossing"))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


class PVLSTM(nn.Module):
    """PV-LSTM's intention path: separate position and velocity encoders, concatenated to a head.

    Two encoders rather than one over the concatenated input is the original's design — position and
    velocity live on different scales and mean different things, and letting each own its recurrence
    stops the larger-magnitude position channel from dominating the gates.
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.pos = nn.LSTM(4, hidden, batch_first=True)
        self.vel = nn.LSTM(4, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hp, _) = self.pos(x[..., :4])
        _, (hv, _) = self.vel(x[..., 4:])
        return self.head(torch.cat([hp[-1], hv[-1]], dim=1)).squeeze(1)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> dict:
    """AUC + average precision. AP is the one that respects a 17% positive rate."""
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
    both_classes = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)), "n_crossers": int(y.sum()),
        "auc": round(float(roc_auc_score(y, scores)), 4) if both_classes else None,
        "average_precision": round(float(average_precision_score(y, scores)), 4) if both_classes else None,
        "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "positive_rate": round(float(y.mean()), 4),
    }


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_parquet(INDEX)
    gold = df.is_in_gold_standard

    is_train = df.split == "train"
    train_df = df[is_train & gold] if args.train_tiers == "gold" else df[is_train]
    # Selection and testing are ALWAYS on verified GOLD, so the two ablation arms are comparable.
    val_df = df[(df.split == "val") & gold]
    test_df = df[(df.split == "test") & gold]

    xtr, ytr = load_split(train_df, args.horizon, args.obs_len)
    xva, yva = load_split(val_df, args.horizon, args.obs_len)
    xte, yte = load_split(test_df, args.horizon, args.obs_len)
    print(f"train {len(ytr)} ({ytr.sum()} crossers, tiers={args.train_tiers}) | "
          f"val {len(yva)} ({yva.sum()}) | test {len(yte)} ({yte.sum()})  [val/test = GOLD only]")

    device = args.device
    model = PVLSTM(args.hidden, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Positives are ~6% of windows once SILVER is in, so an unweighted loss converges to "never
    # crosses". pos_weight rebalances the gradient without resampling (which would duplicate the few
    # real crossers and overfit them).
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    best, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr_t), device=device)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i: i + args.batch_size]
            opt.zero_grad()
            lossf(model(xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        va = evaluate(model, xva, yva, device)
        history.append({"epoch": epoch + 1, "val_auc": va["auc"]})
        if va["auc"] is not None and va["auc"] > best:      # model selection on GOLD val, never test
            best, best_state = va["auc"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    return {
        "train_tiers": args.train_tiers, "horizon": args.horizon, "seed": args.seed,
        "obs_len": args.obs_len, "epochs": args.epochs,
        "n_train_windows": int(len(ytr)), "n_train_crossers": int(ytr.sum()),
        "best_val_auc": round(best, 4),
        "test_gold": evaluate(model, xte, yte, device),
        "val_gold": evaluate(model, xva, yva, device),
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-tiers", choices=("gold", "all"), default="all",
                    help="gold = verified labels only; all = + SILVER weak labels (the ablation)")
    ap.add_argument("--horizon", default="2.0", choices=("1.0", "1.5", "2.0"))
    ap.add_argument("--obs-len", type=int, default=5, help="observation frames (0.5s at ~10Hz)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat with this many seeds; 35 test crossers make a single run noisy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--report-path", type=Path, default=None)
    args = ap.parse_args()
    report_path = args.report_path or DEFAULT_REPORT.with_name(
        f"baseline_pvlstm_{args.train_tiers}.json")

    runs = []
    for s in range(args.seeds):
        args.seed = s
        runs.append(train(args))
        t = runs[-1]["test_gold"]
        print(f"  seed {s}: GOLD test AUC {t['auc']}  AP {t['average_precision']}")

    aucs = [r["test_gold"]["auc"] for r in runs if r["test_gold"]["auc"] is not None]
    aps = [r["test_gold"]["average_precision"] for r in runs]
    summary = {"train_tiers": args.train_tiers, "horizon": args.horizon, "n_seeds": len(runs),
               "gold_test_auc_mean": round(float(np.mean(aucs)), 4),
               "gold_test_auc_std": round(float(np.std(aucs)), 4),
               "gold_test_ap_mean": round(float(np.mean(aps)), 4),
               "gold_test_ap_std": round(float(np.std(aps)), 4),
               "positive_rate": runs[0]["test_gold"]["positive_rate"],
               "n_train_windows": runs[0]["n_train_windows"],
               "n_train_crossers": runs[0]["n_train_crossers"]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2))

    print(f"\n{args.train_tiers.upper()} training, GOLD test ({runs[0]['test_gold']['n']} windows, "
          f"{runs[0]['test_gold']['n_crossers']} crossers) over {len(runs)} seeds:")
    print(f"  AUC {summary['gold_test_auc_mean']:.3f} ± {summary['gold_test_auc_std']:.3f}")
    print(f"  AP  {summary['gold_test_ap_mean']:.3f} ± {summary['gold_test_ap_std']:.3f}  "
          f"(chance AP = positive rate {summary['positive_rate']})")
    print(f"  report → {report_path}")


if __name__ == "__main__":
    main()
