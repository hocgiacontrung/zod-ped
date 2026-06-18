"""Gate #3 — 2D detector recall against the keyframe pedestrian boxes (image plane).

Counterpart to the frustum POC (`scripts/frustum_poc.py`, the 2D->3D gate). A COCO-trained 2D
detector ("person" class) is run on each sequence's KEYFRAME CAMERA IMAGE and matched by
IoU to the verified keyframe 2D pedestrian boxes. Two reasons this is a cleaner experiment
than a 3D-detector gate:
  * zero temporal offset — the keyframe IS a camera frame (the LiDAR gate detects on a scan
    37-55 ms away);
  * COCO "person" generalizes far better than a nuScenes-pretrained LiDAR detector, and the
    GT set is larger (includes the "2D-only" pedestrians the 3D pipeline skips).

Recall is broken down by GT box HEIGHT (px) — the image-plane proxy for range/size that
recall degrades along — and by occlusion. Output mirrors the 3D gate's report schema.

Usage:
    python scripts/eval_detector2d_recall.py                       # yolo11x, all sequences
    python scripts/eval_detector2d_recall.py --max-seqs 5          # smoke test
    python scripts/eval_detector2d_recall.py --model yolo11m.pt --conf 0.25 --iou-thresh 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.keyframe import (  # noqa: E402
    GtBox2D,
    keyframe_image_path,
    load_keyframe_pedestrians_2d,
)

SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "detector2d_recall_report.json"

# GT box height bins (px) — the image-plane analogue of the 3D gate's range bins.
HEIGHT_BINS = [0, 40, 80, 160, 320, np.inf]
COCO_PERSON_CLASS = 0


@dataclass
class Detection2D:
    """A predicted 2D box in image pixels (x1, y1, x2, y2)."""

    xyxy: np.ndarray
    score: float


def make_yolo(model_name: str = "yolo11x.pt", conf: float = 0.25, imgsz: int = 1280,
              person_class: int = COCO_PERSON_CLASS):
    """Build a YOLO detector once and return a per-image `(path) -> List[Detection2D]` closure.

    `imgsz=1280` (vs the 640 default) matters: ZOD images are 3848x2168, so distant
    pedestrians are tiny and get lost at low input resolution.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not import ultralytics. `pip install ultralytics`. {exc!r}") from exc

    model = YOLO(model_name)

    def detector(image_path: Path) -> List[Detection2D]:
        res = model.predict(source=str(image_path), imgsz=imgsz, conf=conf,
                            classes=[person_class], verbose=False, device=0)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        return [Detection2D(xyxy=b, score=float(s)) for b, s in zip(xyxy, scores)]

    return detector


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (N,4) array of xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_o = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_o - inter
    return np.where(union > 0, inter / union, 0.0)


def match_greedy_iou(preds: List[Detection2D], gts: List[GtBox2D], iou_thresh: float) -> List[bool]:
    """Greedy one-to-one IoU matching, highest-score prediction first. Returns per-GT mask."""
    gt_matched = [False] * len(gts)
    if not gts:
        return gt_matched
    gt_xyxy = np.array([g.xyxy for g in gts])
    for pred in sorted(preds, key=lambda d: -d.score):
        ious = iou_one_to_many(pred.xyxy, gt_xyxy)
        ious[gt_matched] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_thresh:
            gt_matched[best] = True
    return gt_matched


@dataclass
class GtRecord:
    seq_id: str
    height: float
    occlusion: Optional[str]
    has_3d: bool
    matched: bool


@dataclass
class Accumulator:
    records: List[GtRecord] = field(default_factory=list)
    n_pred: int = 0
    n_pred_matched: int = 0
    per_seq: dict = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)

    def add_image(self, seq_id: str, gts: List[GtBox2D], matched: List[bool], n_pred: int) -> None:
        tp = sum(matched)
        self.n_pred += n_pred
        self.n_pred_matched += tp
        self.per_seq[seq_id] = {"n_gt": len(gts), "recall": _safe_div(tp, len(gts))}
        for box, hit in zip(gts, matched):
            self.records.append(GtRecord(seq_id, box.height, box.occlusion, box.has_3d, hit))


def _safe_div(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def _binned_recall(records: List[GtRecord], key, edges) -> List[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in records if lo <= key(r) < hi]
        hit = sum(r.matched for r in sel)
        hi_label = "inf" if hi == np.inf else int(hi)
        out.append({"bin": f"[{int(lo)},{hi_label})", "n_gt": len(sel),
                    "recall": _safe_div(hit, len(sel))})
    return out


def build_report(acc: Accumulator, config: dict) -> dict:
    n_gt = len(acc.records)
    n_hit = sum(r.matched for r in acc.records)
    occl: dict = {}
    for r in acc.records:
        occl.setdefault(r.occlusion, [0, 0])
        occl[r.occlusion][0] += int(r.matched)
        occl[r.occlusion][1] += 1
    has3d = [r for r in acc.records if r.has_3d]
    no3d = [r for r in acc.records if not r.has_3d]
    return {
        "config": config,
        "overall": {
            "n_sequences": len(acc.per_seq),
            "n_gt": n_gt,
            "n_pred": acc.n_pred,
            "tp": n_hit,
            "fn": n_gt - n_hit,
            "fp": acc.n_pred - acc.n_pred_matched,
            "recall": _safe_div(n_hit, n_gt),
            "precision": _safe_div(acc.n_pred_matched, acc.n_pred),
        },
        "by_height_px": _binned_recall(acc.records, lambda r: r.height, HEIGHT_BINS),
        "by_occlusion": {str(k): {"n_gt": v[1], "recall": _safe_div(v[0], v[1])}
                         for k, v in sorted(occl.items(), key=lambda kv: str(kv[0]))},
        "by_has_3d": {
            "with_3d": {"n_gt": len(has3d), "recall": _safe_div(sum(r.matched for r in has3d), len(has3d))},
            "2d_only": {"n_gt": len(no3d), "recall": _safe_div(sum(r.matched for r in no3d), len(no3d))},
        },
        "per_sequence": [{"seq_id": s, **v} for s, v in sorted(acc.per_seq.items())],
        "failures": acc.failures,
    }


def print_summary(report: dict) -> None:
    o = report["overall"]
    print("\n=== 2D detector recall vs keyframe boxes ===")
    print(f"model={report['config']['model']}  conf={report['config']['conf']}  "
          f"iou_thresh={report['config']['iou_thresh']}  imgsz={report['config']['imgsz']}")
    print(f"sequences={o['n_sequences']}  GT={o['n_gt']}  pred={o['n_pred']}")
    print(f"recall={o['recall']}  precision={o['precision']}  (TP={o['tp']} FN={o['fn']} FP={o['fp']})")
    print("\n  recall by GT box height (px):")
    for b in report["by_height_px"]:
        print(f"    {b['bin']+' px':>12}  n={b['n_gt']:<5} recall={b['recall']}")
    print("\n  recall by 3D availability:")
    for k, v in report["by_has_3d"].items():
        print(f"    {k:>10}  n={v['n_gt']:<5} recall={v['recall']}")
    if report["failures"]:
        print(f"\n  {len(report['failures'])} sequence(s) skipped — see report['failures'].")


def run(detector, seq_ids: List[str], iou_thresh: float, config: dict) -> dict:
    acc = Accumulator()
    for i, seq_id in enumerate(seq_ids):
        seq_dir = SEQ_DIR / seq_id
        try:
            img_path, gap = keyframe_image_path(seq_dir)
            if img_path is None:
                acc.failures.append({"seq_id": seq_id, "reason": "no keyframe image"})
                continue
            gts = load_keyframe_pedestrians_2d(seq_dir)
            preds = detector(img_path)
            matched = match_greedy_iou(preds, gts, iou_thresh)
            acc.add_image(seq_id, gts, matched, len(preds))
        except Exception as exc:  # continue-on-error, like the other gates
            acc.failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences")
    return build_report(acc, config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x.pt", help="ultralytics model (auto-downloads)")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for a GT match")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    detector = make_yolo(args.model, conf=args.conf, imgsz=args.imgsz)
    config = {"detector": "yolo2d", "model": args.model, "conf": args.conf,
              "iou_thresh": args.iou_thresh, "imgsz": args.imgsz,
              "n_sequences_requested": len(seq_ids)}
    print(f"Running 2D detector recall: model={args.model}, sequences={len(seq_ids)}")
    report = run(detector, seq_ids, args.iou_thresh, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nReport written → {args.output}")


if __name__ == "__main__":
    main()
