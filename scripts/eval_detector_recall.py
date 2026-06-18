"""Step 2 bring-up gate #2 — 3D detector recall against the keyframe pedestrian boxes.

This is the GO/NO-GO experiment for the detector-as-measurement architecture (docs/PIPELINE.md):
ZOD gives no per-frame ground truth, but it gives ~1,776 human-verified 3D pedestrian boxes at
the keyframes. We run a 3D detector on each keyframe scan and measure how many of those boxes it
recovers — broken down by RANGE and by point SPARSITY, because the concern is recall on distant,
sparse pedestrians (off-the-shelf PointPillars weights are trained on KITTI/nuScenes/Waymo, not a
Nordic Velodyne). Good recall → build the MOT around the detector; poor recall at range → fine-tune
on these boxes or keep the gated-centroid measurement as a fallback.

The harness (load → detect → match → metrics → report) is detector-agnostic. The detector sits
behind a small adapter so the harness can be validated TODAY against an `oracle` detector with no
3D-detection framework installed, then switched to `pointpillars` once OpenPCDet is set up.

Usage:
    python scripts/eval_detector_recall.py --detector oracle            # validate the harness
    python scripts/eval_detector_recall.py --detector oracle --max-seqs 5
    python scripts/eval_detector_recall.py --detector pointpillars      # real run (needs adapter)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.keyframe import (  # noqa: E402
    MAX_LIDAR_GAP_S,
    GtBox,
    lidar_ts_from_filename,
    load_keyframe_pedestrians,
    load_xyzi,
    motion_compensate_to_keyframe,
    nearest_lidar_scan,
    parse_zod_ts,
    points_in_box,
)

SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "detector_recall_report.json"

# Bin edges for the breakdowns the decision hinges on.
RANGE_BINS = [0.0, 10.0, 20.0, 30.0, 40.0, np.inf]   # metres, horizontal
POINT_BINS = [1, 5, 10, 20, 50, np.inf]              # LiDAR returns inside the GT box


# ---------------------------------------------------------------------------
# Detection + detector adapters
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A predicted 3D box in the LiDAR sensor frame (same frame as the GT boxes)."""

    center: np.ndarray   # (3,) x, y, z
    size: np.ndarray     # (3,) length, width, height
    score: float


# A detector maps (points (N,4) xyzi, GT boxes) -> predicted pedestrian detections.
# `gt` is supplied only so the oracle can synthesise plausible predictions; real detectors
# must ignore it (predicting from `gt` would defeat the purpose of the experiment).
Detector = Callable[[np.ndarray, List[GtBox]], List[Detection]]


def make_oracle(seed: int = 0, jitter_m: float = 0.25, fp_per_scan: float = 0.5) -> Detector:
    """A stand-in detector that lets us validate the harness with no real model installed.

    It derives predictions from the GT boxes, then deliberately degrades them so the metrics
    have something to measure:
      * drops boxes with a probability that RISES with range (mimics the real failure mode —
        sparse, distant pedestrians are missed), so per-range recall should visibly fall off;
      * jitters surviving centres by `jitter_m`;
      * sprinkles a few false positives so precision < 1.
    Deterministic given `seed`. This is test scaffolding, NOT a model.
    """
    rng = np.random.default_rng(seed)

    def detector(points: np.ndarray, gt: List[GtBox]) -> List[Detection]:
        dets: List[Detection] = []
        for box in gt:
            p_keep = float(np.clip(1.0 - box.range_xy / 60.0, 0.05, 1.0))  # ~1.0 near, ~0.3 far
            if rng.random() > p_keep:
                continue
            dets.append(
                Detection(
                    center=box.center + rng.normal(0.0, jitter_m, size=3),
                    size=box.size.copy(),
                    score=float(rng.uniform(0.4, 0.99)),
                )
            )
        for _ in range(rng.poisson(fp_per_scan)):
            dets.append(
                Detection(
                    center=rng.uniform([-40, -40, -2], [40, 40, 2]),
                    size=np.array([0.7, 0.7, 1.7]),
                    score=float(rng.uniform(0.3, 0.6)),
                )
            )
        return dets

    return detector


# --- Coordinate adapter: ZOD LiDAR frame -> OpenPCDet model frame --------------------
# OpenPCDet's unified frame is x-forward, y-left, z-up. DATA_FORMAT.md documents ZOD LiDAR as
# x-forward, y-left, z-up too, so the default is the IDENTITY rotation. Override only after a
# visual sanity check (see WARNING). `None` means identity.
#
# WARNING — VERIFY BEFORE TRUSTING ANY NUMBER. On seq 000007 the *y* axis has the larger extent
# (p99 ~59m vs x ~27m), which on a single scan can look like y-forward. That is most likely just
# this (intersection) scene's geometry, but a wrong axis mapping silently destroys recall. Before
# the full run: overlay predictions on ONE scan (e.g. with open3d) and confirm boxes land on real
# pedestrians. If they are rotated 90deg, set LIDAR_TO_MODEL_ROT to the swap below.
LIDAR_TO_MODEL_ROT = None  # e.g. np.array([[0,1,0],[-1,0,0],[0,0,1]]) to map y-forward -> x-forward


def make_pointpillars(
    cfg_file: Path,
    ckpt: Path,
    score_thresh: float = 0.3,
    ped_class: str = "pedestrian",
    intensity_scale: float = 1.0,
    num_point_features: Optional[int] = None,
) -> Detector:
    """Build an OpenPCDet detector ONCE and return a per-scan `Detector` closure.

    Env prerequisites (see requirements-detector.txt):
      * numpy < 2  (torch 2.2 here is built against NumPy 1.x; NumPy 2.x breaks the interop)
      * OpenPCDet + spconv-cu121 + a pedestrian-capable checkpoint (nuScenes/Waymo pretrained).

    Adapter behaviour (the parts that silently break recall if wrong):
      * Coordinate frame: ZOD x-fwd/y-left/z-up -> model frame (identity by default; see
        LIDAR_TO_MODEL_ROT and its WARNING).
      * Intensity: ZOD is uint8 0-255; scaled by `intensity_scale`. DEFAULT 1.0 (raw 0-255):
        OpenPCDet's nuScenes pipeline feeds intensity raw, and the nuScenes PP-multihead
        checkpoint was verified to need it — scaling by 1/255 silently drops ALL pedestrian
        detections (weak returns) on seq 000007 while cars still survive. Override per checkpoint.
      * Feature width: padded/truncated to what the model expects (e.g. nuScenes wants a 5th
        'timestamp' channel; we pad zeros).
    Returns pedestrian detections above `score_thresh`, in the LiDAR frame (so they line up
    with the GT boxes).
    """
    import logging

    try:
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
    except Exception as exc:  # noqa: BLE001 — surface the real cause with guidance
        raise RuntimeError(
            "Could not import torch/OpenPCDet. Install per requirements-detector.txt "
            f"(numpy<2 + OpenPCDet + spconv-cu121). Original error: {exc!r}"
        ) from exc

    cfg_from_yaml_file(str(cfg_file), cfg)
    class_names = list(cfg.CLASS_NAMES)

    class _InferDataset(DatasetTemplate):
        def __init__(self) -> None:
            super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                             training=False, root_path=None, logger=logging.getLogger("pp"))

    infer_ds = _InferDataset()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=infer_ds)
    model.load_params_from_file(filename=str(ckpt), logger=logging.getLogger("pp"), to_cpu=False)
    model.cuda()
    model.eval()

    n_feat = num_point_features or len(cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list)
    ped_label = next((i + 1 for i, c in enumerate(class_names) if c.lower() == ped_class.lower()), None)
    if ped_label is None:
        raise ValueError(f"'{ped_class}' not in checkpoint classes {class_names}")

    def detector(points: np.ndarray, gt: List[GtBox]) -> List[Detection]:
        pts = points[:, :4].astype(np.float32).copy()
        pts[:, 3] *= intensity_scale
        if LIDAR_TO_MODEL_ROT is not None:
            pts[:, :3] = pts[:, :3] @ LIDAR_TO_MODEL_ROT.T
        if n_feat > pts.shape[1]:                      # model expects extra channels (e.g. time)
            pad = np.zeros((pts.shape[0], n_feat - pts.shape[1]), np.float32)
            pts = np.concatenate([pts, pad], axis=1)
        elif n_feat < pts.shape[1]:
            pts = pts[:, :n_feat]

        data = infer_ds.collate_batch([infer_ds.prepare_data({"points": pts})])
        load_data_to_gpu(data)
        with torch.no_grad():
            pred, _ = model.forward(data)
        boxes = pred[0]["pred_boxes"].cpu().numpy()    # (M,7) x,y,z,dx,dy,dz,heading
        scores = pred[0]["pred_scores"].cpu().numpy()
        labels = pred[0]["pred_labels"].cpu().numpy()

        dets: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            if score < score_thresh or label != ped_label:
                continue
            center = box[:3]
            if LIDAR_TO_MODEL_ROT is not None:         # map detections back to the LiDAR frame
                center = center @ LIDAR_TO_MODEL_ROT
            dets.append(Detection(center=center, size=box[3:6], score=float(score)))
        return dets

    return detector


# Point-cloud range the KITTI checkpoint was trained on (x-fwd / y-left / z-up, metres).
# Forward-only and capped at ~69 m — distant/behind pedestrians fall outside the pillar grid,
# so this range alone reproduces the long-range collapse we expect from a KITTI-trained model.
ZHU_RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)


def make_pointpillars_zhu(
    ckpt: Path,
    score_thresh: float = 0.1,
    intensity_scale: float = 1.0 / 255.0,
    repo_path: Path = Path.home() / "PointPillars",
) -> Detector:
    """Build the zhulf0804/PointPillars KITTI detector and return a per-scan `Detector`.

    This is the supervisor-recommended PointPillars run OFF-THE-SHELF (KITTI weights, no
    fine-tuning) to measure whether vanilla PointPillars transfers to ZOD pedestrians. It is a
    clean single-model reimplementation of Lang et al., CVPR 2019 — pillar encoder + 2D CNN, so
    it needs NO sparse-conv (spconv) stack, unlike the OpenPCDet path.

    KITTI conventions the adapter honours (each silently wrecks recall if wrong):
      * Classes are {Pedestrian:0, Cyclist:1, Car:2}; we keep label 0.
      * Frame is x-fwd / y-left / z-up — identical to ZOD, so no rotation.
      * Intensity is reflectance in [0, 1]; ZOD is uint8 0-255 -> default scale 1/255. This is the
        OPPOSITE of the nuScenes checkpoint (which needed raw 0-255); feeding raw here saturates.
      * Points outside `ZHU_RANGE` are dropped by the pillar grid regardless; we pre-filter so
        distant/behind pedestrians are honestly scored as misses.
    """
    import torch

    sys.path.insert(0, str(repo_path))
    try:
        from pointpillars.model import PointPillars
    except Exception as exc:  # noqa: BLE001 — surface the real cause with guidance
        raise RuntimeError(
            f"Could not import zhulf0804 PointPillars from {repo_path}. Clone it and build the ops "
            f"(`pip install -e . --no-build-isolation`). Original error: {exc!r}"
        ) from exc

    model = PointPillars(nclasses=3).cuda()
    model.load_state_dict(torch.load(str(ckpt)))
    model.eval()
    lo = np.array(ZHU_RANGE[:3], np.float32)
    hi = np.array(ZHU_RANGE[3:], np.float32)

    def detector(points: np.ndarray, gt: List[GtBox]) -> List[Detection]:
        pts = points[:, :4].astype(np.float32).copy()
        pts[:, 3] *= intensity_scale
        keep = np.all((pts[:, :3] > lo) & (pts[:, :3] < hi), axis=1)
        pts = pts[keep]
        if len(pts) == 0:
            return []
        with torch.no_grad():
            res = model(batched_pts=[torch.from_numpy(pts).cuda()], mode="test")[0]
        dets: List[Detection] = []
        for box, score, label in zip(res["lidar_bboxes"], res["scores"], res["labels"]):
            if int(label) != 0 or float(score) < score_thresh:  # 0 == Pedestrian
                continue
            dets.append(Detection(center=np.asarray(box[:3]), size=np.asarray(box[3:6]),
                                  score=float(score)))
        return dets

    return detector


DETECTOR_NAMES = ["oracle", "pointpillars", "pointpillars_zhu"]


# ---------------------------------------------------------------------------
# Matching + metrics
# ---------------------------------------------------------------------------
def match_greedy(preds: List[Detection], gts: List[GtBox], match_dist_m: float) -> List[bool]:
    """Greedy one-to-one matching by BEV centre distance, highest-score prediction first.

    BEV centre distance (not 3D-box IoU) for the same reason the tracker uses centroids: at range
    a pedestrian is a few points, so IoU between tiny boxes is unstable while centre proximity is
    robust. Returns a per-GT matched mask aligned with `gts`.
    """
    gt_matched = [False] * len(gts)
    if not gts:
        return gt_matched
    gt_xy = np.array([g.center[:2] for g in gts])
    for pred in sorted(preds, key=lambda d: -d.score):
        d = np.linalg.norm(gt_xy - pred.center[:2], axis=1)
        d[gt_matched] = np.inf
        best = int(np.argmin(d))
        if d[best] <= match_dist_m:
            gt_matched[best] = True
    return gt_matched


@dataclass
class GtRecord:
    """One keyframe GT box plus whether the detector recovered it (for binning)."""

    seq_id: str
    range_xy: float
    n_points: int
    occlusion: Optional[str]
    matched: bool


@dataclass
class Accumulator:
    records: List[GtRecord] = field(default_factory=list)
    n_pred: int = 0
    n_pred_matched: int = 0       # true positives
    per_seq: dict = field(default_factory=dict)
    failures: List[dict] = field(default_factory=list)

    def add_scan(self, seq_id: str, gts: List[GtBox], points: np.ndarray,
                 matched: List[bool], n_pred: int) -> None:
        tp = sum(matched)
        self.n_pred += n_pred
        self.n_pred_matched += tp
        self.per_seq[seq_id] = {"n_gt": len(gts), "recall": _safe_div(tp, len(gts))}
        for box, hit in zip(gts, matched):
            self.records.append(
                GtRecord(seq_id, box.range_xy, points_in_box(points, box),
                         box.occlusion, hit)
            )


def _safe_div(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def _binned_recall(records: List[GtRecord], key, edges) -> List[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in records if lo <= key(r) < hi]
        hit = sum(r.matched for r in sel)
        hi_label = "inf" if hi == np.inf else (int(hi) if float(hi).is_integer() else hi)
        out.append({"bin": f"[{int(lo)},{hi_label})", "n_gt": len(sel),
                    "recall": _safe_div(hit, len(sel))})
    return out


def build_report(acc: Accumulator, config: dict) -> dict:
    n_gt = len(acc.records)
    n_hit = sum(r.matched for r in acc.records)
    occl = {}
    for r in acc.records:
        occl.setdefault(r.occlusion, [0, 0])
        occl[r.occlusion][0] += int(r.matched)
        occl[r.occlusion][1] += 1
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
        "by_range_xy_m": _binned_recall(acc.records, lambda r: r.range_xy, RANGE_BINS),
        "by_lidar_points": _binned_recall(acc.records, lambda r: r.n_points, POINT_BINS),
        "by_occlusion": {str(k): {"n_gt": v[1], "recall": _safe_div(v[0], v[1])}
                         for k, v in sorted(occl.items(), key=lambda kv: str(kv[0]))},
        "per_sequence": [{"seq_id": s, **v} for s, v in sorted(acc.per_seq.items())],
        "failures": acc.failures,
    }


def print_summary(report: dict) -> None:
    o = report["overall"]
    print("\n=== Detector recall vs keyframe boxes ===")
    print(f"detector={report['config']['detector']}  match_dist={report['config']['match_dist_m']}m")
    print(f"sequences={o['n_sequences']}  GT={o['n_gt']}  pred={o['n_pred']}")
    print(f"recall={o['recall']}  precision={o['precision']}  (TP={o['tp']} FN={o['fn']} FP={o['fp']})")
    print("\n  recall by horizontal range:")
    for b in report["by_range_xy_m"]:
        print(f"    {b['bin']+' m':>12}  n={b['n_gt']:<5} recall={b['recall']}")
    print("\n  recall by #LiDAR points in box:")
    for b in report["by_lidar_points"]:
        print(f"    {b['bin']:>12}  n={b['n_gt']:<5} recall={b['recall']}")
    if report["failures"]:
        print(f"\n  {len(report['failures'])} sequence(s) skipped — see report['failures'].")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(detector: Detector, seq_ids: List[str], match_dist_m: float, config: dict,
        compensate: bool = True) -> dict:
    acc = Accumulator()
    for i, seq_id in enumerate(seq_ids):
        seq_dir = SEQ_DIR / seq_id
        try:
            keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])
            scan_path, gap = nearest_lidar_scan(seq_dir / "lidar_velodyne", keyframe_ts)
            if scan_path is None or gap > MAX_LIDAR_GAP_S:
                acc.failures.append({"seq_id": seq_id, "reason": f"no LiDAR within {MAX_LIDAR_GAP_S}s (gap={gap:.3f})"})
                continue
            gts = load_keyframe_pedestrians(seq_dir)
            points = load_xyzi(scan_path)
            if compensate:
                # Lift the +37ms scan into the LiDAR frame at the keyframe instant, where the GT
                # boxes live (see motion_compensate_to_keyframe). Matches the frustum eval.
                points = motion_compensate_to_keyframe(
                    points, seq_dir, lidar_ts_from_filename(scan_path.name), keyframe_ts)
            preds = detector(points, gts)
            matched = match_greedy(preds, gts, match_dist_m)
            acc.add_scan(seq_id, gts, points, matched, len(preds))
        except Exception as exc:  # continue-on-error, like Step 1
            acc.failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences")
    return build_report(acc, config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=DETECTOR_NAMES, default="oracle")
    ap.add_argument("--match-dist", type=float, default=2.0, help="BEV centre-distance match threshold (m)")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="oracle RNG seed")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # pointpillars-only:
    ap.add_argument("--pp-cfg", type=Path, help="OpenPCDet model .yaml (required for pointpillars)")
    ap.add_argument("--pp-ckpt", type=Path, help="OpenPCDet checkpoint .pth (required for pointpillars)")
    ap.add_argument("--score-thresh", type=float, default=0.3, help="detector score threshold")
    ap.add_argument("--intensity-scale", type=float, default=None,
                    help="multiplier on ZOD's 0-255 intensity; default per detector "
                         "(nuScenes PP=1.0 raw, KITTI zhu=1/255 reflectance)")
    ap.add_argument("--no-compensate", action="store_true",
                    help="skip keyframe motion compensation (NOT recommended; GT is keyframe-aligned)")
    args = ap.parse_args()

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    iscale = args.intensity_scale
    if args.detector == "oracle":
        detector: Detector = make_oracle(seed=args.seed)
        iscale = None
    elif args.detector == "pointpillars":
        if not (args.pp_cfg and args.pp_ckpt):
            ap.error("--detector pointpillars requires --pp-cfg and --pp-ckpt")
        iscale = 1.0 if iscale is None else iscale  # nuScenes wants raw 0-255
        detector = make_pointpillars(args.pp_cfg, args.pp_ckpt, score_thresh=args.score_thresh,
                                     intensity_scale=iscale)
    else:  # pointpillars_zhu
        if not args.pp_ckpt:
            ap.error("--detector pointpillars_zhu requires --pp-ckpt")
        iscale = 1.0 / 255.0 if iscale is None else iscale  # KITTI wants reflectance 0-1
        detector = make_pointpillars_zhu(args.pp_ckpt, score_thresh=args.score_thresh,
                                         intensity_scale=iscale)

    config = {"detector": args.detector, "match_dist_m": args.match_dist,
              "n_sequences_requested": len(seq_ids), "seed": args.seed,
              "score_thresh": args.score_thresh, "intensity_scale": iscale,
              "keyframe_compensated": not args.no_compensate,
              "pp_cfg": str(args.pp_cfg) if args.pp_cfg else None,
              "pp_ckpt": str(args.pp_ckpt) if args.pp_ckpt else None}
    print(f"Running detector recall: detector={args.detector}, sequences={len(seq_ids)}")
    report = run(detector, seq_ids, args.match_dist, config, compensate=not args.no_compensate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_summary(report)
    print(f"\nReport written → {args.output}")


if __name__ == "__main__":
    main()
