"""Step 1 — GOLD-tier trajectory generation (frustum measurement + KF/RTS linker).

Tracks each KEYFRAME-ANNOTATED pedestrian once over its full 20 s clip and writes a per-
pedestrian world-frame trajectory. This is the GOLD tier: every track is anchored on the
verified ZOD keyframe box (known position + identity), so there is no track-birth / identity
problem — that is deferred to the SILVER tier (detector-discovered peds). See docs/PIPELINE.md
"Step 1 — Direction & open options".

Architecture (DETECTOR-AS-MEASUREMENT + KF/RTS-AS-LINKER):
  0. Detections come from the Step 0 cache (data/processed/detections/, run scripts/00_detect.py
     once); if a sequence is not cached, YOLO is loaded lazily and run live.
  1. Per sequence, build a CANDIDATE POOL once: for every LiDAR scan, take the nearest camera
     image's 2D boxes and lift each to a 3D world position via in-frustum LiDAR depth
     (nearest-depth slab; zodped.labeling.frustum). The pool is pedestrian-independent and shared
     across the sequence's GOLD peds.
  2. Per pedestrian, seed at the keyframe box (world frame) and run the detector-association
     linker (zodped.labeling.tracker.track_pedestrian_from_detections): CV-Kalman predict, gate the
     pool's candidates, take the nearest in-gate one as the measurement, coast on misses, RTS smooth.
  3. After smoothing, assemble the shipped per-frame 3D box: tracked centre + rigid keyframe extent
     + velocity heading (zodped.labeling.boxes.assemble_track_boxes).

Frame convention: each pool frame is timestamped at its LiDAR SCAN time (the world transform that
lifts the frustum centre uses the scan pose), so the filter's dt and the output timestamps are
consistent. The paired camera image is ~tens of ms away (within --max-gap); that small offset is
well inside the frustum's ~0.15 m localization budget.

Output: data/processed/trajectories/{seq_id}_{pedestrian_id}.json — each frame carries the tracked
position and the shipped 3D `box` (centre/size/yaw).
Note: position_ego_rel is per-window and is added at sample assembly, NOT here.

Usage:
    python scripts/00_detect.py                                  # cache YOLO boxes ONCE (~3-4 h)
    python scripts/01_generate_trajectories.py --max-seqs 2       # smoke test (cheap, reads cache)
    python scripts/01_generate_trajectories.py                    # full GOLD set
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from zodped.dataset.keyframe import (
    MAX_LIDAR_GAP_S, GtBox, load_keyframe_pedestrians, parse_zod_ts,
)
from zodped.labeling.boxes import assemble_track_boxes
from zodped.labeling.detection_cache import cache_path, cached_detector, decode_detections, read_doc
from zodped.labeling.detector import make_detector
from zodped.labeling.frustum import build_candidate_pool
from zodped.labeling.tracker import track_pedestrian_from_detections
from zodped.utils.ego_motion import get_T_world_lidar, load_ego_motion
from zodped.utils.projection import load_calibration

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "trajectories"          # per-ped track data only
DEFAULT_DET_DIR = ROOT / "data" / "processed" / "detections"           # Step 0 2D-box cache
DEFAULT_REPORT_PATH = ROOT / "data" / "processed" / "reports" / "trajectories_run_report.json"


class LazyDetector:
    """Build the live YOLO detector only on first cache miss (so a fully-cached run never loads it)."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._detector = None

    def __call__(self, image_path):
        if self._detector is None:
            print("  (detection-cache miss → loading YOLO; run scripts/00_detect.py to avoid this)")
            self._detector = make_detector(self._args.model, conf=self._args.conf, imgsz=self._args.imgsz)
        return self._detector(image_path)


def resolve_detector(seq_id: str, live: LazyDetector, args: argparse.Namespace):
    """Return a per-image detector closure for one sequence: the cache if present, else live YOLO."""
    path = cache_path(args.det_dir, seq_id)
    if not path.exists():
        return live
    doc = read_doc(path)
    meta = doc.get("meta", {})
    if (meta.get("model"), meta.get("imgsz")) != (args.model, args.imgsz):
        print(f"  [warn] {seq_id}: cache built with {meta} but run config is "
              f"model={args.model} imgsz={args.imgsz}; using cache anyway")
    if args.conf < meta.get("conf", 0.0):
        print(f"  [warn] {seq_id}: --conf {args.conf} is below the cache floor "
              f"{meta.get('conf')}; boxes under the floor were never cached")
    return cached_detector(decode_detections(doc, min_conf=args.conf))


def _seed_world(box: GtBox, em: dict, lidar_ext: np.ndarray, keyframe_ts: float) -> np.ndarray:
    """Lift a keyframe box centre (LiDAR frame, keyframe-compensated) into the world frame."""
    T = get_T_world_lidar(em, lidar_ext, keyframe_ts)
    return (T @ np.append(box.center, 1.0))[:3]


def _trajectory_doc(seq_id: str, box: GtBox, keyframe_iso: str, seed_world: np.ndarray,
                    frames: List[dict], config: dict) -> dict:
    """Assemble the per-pedestrian output document (GOLD tier)."""
    n_obs = sum(f["in_observation"] for f in frames)
    return {
        "schema": "trajectory/v0.2",
        "sequence_id": seq_id,
        "pedestrian_id": box.uuid,
        "label_confidence_tier": "high",
        "is_in_gold_standard": True,
        "keyframe_timestamp": keyframe_iso,
        "anchor": {
            "position_world": seed_world.tolist(),
            "box_size_lwh": box.size.tolist(),
            "occlusion": box.occlusion,
        },
        "stats": {
            "n_frames": len(frames),
            "n_observed": int(n_obs),
            "n_coasted": len(frames) - int(n_obs),
            "observed_fraction": round(n_obs / len(frames), 4) if frames else None,
        },
        "config": config,
        "frames": frames,
    }


def process_sequence(seq_id: str, live_detector: LazyDetector, args: argparse.Namespace,
                     config: dict) -> dict:
    """Track every GOLD pedestrian in one sequence; write a JSON per pedestrian.

    Returns a summary dict: {seq_id, n_peds, n_written, n_pool_frames, [error]}.
    """
    seq_dir = SEQ_DIR / seq_id
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    keyframe_iso = json.loads((seq_dir / "info.json").read_text())["keyframe_time"]
    keyframe_ts = parse_zod_ts(keyframe_iso)

    gold = load_keyframe_pedestrians(seq_dir)
    if not gold:
        return {"seq_id": seq_id, "n_peds": 0, "n_written": 0, "n_pool_frames": 0}

    detector = resolve_detector(seq_id, live_detector, args)
    pool = build_candidate_pool(
        seq_dir, detector, em, lidar_ext, calib,
        camera=args.camera, max_gap=args.max_gap, box_shrink=args.box_shrink,
        slab=args.slab, min_pts=args.min_pts,
    )
    if not pool:
        return {"seq_id": seq_id, "n_peds": len(gold), "n_written": 0, "n_pool_frames": 0,
                "error": "empty candidate pool (no paired image/LiDAR frames)"}

    n_written = 0
    for box in gold:
        seed = _seed_world(box, em, lidar_ext, keyframe_ts)
        frames = track_pedestrian_from_detections(
            pool, keyframe_ts, seed,
            gate_mahal2=args.gate_mahal2,
            max_consecutive_misses=args.max_misses,
            meas_sigma=args.meas_sigma,
        )
        if not frames:
            continue
        assemble_track_boxes(frames, box.size, min_speed=args.min_speed)
        doc = _trajectory_doc(seq_id, box, keyframe_iso, seed, frames, config)
        (args.out_dir / f"{seq_id}_{box.uuid}.json").write_text(json.dumps(doc))
        n_written += 1

    return {"seq_id": seq_id, "n_peds": len(gold), "n_written": n_written,
            "n_pool_frames": len(pool)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11x.pt", help="2D detector weights (yolo*/rtdetr*)")
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560, help="detector inference image size")
    ap.add_argument("--camera", default="camera_front_blur")
    ap.add_argument("--max-gap", type=float, default=MAX_LIDAR_GAP_S, help="max image↔scan gap (s)")
    ap.add_argument("--box-shrink", type=float, default=0.6, help="kept central width fraction of each 2D box")
    ap.add_argument("--slab", type=float, default=1.5, help="nearest-depth slab thickness (m)")
    ap.add_argument("--min-pts", type=int, default=3, help="min in-frustum LiDAR points to lift a box")
    ap.add_argument("--gate-mahal2", type=float, default=9.0, help="squared-Mahalanobis association gate")
    ap.add_argument("--max-misses", type=int, default=5, help="consecutive coasted frames before a pass ends")
    ap.add_argument("--meas-sigma", type=float, default=0.3, help="frustum measurement noise std (m)")
    ap.add_argument("--min-speed", type=float, default=0.3, help="speed below which box yaw is filled, not from velocity (m/s)")
    ap.add_argument("--det-dir", type=Path, default=DEFAULT_DET_DIR, help="Step 0 2D-detection cache dir")
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="per-ped trajectory JSONs (data only)")
    ap.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="run report (kept out of the data dir)")
    args = ap.parse_args()

    live_detector = LazyDetector(args)

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {"detector": "frustum(2d+lidar)", "model": args.model, "conf": args.conf,
              "imgsz": args.imgsz, "box_shrink": args.box_shrink, "slab_m": args.slab,
              "min_pts": args.min_pts, "gate_mahal2": args.gate_mahal2,
              "max_consecutive_misses": args.max_misses, "meas_sigma": args.meas_sigma,
              "max_gap_s": args.max_gap, "min_speed": args.min_speed}
    print(f"Step 1 (GOLD): tracking pedestrians over {len(seq_ids)} sequences → {args.out_dir}")

    summaries: List[dict] = []
    failures: List[dict] = []
    total_written = 0
    for i, seq_id in enumerate(seq_ids):
        try:
            s = process_sequence(seq_id, live_detector, args, config)
            summaries.append(s)
            total_written += s["n_written"]
        except Exception as exc:  # continue-on-error, like the other steps
            failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences  (trajectories written: {total_written})")

    report = {"config": config, "n_sequences": len(seq_ids), "n_trajectories": total_written,
              "per_sequence": summaries, "failures": failures}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDone. {total_written} trajectories from {len(seq_ids)} sequences "
          f"({len(failures)} failed). Report → {args.report_path}")


if __name__ == "__main__":
    main()
