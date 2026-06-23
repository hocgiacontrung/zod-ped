"""Step 1 — GOLD-tier trajectory generation (frustum measurement + KF/RTS linker).

Tracks each KEYFRAME-ANNOTATED pedestrian once over its full 20 s clip and writes a per-
pedestrian world-frame trajectory. This is the GOLD tier: every track is anchored on the
verified ZOD keyframe box (known position + identity), so there is no track-birth / identity
problem — that is deferred to the SILVER tier (detector-discovered peds). See docs/PIPELINE.md
"Step 1 — Direction & open options".

Architecture (DETECTOR-AS-MEASUREMENT + KF/RTS-AS-LINKER):
  1. Per sequence, build a CANDIDATE POOL once: for every LiDAR scan, run the 2D detector on the
     nearest camera image, lift each person box to a 3D world position via the in-frustum LiDAR
     depth (src/labeling/frustum.py). The pool is pedestrian-independent and shared across the
     sequence's GOLD peds (the detector is the expensive part — run it once per frame).
  2. Per pedestrian, seed at the keyframe box (world frame) and run the detector-association
     linker (src/labeling/tracker.track_pedestrian_from_detections): CV-Kalman predict, gate the
     pool's candidates, take the nearest in-gate one as the measurement, coast on misses, RTS smooth.

Frame convention: each pool frame is timestamped at its LiDAR SCAN time (the world transform that
lifts the frustum centre uses the scan pose), so the filter's dt and the output timestamps are
consistent. The paired camera image is ~tens of ms away (within --max-gap); that small offset is
well inside the frustum's ~0.15 m localization budget.

Output: data/processed/trajectories/{seq_id}_{pedestrian_id}.json
Note: position_ego_rel is per-window and is added at sample assembly, NOT here.

Usage:
    python scripts/01_generate_trajectories.py --max-seqs 2        # smoke test
    python scripts/01_generate_trajectories.py                     # full GOLD set
    python scripts/01_generate_trajectories.py --conf 0.1 --imgsz 2560 --gate-mahal2 9.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset.keyframe import (  # noqa: E402
    MAX_LIDAR_GAP_S, GtBox, lidar_ts_from_filename, load_keyframe_pedestrians,
    load_xyzi, parse_zod_ts,
)
from labeling.frustum import lift_image_detections  # noqa: E402
from labeling.tracker import track_pedestrian_from_detections  # noqa: E402
from utils.ego_motion import get_T_world_lidar, load_ego_motion  # noqa: E402
from utils.projection import load_calibration  # noqa: E402

SEQ_DIR = ROOT / "data" / "raw" / "sequences"
PED_SEQUENCES = ROOT / "data" / "pedestrian_sequences.json"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "trajectories"


def _image_table(seq_dir: Path, camera: str) -> Tuple[List[Path], np.ndarray]:
    """Return (image paths, their Unix timestamps) for a sequence's camera stream, sorted."""
    files = sorted((seq_dir / camera).glob("*.jpg"))
    ts = np.array([parse_zod_ts(f.stem.rsplit("_", 1)[-1]) for f in files])
    return files, ts


def build_candidate_pool(
    seq_dir: Path, detector, em: dict, lidar_ext: np.ndarray, calib: dict,
    args: argparse.Namespace,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Lift every LiDAR scan's 2D detections to world-frame candidates (once per sequence).

    Scan-driven (LiDAR is the depth source): each scan is paired with the nearest camera image
    within --max-gap; scans with no close image are skipped. Detector results are cached per
    image, since consecutive scans can share one image.
    """
    img_paths, img_ts = _image_table(seq_dir, args.camera)
    scans = sorted((seq_dir / "lidar_velodyne").glob("*.npy"))
    if not img_paths or not scans:
        return []

    det_cache: Dict[Path, list] = {}
    pool: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for scan in scans:
        scan_ts = lidar_ts_from_filename(scan.name)
        j = int(np.argmin(np.abs(img_ts - scan_ts)))
        if abs(img_ts[j] - scan_ts) > args.max_gap:
            continue
        img_path = img_paths[j]
        if img_path not in det_cache:
            det_cache[img_path] = detector(img_path)
        boxes = det_cache[img_path]

        points = load_xyzi(scan)[:, :3].astype(np.float64)
        T_world_lidar = get_T_world_lidar(em, lidar_ext, scan_ts)
        lifts = lift_image_detections(
            boxes, points, calib, T_world_lidar, args.box_shrink, args.slab, args.min_pts,
        )
        cand_world = np.array([lf.center_world for lf in lifts], dtype=np.float64).reshape(-1, 3)
        cand_n = np.array([lf.n_points for lf in lifts], dtype=int)
        pool.append((scan_ts, cand_world, cand_n))
    return pool


def _seed_world(box: GtBox, em: dict, lidar_ext: np.ndarray, keyframe_ts: float) -> np.ndarray:
    """Lift a keyframe box centre (LiDAR frame, keyframe-compensated) into the world frame."""
    T = get_T_world_lidar(em, lidar_ext, keyframe_ts)
    return (T @ np.append(box.center, 1.0))[:3]


def _trajectory_doc(seq_id: str, box: GtBox, keyframe_iso: str, seed_world: np.ndarray,
                    frames: List[dict], config: dict) -> dict:
    """Assemble the per-pedestrian output document (GOLD tier)."""
    n_obs = sum(f["in_observation"] for f in frames)
    return {
        "schema": "trajectory/v0.1",
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


def process_sequence(seq_id: str, detector, args: argparse.Namespace, config: dict) -> dict:
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

    pool = build_candidate_pool(seq_dir, detector, em, lidar_ext, calib, args)
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
        doc = _trajectory_doc(seq_id, box, keyframe_iso, seed, frames, config)
        out_path = args.out_dir / f"{seq_id}_{box.uuid}.json"
        out_path.write_text(json.dumps(doc))
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
    ap.add_argument("--max-seqs", type=int, default=None, help="cap sequences (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    from eval_detector2d_recall import make_detector  # noqa: E402
    detector = make_detector(args.model, conf=args.conf, imgsz=args.imgsz)

    seq_ids = [e["seq_id"] for e in json.loads(PED_SEQUENCES.read_text())]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {"detector": "frustum(2d+lidar)", "model": args.model, "conf": args.conf,
              "imgsz": args.imgsz, "box_shrink": args.box_shrink, "slab_m": args.slab,
              "min_pts": args.min_pts, "gate_mahal2": args.gate_mahal2,
              "max_consecutive_misses": args.max_misses, "meas_sigma": args.meas_sigma,
              "max_gap_s": args.max_gap, "tier": "gold"}
    print(f"Step 1 (GOLD): tracking pedestrians over {len(seq_ids)} sequences → {args.out_dir}")

    summaries: List[dict] = []
    failures: List[dict] = []
    total_written = 0
    for i, seq_id in enumerate(seq_ids):
        try:
            s = process_sequence(seq_id, detector, args, config)
            summaries.append(s)
            total_written += s["n_written"]
        except Exception as exc:  # continue-on-error, like the other steps
            failures.append({"seq_id": seq_id, "reason": repr(exc)[:200]})
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(seq_ids)} sequences  (trajectories written: {total_written})")

    report = {"config": config, "n_sequences": len(seq_ids), "n_trajectories": total_written,
              "per_sequence": summaries, "failures": failures}
    report_path = args.out_dir / "_run_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDone. {total_written} trajectories from {len(seq_ids)} sequences "
          f"({len(failures)} failed). Report → {report_path}")


if __name__ == "__main__":
    main()
