"""Render a full-frame demo video of a sequence — every GOLD pedestrian tracked at once.

The SUPERVISOR demo: the whole camera frame over a central segment of the clip, with EVERY
keyframe-annotated (GOLD) pedestrian drawn simultaneously — its 2D detection box and its tracked 3D
position marker, each in a consistent per-pedestrian colour so identity is legible over time. Coasted
frames (detector matched nothing — typically occlusion) draw a hollow marker labelled "coast", so the
CV-Kalman bridging an occlusion is visible. Unmatched detector boxes are drawn faint grey.

This shows GOLD tier only (keyframe-anchored pedestrians); SILVER (detector-found) is not built yet.

Per camera frame in the window: run the 2D detector once, then for each pedestrian look up its tracked
world position at this instant, project world -> LiDAR(t) -> image (Kannala), draw the marker, and
highlight the detector box the marker falls into. Frames are written to an MP4 via OpenCV.

Usage:
    python scripts/viz_render_sequence_video.py --seq 000007
    python scripts/viz_render_sequence_video.py --seq 000007 --window 8 --fps 10 --scale 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from zodped.dataset.keyframe import parse_zod_ts
from zodped.labeling.detector import make_detector
from zodped.utils.ego_motion import get_T_world_lidar, load_ego_motion
from zodped.utils.projection import load_calibration, project_lidar_to_image

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"
DEFAULT_REVIEW_DIR = ROOT / "data" / "processed" / "review"

# Distinct per-pedestrian colours (BGR for OpenCV). Cycled by sorted pedestrian index.
_PALETTE = [
    (66, 233, 245), (80, 175, 76), (60, 76, 231), (244, 124, 33), (180, 70, 200),
    (0, 215, 255), (139, 195, 74), (130, 130, 0), (203, 100, 255), (0, 140, 255),
]


def load_ped_tracks(traj_dir: Path, seq: str) -> List[dict]:
    """Load every GOLD pedestrian track for one sequence as time-indexed arrays."""
    tracks = []
    for f in sorted(traj_dir.glob(f"{seq}_*.json")):
        if f.name.startswith("_"):
            continue
        doc = json.loads(f.read_text())
        frames = doc["frames"]
        tracks.append({
            "id": doc["pedestrian_id"][:8],
            "ts": np.array([parse_zod_ts(fr["timestamp"]) for fr in frames]),
            "pos": np.array([fr["position_world"] for fr in frames]),
            "obs": np.array([fr["in_observation"] for fr in frames]),
        })
    return tracks


def _pos_at(track: dict, ts: float, tol: float) -> Optional[tuple]:
    """Tracked (world position, in_observation) nearest `ts`, or None if outside coverage/tol."""
    i = int(np.argmin(np.abs(track["ts"] - ts)))
    if abs(track["ts"][i] - ts) > tol:
        return None
    return track["pos"][i], bool(track["obs"][i])


def _project(p_world: np.ndarray, T_world_lidar: np.ndarray, calib: dict) -> Optional[np.ndarray]:
    """World point -> image pixel (u, v), or None if behind/outside the camera."""
    p_lidar = (np.linalg.inv(T_world_lidar) @ np.append(p_world, 1.0))[:3]
    uv, valid = project_lidar_to_image(p_lidar.reshape(1, 3), calib)
    return uv[0] if valid[0] else None


def _box_holding(boxes, u: float, v: float):
    """Smallest detector box containing pixel (u, v), or None."""
    holding = [b for b in boxes if b.xyxy[0] <= u <= b.xyxy[2] and b.xyxy[1] <= v <= b.xyxy[3]]
    if not holding:
        return None
    return min(holding, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))


def render_sequence_video(seq: str, detector, traj_dir: Path, out_path: Path,
                          window_s: float, fps: int, scale: float, match_tol: float) -> int:
    """Render the demo MP4 for one sequence. Returns the number of frames written."""
    import cv2

    seq_dir = SEQ_DIR / seq
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])

    tracks = load_ped_tracks(traj_dir, seq)
    if not tracks:
        raise SystemExit(f"No GOLD trajectories for seq {seq} in {traj_dir}")
    colors = {t["id"]: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(tracks)}

    imgs = sorted((seq_dir / "camera_front_blur").glob("*.jpg"))
    img_ts = np.array([parse_zod_ts(p.stem.rsplit("_", 1)[-1]) for p in imgs])
    in_win = np.abs(img_ts - keyframe_ts) <= window_s / 2.0
    frame_paths = [imgs[i] for i in np.where(in_win)[0]]
    if not frame_paths:
        raise SystemExit(f"No camera frames within {window_s}s of the keyframe for seq {seq}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h0, w0 = cv2.imread(str(frame_paths[0])).shape[:2]
    W, H = int(w0 * scale), int(h0 * scale)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for fp in frame_paths:
        ts = parse_zod_ts(fp.stem.rsplit("_", 1)[-1])
        img = cv2.resize(cv2.imread(str(fp)), (W, H))
        boxes = detector(fp)
        T_world_lidar = get_T_world_lidar(em, lidar_ext, ts)

        claimed = set()
        n_drawn = 0
        for track in tracks:
            at = _pos_at(track, ts, match_tol)
            if at is None:
                continue
            p_world, observed = at
            uv = _project(p_world, T_world_lidar, calib)
            if uv is None:
                continue
            u, v = float(uv[0]) * scale, float(uv[1]) * scale
            color = colors[track["id"]]
            n_drawn += 1
            box = _box_holding(boxes, uv[0], uv[1]) if observed else None
            if box is not None:
                claimed.add(id(box))
                x1, y1, x2, y2 = (box.xyxy * scale).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, track["id"], (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if observed:
                cv2.circle(img, (int(u), int(v)), 5, color, -1)
            else:  # coasted / occluded — hollow marker, labelled
                cv2.circle(img, (int(u), int(v)), 6, color, 2)
                cv2.putText(img, "coast", (int(u) + 8, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        for b in boxes:  # detector boxes no pedestrian claimed -> faint grey context
            if id(b) in claimed:
                continue
            x1, y1, x2, y2 = (b.xyxy * scale).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)

        dt = ts - keyframe_ts
        cv2.putText(img, f"seq {seq}  t={dt:+.1f}s from keyframe   {n_drawn} GOLD peds tracked",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    return len(frame_paths)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence id, e.g. 000007")
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--out", type=Path, default=None, help="output mp4 (default: review/seq_{seq}.mp4)")
    ap.add_argument("--window", type=float, default=8.0, help="seconds centred on the keyframe")
    ap.add_argument("--fps", type=int, default=10, help="output video frame rate")
    ap.add_argument("--scale", type=float, default=0.5, help="downscale factor for the 3848x2168 frames")
    ap.add_argument("--match-tol", type=float, default=0.08, help="max |image-track| time gap to draw a ped (s)")
    ap.add_argument("--model", default="yolo11x.pt", help="2D detector weights")
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560, help="detector inference image size")
    args = ap.parse_args()

    detector = make_detector(args.model, conf=args.conf, imgsz=args.imgsz)

    out = args.out or DEFAULT_REVIEW_DIR / f"seq_{args.seq}.mp4"
    n = render_sequence_video(args.seq, detector, args.traj_dir, out,
                              args.window, args.fps, args.scale, args.match_tol)
    print(f"wrote {n} frames → {out}")


if __name__ == "__main__":
    main()
