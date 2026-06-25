"""Render a demo video of tracked pedestrians — full camera, or ZOD-website-style camera | 3D.

Two layouts (`--layout`):

* ``split`` (default) — the ZOD-website look: a wide side-by-side frame with the FRONT CAMERA on the
  left and an oblique 3D PERSPECTIVE point-cloud render on the right (white background, LiDAR coloured
  by log-intensity, same scheme as the devkit's ``zod/cli/visualize_lidar.py``). On the cloud we draw
  each GOLD pedestrian as an oriented 3D box at its tracked world centroid (heading from the track's
  velocity, size from the keyframe anchor box), dimmed to a thin wireframe while the track is COASTING
  through an occlusion. The same centroid is projected back onto the camera image so the panels are
  linked. The 3D panel uses Open3D's headless OFFSCREEN renderer (EGL) and is supersampled for crisp,
  anti-aliased points.

* ``camera`` — the full camera frame only (no 3D panel, no Open3D dependency): every GOLD pedestrian's
  2D detector box + projected 3D centroid at full resolution, the lightweight 2D-QC / supervisor demo.
  Driven by camera frames (smoother) rather than the ~10 Hz LiDAR scans.

Both layouts share the camera-panel drawing; identity is a consistent per-pedestrian colour, coasted
frames draw a hollow marker labelled "coast", and detector boxes no pedestrian claimed are faint grey.
Only GOLD tier (keyframe-anchored pedestrians) is drawn; SILVER (detector-found) is not built yet.

Usage:
    python scripts/viz_render_video.py --seq 000007
    python scripts/viz_render_video.py --seq 000007 --layout camera --scale 0.5
    python scripts/viz_render_video.py --seq 000007 --window 10 --fps 15
    python scripts/viz_render_video.py --seq 000007 --eye 0 -16 11 --look 0 18 0   # tune view
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zodped.dataset.keyframe import lidar_ts_from_filename, load_xyzi, parse_zod_ts
from zodped.labeling.detector import make_detector
from zodped.utils.ego_motion import get_T_world_lidar, load_ego_motion
from zodped.utils.projection import load_calibration, project_lidar_to_image
from zodped.utils.video import FFmpegWriter

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ROOT / "data" / "raw" / "sequences"
DEFAULT_TRAJ_DIR = ROOT / "data" / "processed" / "trajectories"
DEFAULT_REVIEW_DIR = ROOT / "data" / "processed" / "review"

# Distinct per-pedestrian box colours (RGB 0-1), chosen to contrast with the reddish point cloud.
_PALETTE = [
    (0.00, 0.60, 1.00), (0.00, 0.78, 0.40), (0.65, 0.20, 0.90), (0.00, 0.80, 0.85),
    (0.15, 0.35, 0.95), (1.00, 0.85, 0.00), (0.95, 0.10, 0.55), (0.10, 0.50, 0.25),
    (0.40, 0.85, 0.94), (0.55, 0.45, 0.95),
]


def load_ped_tracks(traj_dir: Path, seq: str, ped_filter: Optional[str]) -> List[dict]:
    """Load GOLD tracks for one sequence as time arrays + anchor size + per-frame world velocity."""
    tracks = []
    for f in sorted(traj_dir.glob(f"{seq}_*.json")):
        if f.name.startswith("_"):
            continue
        doc = json.loads(f.read_text())
        pid = doc["pedestrian_id"]
        if ped_filter and not pid.startswith(ped_filter):
            continue
        frames = doc["frames"]
        ts = np.array([parse_zod_ts(fr["timestamp"]) for fr in frames])
        pos = np.array([fr["position_world"] for fr in frames])
        vel = np.gradient(pos, axis=0) if len(pos) > 1 else np.zeros_like(pos)  # world m/frame
        tracks.append({
            "id": pid[:8], "ts": ts, "pos": pos, "vel": vel,
            "obs": np.array([fr["in_observation"] for fr in frames]),
            "size_lwh": np.array(doc["anchor"]["box_size_lwh"], dtype=float),
        })
    return tracks


def _state_at(track: dict, ts: float, tol: float) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
    """(world position, world velocity, in_observation) nearest `ts`, or None if outside tol."""
    i = int(np.argmin(np.abs(track["ts"] - ts)))
    if abs(track["ts"][i] - ts) > tol:
        return None
    return track["pos"][i], track["vel"][i], bool(track["obs"][i])


def _box_corners_lidar(center_l: np.ndarray, heading_l: np.ndarray, size_lwh: np.ndarray) -> np.ndarray:
    """8 corners (8,3) of an oriented box in the LiDAR frame from centre + ground heading + l/w/h."""
    l, w, h = size_lwh
    fwd = heading_l.copy(); fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-3 else np.array([1.0, 0.0, 0.0])     # fall back to +x if ~stationary
    left = np.array([-fwd[1], fwd[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    R = np.stack([fwd, left, up], axis=1)                        # box-local -> lidar
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    local = signs * (np.array([l, w, h]) / 2.0)
    return center_l + local @ R.T


# Edge index pairs of a cuboid given the corner ordering in `_box_corners_lidar`.
_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
              (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def _lineset(corners: np.ndarray, color) -> "object":
    """Open3D LineSet wireframe for a set of cuboid corners in one colour."""
    import open3d as o3d
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(np.array(_BOX_EDGES))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(_BOX_EDGES), 1)))
    return ls


def _box_holding(boxes, u: float, v: float):
    """Smallest detector person box containing pixel (u, v), or None."""
    holding = [b for b in boxes if b.xyxy[0] <= u <= b.xyxy[2] and b.xyxy[1] <= v <= b.xyxy[3]]
    if not holding:
        return None
    return min(holding, key=lambda b: (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]))


def _point_colors(intensity: np.ndarray, cmap_name: str, floor: float = 0.3) -> np.ndarray:
    """Map LiDAR log-intensity to RGB (devkit scheme), readable on a white background.

    `floor` lifts the low end off the colormap's white tail so faint points stay visible against
    the white background (mirrors the reddish look of the official ZOD clips).
    """
    import matplotlib
    v = np.log1p(np.clip(intensity, 0, None))
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    v = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    v = floor + (1.0 - floor) * v
    return matplotlib.colormaps[cmap_name](v)[:, :3]


def _draw_camera_panel(img: np.ndarray, boxes, active: List[tuple], T_lw: np.ndarray,
                       calib: dict, colors: dict) -> None:
    """Draw 2D detector boxes + projected track centroids onto a full-res camera image, in place.

    `active` is a list of (track, world_position, observed); each tracked centroid is projected
    world -> LiDAR(t) -> image and drawn, highlighting the detector box it falls into. Boxes no
    pedestrian claimed are drawn faint grey for context.
    """
    claimed = set()
    for track, pos_w, observed in active:
        p_l = (T_lw @ np.append(pos_w, 1.0))[:3]
        uv, valid = project_lidar_to_image(p_l.reshape(1, 3), calib)
        if not valid[0]:
            continue
        u, v = int(uv[0][0]), int(uv[0][1])
        bgr = tuple(int(c * 255) for c in colors[track["id"]][::-1])
        box = _box_holding(boxes, u, v) if observed else None    # the 2D measurement, if matched
        if box is not None:
            claimed.add(id(box))
            x1, y1, x2, y2 = box.xyxy.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)
        cv2.circle(img, (u, v), 14, (255, 255, 255), 4)          # white halo for contrast
        cv2.circle(img, (u, v), 14, bgr, -1 if observed else 3)  # filled when observed, ring when coasting
        label = track["id"] if observed else f"{track['id']} coast"
        cv2.putText(img, label, (u + 18, v), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, label, (u + 18, v), cv2.FONT_HERSHEY_SIMPLEX, 1.3, bgr, 2, cv2.LINE_AA)
    for b in boxes:                                              # other persons -> faint grey
        if id(b) in claimed:
            continue
        x1, y1, x2, y2 = b.xyxy.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 2)


def _render_cloud_panel(renderer, mats: dict, ss: int, rwidth: int, rheight: int,
                        pts_l: np.ndarray, intensity: np.ndarray, cmap: str,
                        boxes3d: List[tuple], look_v, eye_v, up) -> np.ndarray:
    """Render the 3D point-cloud panel (cloud + oriented boxes) to a BGR (rheight, rwidth) image."""
    import open3d as o3d
    renderer.scene.clear_geometry()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_l)
    pcd.colors = o3d.utility.Vector3dVector(_point_colors(intensity, cmap))
    renderer.scene.add_geometry("cloud", pcd, mats["pcd"])
    for bid, corners, col, observed in boxes3d:
        renderer.scene.add_geometry(f"box_{bid}", _lineset(corners, col),
                                    mats["line"] if observed else mats["coast"])
    renderer.setup_camera(60.0, np.array(look_v, float), np.array(eye_v, float), np.array(up, float))
    panel = cv2.cvtColor(np.asarray(renderer.render_to_image()), cv2.COLOR_RGB2BGR)  # supersampled
    if ss > 1:
        panel = cv2.resize(panel, (rwidth, rheight), interpolation=cv2.INTER_AREA)   # SSAA downscale
    return panel


def render_video(seq: str, layout: str, traj_dir: Path, out_path: Path, window_s: float, fps: int,
                 detector, *, scale: float, rwidth: int, rheight: int, eye, look, up,
                 point_size: float, cmap: str, max_points: int, match_tol: float,
                 ped_filter: Optional[str], follow_dist: float, follow_h: float,
                 supersample: int) -> int:
    """Render the demo MP4 for one sequence in the chosen layout. Returns frames written."""
    seq_dir = SEQ_DIR / seq
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])
    keyframe_ts = parse_zod_ts(json.loads((seq_dir / "info.json").read_text())["keyframe_time"])

    tracks = load_ped_tracks(traj_dir, seq, ped_filter)
    if not tracks:
        raise SystemExit(f"No GOLD trajectories for seq {seq} ({ped_filter=}) in {traj_dir}")
    colors = {t["id"]: np.array(_PALETTE[i % len(_PALETTE)]) for i, t in enumerate(tracks)}

    imgs = sorted((seq_dir / "camera_front_blur").glob("*.jpg"))
    img_ts = np.array([parse_zod_ts(p.stem.rsplit("_", 1)[-1]) for p in imgs])

    # Output-frame schedule: (ts, scan_path | None, image_path). `split` is LiDAR-scan driven (one 3D
    # render per scan, nearest camera image alongside); `camera` is camera-frame driven (smoother).
    items: List[tuple] = []
    if layout == "split":
        scans = sorted((seq_dir / "lidar_velodyne").glob("*.npy"))
        scan_ts = np.array([lidar_ts_from_filename(p.name) for p in scans])
        sel = np.where(np.abs(scan_ts - keyframe_ts) <= window_s / 2.0)[0]
        if len(sel) == 0:
            raise SystemExit(f"No LiDAR scans within {window_s}s of the keyframe for seq {seq}")
        for i in sel:
            j = int(np.argmin(np.abs(img_ts - scan_ts[i])))
            items.append((scan_ts[i], scans[i], imgs[j]))
    else:  # camera
        sel = np.where(np.abs(img_ts - keyframe_ts) <= window_s / 2.0)[0]
        if len(sel) == 0:
            raise SystemExit(f"No camera frames within {window_s}s of the keyframe for seq {seq}")
        items = [(img_ts[i], None, imgs[i]) for i in sel]

    # The 3D panel uses Open3D's offscreen (Filament) renderer, which does not anti-alias points, so
    # render it at `supersample`x resolution and area-downscale: cheap SSAA that turns the jagged/grainy
    # point splats into smooth dots. Built only for `split` so `camera` stays Open3D-free.
    ss = max(1, int(supersample))
    renderer, mats = None, {}
    if layout == "split":
        from open3d.visualization import rendering
        renderer = rendering.OffscreenRenderer(rwidth * ss, rheight * ss)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.scene.set_sun_light([0.5, 0.5, -1.0], [1, 1, 1], 60000)
        mats["pcd"] = rendering.MaterialRecord(); mats["pcd"].shader = "defaultUnlit"; mats["pcd"].point_size = point_size * ss
        mats["line"] = rendering.MaterialRecord(); mats["line"].shader = "unlitLine"; mats["line"].line_width = 8.0 * ss
        mats["coast"] = rendering.MaterialRecord(); mats["coast"].shader = "unlitLine"; mats["coast"].line_width = 3.0 * ss

    rng = np.random.default_rng(0)
    det_cache: dict = {}                        # 2D detections cached per image (scans can share one)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    for ts, scan, img_path in items:
        T_lw = np.linalg.inv(get_T_world_lidar(em, lidar_ext, ts))
        active = []                              # (track, world_pos, world_vel, observed) present at ts
        for t in tracks:
            st = _state_at(t, ts, match_tol)
            if st is not None:
                active.append((t, st[0], st[1], st[2]))

        # --- camera panel (full resolution, both layouts) ---
        if img_path not in det_cache:
            det_cache[img_path] = detector(img_path)
        boxes = det_cache[img_path]
        cam = cv2.imread(str(img_path))
        _draw_camera_panel(cam, boxes, [(t, pos_w, obs) for t, pos_w, _vel, obs in active], T_lw, calib, colors)
        ch, cw = cam.shape[:2]

        if layout == "split":
            raw = load_xyzi(scan)
            if len(raw) > max_points:
                raw = raw[rng.choice(len(raw), max_points, replace=False)]
            pts_l, intensity = raw[:, :3].astype(np.float64), raw[:, 3]

            boxes3d, centers = [], []
            for track, pos_w, vel_w, observed in active:
                center_l = (T_lw @ np.append(pos_w, 1.0))[:3]
                corners = _box_corners_lidar(center_l, T_lw[:3, :3] @ vel_w, track["size_lwh"])
                col = colors[track["id"]] if observed else colors[track["id"]] * 0.45 + 0.4
                boxes3d.append((track["id"], corners, col, observed))
                centers.append(center_l)

            # Fixed forward view, or auto-aim at the focus pedestrian (--ped) so its box stays framed.
            look_v, eye_v = np.array(look, float), np.array(eye, float)
            if ped_filter and centers:
                tgt = np.mean(centers, axis=0)
                horiz = tgt[:2] / (np.linalg.norm(tgt[:2]) + 1e-6)
                look_v = np.array([tgt[0], tgt[1], 0.5])
                eye_v = np.array([tgt[0] - horiz[0] * follow_dist, tgt[1] - horiz[1] * follow_dist, follow_h])

            right = _render_cloud_panel(renderer, mats, ss, rwidth, rheight,
                                        pts_l, intensity, cmap, boxes3d, look_v, eye_v, up)
            left = cv2.resize(cam, (int(cw * rheight / ch), rheight))
            frame = np.hstack([left, right])
        else:  # camera
            frame = cv2.resize(cam, (int(cw * scale), int(ch * scale)))

        dt = ts - keyframe_ts
        fs = max(0.6, frame.shape[0] / 600.0)
        suffix = "tracked in 3D" if layout == "split" else "tracked"
        msg = f"seq {seq}   t={dt:+.1f}s   {len(active)} GOLD ped(s) {suffix}"
        cv2.putText(frame, msg, (16, int(38 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), max(1, int(2 * fs)), cv2.LINE_AA)
        if writer is None:
            h, w = frame.shape[:2]
            writer = FFmpegWriter(out_path, fps, (w, h))
        writer.write(frame)

    writer.release()
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence id, e.g. 000007")
    ap.add_argument("--layout", choices=["split", "camera"], default="split",
                    help="split = camera|3D side-by-side (default); camera = full camera frame only")
    ap.add_argument("--ped", default=None, help="render only this pedestrian (uuid prefix)")
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--out", type=Path, default=None, help="output mp4 (default: review/{zod|cam}_{seq}.mp4)")
    ap.add_argument("--window", type=float, default=8.0, help="seconds centred on the keyframe")
    ap.add_argument("--fps", type=int, default=10, help="output video frame rate")
    ap.add_argument("--scale", type=float, default=0.5, help="camera layout: downscale factor for the 3848x2168 frame")
    ap.add_argument("--rwidth", type=int, default=960, help="split layout: 3D panel width (px)")
    ap.add_argument("--rheight", type=int, default=540, help="split layout: 3D panel / output height (px)")
    # ZOD's Velodyne frame is +Y forward, X lateral, Z up — so the default forward view looks
    # DOWN +Y (behind & above the ego, gazing ahead), matching the official ZOD sample clips.
    # Flat-ish and well behind the ego so the road recedes naturally into the distance.
    ap.add_argument("--eye", nargs=3, type=float, default=[0.0, -20.0, 10.0], help="camera eye (LiDAR frame)")
    ap.add_argument("--look", nargs=3, type=float, default=[0.0, 30.0, 0.0], help="camera look-at (LiDAR frame)")
    ap.add_argument("--up", nargs=3, type=float, default=[0.0, 0.0, 1.0], help="camera up vector")
    ap.add_argument("--follow-dist", type=float, default=11.0, help="--ped auto-aim: eye distance from ped (m)")
    ap.add_argument("--follow-height", type=float, default=7.0, help="--ped auto-aim: eye height (m)")
    ap.add_argument("--supersample", type=int, default=2, help="SSAA factor for the 3D panel (1=off)")
    ap.add_argument("--point-size", type=float, default=2.5, help="LiDAR point size (px)")
    ap.add_argument("--cmap", default="Reds", help="matplotlib colormap for log-intensity")
    ap.add_argument("--max-points", type=int, default=250000, help="points kept per scan")
    ap.add_argument("--match-tol", type=float, default=0.08, help="max |frame-track| time gap to draw a ped (s)")
    ap.add_argument("--model", default="yolo11x.pt", help="2D detector weights for the camera-panel boxes")
    ap.add_argument("--conf", type=float, default=0.1, help="detector confidence threshold")
    ap.add_argument("--imgsz", type=int, default=2560, help="detector inference image size")
    args = ap.parse_args()

    detector = make_detector(args.model, conf=args.conf, imgsz=args.imgsz)
    tag = f"{args.seq}_{args.ped}" if args.ped else args.seq
    prefix = "zod" if args.layout == "split" else "cam"
    out = args.out or DEFAULT_REVIEW_DIR / f"{prefix}_{tag}.mp4"
    n = render_video(args.seq, args.layout, args.traj_dir, out, args.window, args.fps, detector,
                     scale=args.scale, rwidth=args.rwidth, rheight=args.rheight, eye=args.eye,
                     look=args.look, up=args.up, point_size=args.point_size, cmap=args.cmap,
                     max_points=args.max_points, match_tol=args.match_tol, ped_filter=args.ped,
                     follow_dist=args.follow_dist, follow_h=args.follow_height, supersample=args.supersample)
    print(f"wrote {n} frames → {out}")


if __name__ == "__main__":
    main()
