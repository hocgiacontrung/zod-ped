"""Overlay one pedestrian's tracked 3D position back onto the camera images (identity check).

The BEV view (scripts/viz_trajectories.py) shows track SHAPE; this shows track IDENTITY. For one
pedestrian it projects the smoothed world position into the nearest camera image at each frame and
draws a marker, then tiles evenly-sampled frames into a montage. If the marker stays on the SAME
person across the clip the track is correct; if it jumps onto another person you've found an ID error
(the known independent-per-ped-tracking limitation). Coasted frames (detector saw nothing) are
labelled so you can tell measured from predicted.

Projection: world → LiDAR frame at the scan time [ inv(T_world_lidar(t)) ] → image (Kannala),
reusing the exact transforms Step 1 tracked with.

Usage:
    # auto-pick the best-tracked mover in a sequence:
    python scripts/viz_pedestrian_overlay.py --seq 000007
    # a specific pedestrian (8-char prefix is enough):
    python scripts/viz_pedestrian_overlay.py --seq 000007 --ped 2A2ACAF4
    python scripts/viz_pedestrian_overlay.py --seq 000007 --ped 2A2ACAF4 --crop 240 380 --n 16
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.keyframe import parse_zod_ts  # noqa: E402
from utils.ego_motion import get_T_world_lidar, load_ego_motion  # noqa: E402
from utils.projection import load_calibration, project_lidar_to_image  # noqa: E402

SEQ_DIR = ROOT / "data" / "raw" / "sequences"
DEFAULT_TRAJ = ROOT / "data" / "processed" / "trajectories"


def _pick_pedestrian(traj_dir: Path, seq: str, ped: str | None) -> dict:
    """Load the requested pedestrian's trajectory, or auto-pick the best-tracked mover."""
    cands = sorted(glob.glob(str(traj_dir / f"{seq}_*.json")))
    cands = [c for c in cands if "_run_report" not in c]
    if not cands:
        raise SystemExit(f"No trajectories for seq {seq} in {traj_dir}")
    docs = [json.loads(Path(c).read_text()) for c in cands]
    if ped:
        hit = [d for d in docs if d["pedestrian_id"].startswith(ped) or d["pedestrian_id"][:8] == ped.upper()]
        if not hit:
            ids = ", ".join(d["pedestrian_id"][:8] for d in docs)
            raise SystemExit(f"ped {ped} not found in {seq}. Available: {ids}")
        return hit[0]
    # auto: prefer a well-observed mover (interesting + verifiable)
    def score(d):
        pos = np.array([f["position_world"] for f in d["frames"]])
        path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        return (d["stats"]["observed_fraction"] or 0) * min(path, 10)
    best = max(docs, key=score)
    print(f"auto-picked ped {best['pedestrian_id'][:8]} "
          f"(obs={best['stats']['observed_fraction']}, of {len(docs)} peds in {seq})")
    return best


def _project_world_point(p_world: np.ndarray, em: dict, lidar_ext: np.ndarray,
                         calib: dict, scan_ts: float):
    """World point → (pixel (u,v), camera depth m), or (None, None) if behind/outside the camera."""
    T = get_T_world_lidar(em, lidar_ext, scan_ts)
    p_lidar = (np.linalg.inv(T) @ np.append(p_world, 1.0))[:3]
    uvz, valid = project_lidar_to_image(p_lidar.reshape(1, 3), calib, return_depth=True)
    if not valid[0]:
        return None, None
    return uvz[0, :2], float(uvz[0, 2])


def _adaptive_half_crop(depth: float, calib: dict, fill: float = 0.5) -> tuple:
    """Crop half-(w,h) px so a ~1.8 m pedestrian fills `fill` of the cell height at this depth.

    Distant peds project small, so we zoom in (smaller crop); near peds zoom out. Keeps the person
    a roughly constant on-screen size across frames, which is what makes the identity check legible.
    """
    fy = float(np.array(calib["FC"]["intrinsics"])[1, 1])
    person_px = fy * 1.8 / max(depth, 1.0)
    half_h = int(np.clip(person_px / (2 * fill), 70, 700))
    return int(half_h * 0.7), half_h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence id, e.g. 000007")
    ap.add_argument("--ped", default=None, help="pedestrian id prefix (default: auto-pick best mover)")
    ap.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--camera", default="camera_front_blur")
    ap.add_argument("--n", type=int, default=12, help="montage cells (frames sampled across the track)")
    ap.add_argument("--crop", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="fixed crop half-size px (default: depth-adaptive so the person stays a constant size)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    doc = _pick_pedestrian(args.traj_dir, args.seq, args.ped)
    ped8 = doc["pedestrian_id"][:8]
    seq_dir = SEQ_DIR / args.seq
    em = load_ego_motion(seq_dir / "ego_motion.json")
    calib = load_calibration(seq_dir / "calibration.json")
    lidar_ext = np.array(calib["FC"]["lidar_extrinsics"])

    imgs = sorted((seq_dir / args.camera).glob("*.jpg"))
    img_ts = np.array([parse_zod_ts(f.stem.rsplit("_", 1)[-1]) for f in imgs])

    frames = doc["frames"]
    idxs = np.linspace(0, len(frames) - 1, min(args.n, len(frames))).round().astype(int)

    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.array(axes).reshape(-1)

    for ax, fi in zip(axes, idxs):
        fr = frames[fi]
        scan_ts = parse_zod_ts(fr["timestamp"])
        j = int(np.argmin(np.abs(img_ts - scan_ts)))
        img = mpimg.imread(imgs[j])
        H, W = img.shape[:2]
        uv, depth = _project_world_point(np.array(fr["position_world"]), em, lidar_ext, calib, scan_ts)
        cw, ch = args.crop if args.crop else _adaptive_half_crop(depth or 30.0, calib)

        obs = fr["in_observation"]
        color = "lime" if obs else "orange"
        tag = fr["tracking_method"][:3] if obs else "coast"
        if uv is None:
            ax.imshow(img[::8, ::8]); ax.set_title(f"f{fi}: off-image ({tag})", fontsize=8)
        else:
            u, v = int(uv[0]), int(uv[1])
            x0, x1 = max(0, u - cw), min(W, u + cw)
            y0, y1 = max(0, v - ch), min(H, v + ch)
            ax.imshow(img[y0:y1, x0:x1])
            ax.scatter([u - x0], [v - y0], s=90, facecolors="none", edgecolors=color, linewidths=2)
            ax.scatter([u - x0], [v - y0], s=4, c=color)
            ax.set_title(f"f{fi}  {fr['timestamp'][14:22]}  ({tag})", fontsize=8, color="black")
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")

    fig.suptitle(f"seq {args.seq} — ped {ped8}  "
                 f"(green=detector-observed, orange=coasted; marker = tracked 3D position)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or args.traj_dir / f"overlay_{args.seq}_{ped8}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"montage → {out}")


if __name__ == "__main__":
    main()
