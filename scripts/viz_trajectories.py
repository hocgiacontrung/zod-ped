"""Visualize + sanity-check Step 1 trajectories (BEV plot per sequence + numeric table).

For each sequence with trajectories in --traj-dir, draws a bird's-eye (world x-y) plot of the ego
path and every pedestrian track (observed segments solid, coasted points faded, anchor starred),
and prints a per-pedestrian table. The key automated signal is MAX_STEP — the largest inter-frame
position jump over OBSERVED frames: a pedestrian can't move metres between ~0.11 s scans, so a large
value flags a false detector association (the linker grabbed another object). Doubles as the manual-
review viewer.

Usage:
    python scripts/viz_trajectories.py --traj-dir data/processed/trajectories_verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.keyframe import parse_zod_ts  # noqa: E402
from utils.ego_motion import interpolate_pose, load_ego_motion  # noqa: E402

SEQ_DIR = ROOT / "data" / "raw" / "sequences"


def _ego_world_xy(seq_id: str, keyframe_ts: float) -> tuple:
    """Return (ego path x-y (T,2), ego x-y at keyframe (2,))."""
    em = load_ego_motion(SEQ_DIR / seq_id / "ego_motion.json")
    path = em["poses"][:, :2, 3]
    ego_kf = interpolate_pose(em, keyframe_ts)[:2, 3]
    return path, ego_kf


def _ped_stats(doc: dict, ego_kf_xy: np.ndarray) -> dict:
    frames = doc["frames"]
    pos = np.array([f["position_world"] for f in frames])
    obs_mask = np.array([f["in_observation"] for f in frames])
    obs = pos[obs_mask]
    steps = np.linalg.norm(np.diff(obs, axis=0), axis=1) if len(obs) > 1 else np.zeros(0)
    anchor_xy = np.array(doc["anchor"]["position_world"][:2])
    return {
        "ped": doc["pedestrian_id"][:8],
        "n": len(frames),
        "obs_frac": doc["stats"]["observed_fraction"],
        "path_len": round(float(steps.sum()), 1),
        "max_step": round(float(steps.max()), 2) if steps.size else 0.0,
        "range_kf": round(float(np.linalg.norm(anchor_xy - ego_kf_xy)), 1),
        "pos": pos, "obs_mask": obs_mask, "anchor_xy": anchor_xy,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-dir", type=Path, default=ROOT / "data" / "processed" / "trajectories_verify")
    ap.add_argument("--jump-warn", type=float, default=1.5, help="max_step (m) above which a track is flagged")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(f for f in args.traj_dir.glob("*.json") if not f.name.startswith("_"))
    by_seq = defaultdict(list)
    for f in files:
        d = json.loads(f.read_text())
        by_seq[d["sequence_id"]].append(d)

    print(f"{'seq':>7} {'ped':>9} {'n':>4} {'obs%':>6} {'path_m':>7} {'maxstep':>8} {'range_kf':>9}  flag")
    print("-" * 72)
    n_flag = 0
    for seq_id, docs in sorted(by_seq.items()):
        kf_ts = parse_zod_ts(docs[0]["keyframe_timestamp"])
        ego_path, ego_kf = _ego_world_xy(seq_id, kf_ts)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(ego_path[:, 0], ego_path[:, 1], color="0.6", lw=1, label="ego path")
        ax.scatter(*ego_kf, c="k", marker="s", s=40, zorder=5, label="ego @ keyframe")

        cmap = plt.cm.tab10
        for k, doc in enumerate(docs):
            s = _ped_stats(doc, ego_kf)
            flag = "JUMP" if s["max_step"] > args.jump_warn else ""
            n_flag += bool(flag)
            print(f"{seq_id:>7} {s['ped']:>9} {s['n']:>4} {s['obs_frac']*100:>5.0f}% "
                  f"{s['path_len']:>7} {s['max_step']:>8} {s['range_kf']:>9}  {flag}")
            c = cmap(k % 10)
            pos, om = s["pos"], s["obs_mask"]
            ax.plot(pos[om, 0], pos[om, 1], "-", color=c, lw=1.5, ms=3, marker="o")
            if (~om).any():
                ax.scatter(pos[~om, 0], pos[~om, 1], color=c, s=8, alpha=0.3)
            ax.scatter(*s["anchor_xy"], color=c, marker="*", s=120, edgecolor="k", zorder=6)

        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        ax.set_title(f"seq {seq_id} — {len(docs)} GOLD peds (★=keyframe anchor, faded=coasted)")
        ax.legend(loc="best", fontsize=8)
        out = args.traj_dir / f"bev_{seq_id}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)

    print("-" * 72)
    print(f"{len(files)} trajectories, {len(by_seq)} sequences. "
          f"{n_flag} flagged (max_step > {args.jump_warn} m). BEV plots → {args.traj_dir}/bev_*.png")


if __name__ == "__main__":
    main()
