"""
Step 1 – Window Generation.

Reads:  data/pedestrian_sequences.json
        data/raw/sequences/XXXXXX/{annotations,lidar_velodyne,calibration.json,
                                    ego_motion.json,info.json}
Writes: data/processed/candidate_windows.json

Structural/data-availability filters only (position-based filters are deferred to
Step 2.5, after trajectories are tracked):
  1. Skip 2D-only pedestrians (no location_3d → can't seed the tracker).
  2. Skip pedestrians with no LiDAR detection at keyframe (≥1 point in 3D box
     within 55ms; needed as tracking seed).
  3. Skip windows with fewer than 3 LiDAR scans (insufficient data to track).

distance_to_ego_m and distance_to_road_m are recorded as context metadata using
the keyframe position, but are NOT used to filter. They will be recomputed per
window at Step 2.5 using tracked positions.

Output: list of dicts; one entry per (pedestrian, window) pair.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon

# --- project root on sys.path so src/ imports work -------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.projection import get_T_cam_lidar, load_calibration, project_lidar_to_image  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (mirrors configs/dataset_schema_v0.1.yaml)
# ---------------------------------------------------------------------------
MAX_LIDAR_GAP_S      = 0.055   # 55ms — max gap between keyframe and nearest LiDAR scan
MIN_LIDAR_IN_BOX     = 1       # at keyframe — minimum points in 3D box to seed tracker
WINDOW_S             = 0.5
STRIDE_S             = 0.25
MIN_LIDAR_IN_WINDOW  = 3       # minimum LiDAR scans per window

# NOTE: MAX_DIST_EGO_M and MAX_DIST_ROAD_M are intentionally absent here.
# They are applied at Step 2.5 using per-window tracked positions.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_zod_ts(ts_str: str) -> float:
    return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def lidar_ts_from_filename(filename: str) -> float:
    stem = Path(filename).stem
    ts_part = stem.split("_", 2)[2]
    return parse_zod_ts(ts_part)


def lidar_files_sorted(lidar_dir: Path) -> list[Path]:
    return sorted(lidar_dir.glob("*.npy"))


def nearest_lidar(files: list[Path], target_ts: float) -> tuple[Path | None, float]:
    if not files:
        return None, float("inf")
    timestamps = np.array([lidar_ts_from_filename(f.name) for f in files])
    idx = np.argmin(np.abs(timestamps - target_ts))
    return files[idx], abs(timestamps[idx] - target_ts)


def lidar_in_window(files: list[Path], window_start: float, window_end: float) -> list[Path]:
    return [f for f in files
            if window_start <= lidar_ts_from_filename(f.name) <= window_end]


def points_in_box(pts_xyz: np.ndarray, loc: list, qw: float, qx: float,
                  qy: float, qz: float, length: float, width: float,
                  height: float) -> int:
    centre = np.array(loc)
    rot = Rotation.from_quat([qx, qy, qz, qw])
    local = rot.inv().apply(pts_xyz - centre)
    return int(((np.abs(local[:, 0]) <= length / 2) &
                (np.abs(local[:, 1]) <= width  / 2) &
                (np.abs(local[:, 2]) <= height / 2)).sum())


def distance_to_road_m(loc_lidar: list, road_polygons: list,
                        calib: dict) -> float:
    """Approximate 3D distance from pedestrian centroid to nearest road polygon edge.

    Projects the 3D point to image space, computes pixel distance to the road
    polygon (0 if inside), then scales by z_cam / fx for a metres estimate.
    Returns float('inf') if the point is behind the camera.
    """
    pts = np.array([loc_lidar])
    uv_visible, valid = project_lidar_to_image(pts, calib)
    if not valid[0]:
        return float("inf")

    u, v = float(uv_visible[0, 0]), float(uv_visible[0, 1])

    # z_cam at the pedestrian (needed to convert pixel → metres)
    T = get_T_cam_lidar(calib)
    p_cam = T @ np.append(loc_lidar, 1.0)
    z_cam = float(p_cam[2])
    fx = float(np.array(calib["FC"]["intrinsics"])[0, 0])

    pt_2d = Point(u, v)
    polys = [Polygon(entry["geometry"]["coordinates"][0]) for entry in road_polygons]

    min_px_dist = min(poly.distance(pt_2d) for poly in polys)
    return min_px_dist * z_cam / fx


def ego_speed_at(ego_motion: dict, timestamp: float) -> float:
    """Interpolate ego speed (m/s) from ego_motion velocities."""
    ts = np.array(ego_motion["timestamps"])
    idx = np.searchsorted(ts, timestamp)
    idx = int(np.clip(idx, 0, len(ts) - 1))
    vx, vy, vz = ego_motion["velocities"][idx]
    return float(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(seq_id: str, raw_dir: Path) -> tuple[list, dict]:
    """Return (candidate_records, stats_dict)."""

    seq_dir = raw_dir / seq_id
    candidates: list[dict] = []
    stats = {
        "seq_id": seq_id,
        "peds_total": 0,
        "peds_2d_only": 0,
        "filtered_no_lidar_detection": 0,
        "peds_passing_keyframe": 0,
        "windows_generated": 0,
        "windows_filtered_min_scans": 0,
        "candidates": 0,
    }

    # --- load sequence data ------------------------------------------------
    info        = json.loads((seq_dir / "info.json").read_text())
    ego_motion  = json.loads((seq_dir / "ego_motion.json").read_text())
    detections  = json.loads((seq_dir / "annotations" / "object_detection.json").read_text())
    road_polys  = json.loads((seq_dir / "annotations" / "ego_road.json").read_text())
    calib       = load_calibration(seq_dir / "calibration.json")

    keyframe_ts = parse_zod_ts(info["keyframe_time"])
    start_ts    = parse_zod_ts(info["start_time"])
    end_ts      = parse_zod_ts(info["end_time"])
    duration_s  = end_ts - start_ts

    lidar_dir   = seq_dir / "lidar_velodyne"
    lidar_files = lidar_files_sorted(lidar_dir)

    # keyframe LiDAR scan
    kf_lidar_file, kf_gap_s = nearest_lidar(lidar_files, keyframe_ts)
    kf_lidar_valid = kf_gap_s <= MAX_LIDAR_GAP_S

    if kf_lidar_file and kf_lidar_valid:
        kf_cloud = np.load(kf_lidar_file)
        kf_pts   = np.stack([kf_cloud["x"], kf_cloud["y"], kf_cloud["z"]], axis=1)
    else:
        kf_pts = np.empty((0, 3))

    # --- pedestrian-level filters ------------------------------------------
    peds = [d for d in detections if d["properties"]["class"] == "Pedestrian"]
    stats["peds_total"] = len(peds)

    passing_peds = []
    for ped in peds:
        pr = ped["properties"]

        if "location_3d" not in pr:
            stats["peds_2d_only"] += 1
            continue

        loc      = pr["location_3d"]["coordinates"]
        dist_ego = float(np.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2))
        dist_road = distance_to_road_m(loc, road_polys, calib)

        if kf_pts.size:
            n_box = points_in_box(
                kf_pts, loc,
                pr["orientation_3d_qw"], pr["orientation_3d_qx"],
                pr["orientation_3d_qy"], pr["orientation_3d_qz"],
                pr["size_3d_length"], pr["size_3d_width"], pr["size_3d_height"],
            )
        else:
            n_box = 0

        if n_box < MIN_LIDAR_IN_BOX:
            stats["filtered_no_lidar_detection"] += 1
            continue

        passing_peds.append({
            "ped": ped,
            "dist_ego_m": dist_ego,
            "dist_road_m": dist_road,
            "n_lidar_box": n_box,
            "kf_lidar_gap_ms": kf_gap_s * 1000,
        })

    stats["peds_passing_keyframe"] = len(passing_peds)

    # --- window generation -------------------------------------------------
    window_starts = np.arange(0.0, duration_s - WINDOW_S + 1e-6, STRIDE_S)

    for ped_info in passing_peds:
        ped = ped_info["ped"]
        pr  = ped["properties"]

        for ws_offset in window_starts:
            stats["windows_generated"] += 1

            w_start_ts = start_ts + ws_offset
            w_end_ts   = w_start_ts + WINDOW_S

            w_lidar = lidar_in_window(lidar_files, w_start_ts, w_end_ts)
            if len(w_lidar) < MIN_LIDAR_IN_WINDOW:
                stats["windows_filtered_min_scans"] += 1
                continue

            w_start_str = datetime.datetime.fromtimestamp(
                w_start_ts, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            w_end_str = datetime.datetime.fromtimestamp(
                w_end_ts, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

            w_start_ms = int(round(ws_offset * 1000))
            sample_id = f"{seq_id}_{pr['annotation_uuid']}_{w_start_ms:06d}"

            # camera frames in window
            cam_dir = seq_dir / "camera_front_blur"
            cam_frames = []
            if cam_dir.exists():
                for jpg in sorted(cam_dir.glob("*.jpg")):
                    jpg_ts_str = jpg.stem.split("_", 2)[2]
                    jpg_ts = parse_zod_ts(jpg_ts_str)
                    if w_start_ts <= jpg_ts <= w_end_ts:
                        cam_frames.append({
                            "timestamp": jpg_ts_str,
                            "path": str(jpg.relative_to(raw_dir.parent)),
                        })

            candidates.append({
                "sample_id": sample_id,
                "sequence_id": seq_id,
                "pedestrian_id": pr["annotation_uuid"],
                "window_start_timestamp": w_start_str,
                "window_end_timestamp":   w_end_str,
                "keyframe_timestamp":     info["keyframe_time"],
                "context": {
                    "distance_to_ego_m":  round(ped_info["dist_ego_m"], 2),
                    "distance_to_road_m": round(ped_info["dist_road_m"], 2),
                    "occlusion":          pr.get("occlusion_ratio"),
                    "ego_speed_ms":       round(ego_speed_at(ego_motion, w_start_ts), 2),
                },
                "multimodal": {
                    "lidar_scans": [
                        {
                            "timestamp": f.stem.split("_", 2)[2],
                            "path": str(f.relative_to(raw_dir.parent)),
                        }
                        for f in w_lidar
                    ],
                    "camera_frames": cam_frames,
                },
            })
            stats["candidates"] += 1

    return candidates, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences-json",
        default=str(ROOT / "data" / "pedestrian_sequences.json"),
        help="Path to pedestrian_sequences.json",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw" / "sequences"),
        help="Root of raw ZOD sequence data",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "candidate_windows.json"),
        help="Output path for candidate_windows.json",
    )
    parser.add_argument(
        "--seq-id",
        default=None,
        help="Process a single sequence ID (for debugging)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.sequences_json) as f:
        all_entries = json.load(f)

    # restrict to sequences with LiDAR on disk
    working_set = [
        e for e in all_entries
        if (raw_dir / e["seq_id"] / "lidar_velodyne").exists()
        and any((raw_dir / e["seq_id"] / "lidar_velodyne").iterdir())
    ]

    if args.seq_id:
        working_set = [e for e in working_set if e["seq_id"] == args.seq_id]
        if not working_set:
            print(f"Sequence {args.seq_id} not found in working set (no LiDAR on disk).")
            sys.exit(1)

    print(f"Working set: {len(working_set)} sequences")

    all_candidates: list[dict] = []
    all_stats: list[dict] = []

    for entry in working_set:
        seq_id = entry["seq_id"]
        print(f"  {seq_id} ...", end=" ", flush=True)
        try:
            candidates, stats = process_sequence(seq_id, raw_dir)
            all_candidates.extend(candidates)
            all_stats.append(stats)
            print(
                f"{stats['peds_passing_keyframe']}/{stats['peds_total']} peds pass  "
                f"→ {stats['candidates']} windows"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            raise

    # write output
    output = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "thresholds": {
            "max_lidar_gap_s":          MAX_LIDAR_GAP_S,
            "min_lidar_in_box":         MIN_LIDAR_IN_BOX,
            "window_s":                 WINDOW_S,
            "stride_s":                 STRIDE_S,
            "min_lidar_in_window":      MIN_LIDAR_IN_WINDOW,
        },
        "summary": {
            "sequences_processed":        len(all_stats),
            "total_peds":                 sum(s["peds_total"] for s in all_stats),
            "peds_2d_only":               sum(s["peds_2d_only"] for s in all_stats),
            "filtered_no_lidar":          sum(s["filtered_no_lidar_detection"] for s in all_stats),
            "peds_passing_keyframe":      sum(s["peds_passing_keyframe"] for s in all_stats),
            "windows_generated":          sum(s["windows_generated"] for s in all_stats),
            "windows_filtered_min_scans": sum(s["windows_filtered_min_scans"] for s in all_stats),
            "total_candidates":           len(all_candidates),
        },
        "per_sequence": all_stats,
        "candidates": all_candidates,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print()
    print("=" * 60)
    s = output["summary"]
    print(f"Sequences processed   : {s['sequences_processed']}")
    print(f"Total pedestrians     : {s['total_peds']}")
    print(f"  2D-only (skipped)   : {s['peds_2d_only']}")
    print(f"  No LiDAR detection  : {s['filtered_no_lidar']}")
    print(f"  Passing keyframe    : {s['peds_passing_keyframe']}")
    print(f"  (distance_to_ego / distance_to_road recorded as metadata; filtered at Step 2.5)")
    print(f"Windows generated     : {s['windows_generated']}")
    print(f"  Filtered <{MIN_LIDAR_IN_WINDOW} scans  : {s['windows_filtered_min_scans']}")
    print(f"Candidate windows     : {s['total_candidates']}")
    print(f"\nOutput → {out_path}")


if __name__ == "__main__":
    main()
