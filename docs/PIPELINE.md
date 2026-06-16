# Pipeline Design & Schema

> These are working decisions, not settled doctrine. If you (human or AI, any session) see a
> better approach, push back — alternatives and disagreement are welcome. State the trade-off and
> make the case; don't silently follow the doc if something looks wrong.

## Pipeline Overview

```
[pedestrian_sequences.json — sequences with LiDAR on disk]
        ↓
[Step 1] Window Generation  (scripts/01_filter_sequences.py)
  - Structural/data-availability filters only (see below)
  - Slice each passing pedestrian into overlapping 0.5s windows (stride 0.25s)
  - Record distance_to_ego_m and distance_to_road_m as metadata (not filters)
  - Output: data/processed/candidate_windows.json

        ↓
[Step 2] Trajectory Generation  (scripts/02_generate_trajectories.py)
  - Unit of work: the PEDESTRIAN (one track per keyframe-annotated pedestrian),
    NOT the window. ~1,776 tracks vs 140k windows → does NOT load candidate_windows.json;
    derives the distinct (seq, ped) set and tracks each once over the full 20s clip.
  - Seed: keyframe 3D box from object_detection.json (location_3d is in the LiDAR frame)
  - Compensate FIRST, then associate: lift each scan's points to a single world frame
    [ pose(t) @ T_ego_lidar @ p_lidar ], interpolating the ego pose at scan time (SLERP+lerp),
    so ego motion is never mistaken for pedestrian motion during association.
  - Track: forward + backward from keyframe via GATED CENTROID — fixed keyframe box size,
    gate around the Kalman-predicted position, centroid of enclosed world points = measurement.
  - Smooth: constant-velocity Kalman (online, used for the gate) + RTS backward pass;
    store innovation residual as kalman_confidence.
  - Coast on miss: predict-only when 0 points in the gate (occlusion/range); mark
    in_observation=false; terminate the track after N consecutive misses.
  - Output: per-pedestrian world-frame trajectory →
    data/processed/trajectories/{seq_id}_{pedestrian_id}.json
  - NOTE: position_ego_rel is per-window (relative to ego pose at that window's start),
    so it is NOT produced here — it is added during sample assembly.

        ↓
[Step 3] Proximity Filter  (scripts/03_filter_by_trajectory.py)
  - Now that per-frame positions exist, apply position-based filters per window:
      distance_to_ego_m ≤ 50.0   (at window midpoint, using tracked position)
      distance_to_road_m ≤ 15.0  (at window midpoint, projected to ego_road polygon)
  - Output: data/processed/candidate_windows_filtered.json

        ↓
[Step 4] Intent Labeling  (scripts/04_label_intent.py)
  - Rule-based (~75–80%): trajectory crosses ego road centerline within horizon?
  - Pose estimation (~10–15%): MediaPipe body orientation for ambiguous cases
  - VLM (~5%): Gemini 1.5 Flash (free tier) or Llama 3.2 Vision 11B (local, ~7GB VRAM)
               always provide trajectory coordinates in prompt, not image-only
  - Manual (~5–10%): gold standard validation subset
  - Output: data/annotations/{sample_id}.json + data/annotations/dataset_index.parquet
```

## Windowing Parameters
```yaml
observation_window_s: 0.5     # ~5 camera frames, ~4 LiDAR scans
overlap_ratio: 0.5            # stride = 0.25s
prediction_horizons_s: [1.0, 1.5, 2.0]
```

## Filtering Thresholds

**Step 1 — structural (data availability):**
```yaml
require_3d_annotation: true        # skip 2D-only pedestrians (no location_3d)
require_lidar_detection: true      # ≥1 LiDAR point in 3D box at keyframe; needed to seed tracker
min_lidar_frames_in_window: 3      # out of ~4 expected; skip windows with missing scans
max_occlusion: Heavy               # no occlusion filtering; all included, flagged in metadata
```

**Step 3 — proximity (requires tracked trajectory):**
```yaml
max_distance_to_ego_m: 50.0        # at window midpoint using tracked position
max_distance_to_road_m: 15.0       # at window midpoint, projected to ego_road polygon
```

**Rationale for split:** ZOD has a single keyframe annotation per sequence (center of 20s
clip). Applying distance/road filters at Step 1 would use keyframe position as a proxy for
all 79 windows per pedestrian, which is unreliable for windows far from the keyframe
(pedestrian may have moved ~14m at walking speed). Position-based filters are deferred
until Step 3 when per-frame tracked positions are available.

## Trajectory Tracking Detail
Frame chain per scan: `p_world = pose(t) @ T_ego_lidar @ p_lidar`, where
`T_ego_lidar = calib["FC"]["lidar_extrinsics"]` and `pose(t)` is interpolated from
`ego_motion["poses"]` (T[world←ego], origin = ego at clip start) at the scan timestamp.

1. **Seed** at keyframe (t≈10s): 3D box from annotation; centroid `location_3d` lifted to world.
2. **Compensate first**: lift every candidate scan's points to the world frame *before*
   association, so ego motion is removed and the pedestrian moves smoothly.
3. **Predict** the next position with a constant-velocity Kalman filter.
4. **Gate + measure**: keep world points inside the predicted (fixed-size) box + margin;
   the centroid is the measurement. **Forward** (keyframe→t=20s) then **Backward** (keyframe→t=0s).
5. **Update** Kalman; store innovation residual as `kalman_confidence`. **Coast** (predict-only,
   `in_observation=false`) when the gate is empty; terminate after N consecutive misses.
6. **Smooth**: RTS backward pass over the completed track.

Design choices (debatable — see top of doc): **gated centroid** over 3D-box IoU (pedestrians are
only a handful of points at range — IoU is unstable); **compensate-before-associate** (PIPELINE
originally listed compensation last); **online gate + RTS smooth** rather than smooth-only.
Pedestrian moves <15cm between LiDAR frames (~111ms) → a small gate radius suffices.

## Detectors: validation now, active roles later
No per-frame ground truth beyond the keyframe → off-the-shelf detectors first serve to
cross-check our tracks. **Agreement metric, NOT accuracy** (detectors are themselves weak on
pedestrians at range).
- **3D — PointPillars**: detect on a sample of scans; compare boxes vs our tracked centroids.
- **2D — YOLO + ByteTrack**: project 3D track → camera frame (`projection.py`) for a per-frame
  box; compare vs an independent 2D track.

Later the same detectors can do real work, not just QA: **re-acquire a coasted track** (add-on
inside Step 2), or **detect non-keyframe pedestrians** (unlabeled social context only — no
intent, no identity, so they do NOT grow the labeled set). Both deferred until the core tracker
exists and we've measured how often it loses the pedestrian.

## Known Limitation — keyframe-only annotation
ZOD annotates one keyframe per 20s clip, so a pedestrian present only before/after it is never
annotated and cannot become a labeled sample (no intent, no identity). Out of scope for the
labeled set; state this in the paper.

## Sample Schema (key fields)

```yaml
sample_id: "{sequence_id}_{pedestrian_id}_{window_start_ms}"
sequence_id: str
pedestrian_id: str             # annotation_uuid from object_detection.json

window_start_timestamp: str    # UTC ISO
window_end_timestamp: str
keyframe_timestamp: str        # ZOD annotation frame (not necessarily in this window)

intent:
  label: crossing | not_crossing | uncertain
  crossing_completed: bool     # only meaningful when label = crossing; definition TBD
  confidence: float            # pipeline certainty (independent of label)
  method: rule_based | vlm | manual
  soft_label:
    p_crossing: float
    p_not_crossing: float
    p_uncertain: float

trajectory:
  # Step 2 emits per-PEDESTRIAN world-frame tracks at trajectories/{seq}_{ped}.json.
  # position_ego_rel is added per-window during sample assembly (not by Step 2).
  frames:
    - timestamp: str
      position_world: [x, y, z]
      position_ego_rel: [x, y, z]   # relative to ego pose at window_start; use for training
      in_observation: bool          # false when coasting (gate empty)
      num_lidar_points: int         # points in gate this scan (-1 = coasted, no measurement)
      tracking_lost: bool           # true while coasting past the last good measurement
      kalman_confidence: float      # from innovation residual; 0.0 when coasted
      tracking_method: anchor | forward | backward

multimodal:
  camera_frames: [{timestamp, path}]
  lidar_scans:   [{timestamp, path}]
  radar_path: str

context:
  ego_speed_ms: float
  ego_heading_deg: float
  turn_indicator: none | left | right
  brake_pedal_ratio: float     # PENDING: confirm with supervisor
  distance_to_ego_m: float
  distance_to_road_m: float
  occlusion: null | Light | Heavy

metadata:
  location: str
  collection_date: str
  split: train | val | test    # assigned at sequence level to prevent leakage
  label_confidence_tier: high | medium | low
  is_in_gold_standard: bool
  num_pedestrians_in_scene: int
  is_key_pedestrian: bool
```

## Output Format
- **Per sample**: `data/annotations/{sample_id}.json`
- **Dataset index**: `data/annotations/dataset_index.parquet` (all scalar fields; fast filtering)
- **Raw sensor data**: file paths only, not embedded

## Dataset Scale Estimate
- 130 sequences × ~10–15 valid windows = **1,300–1,950 samples**
- Target crossing ratio: ~20–30% (consistent with JAAD/PIE)
- Split: 70/15/15 train/val/test at sequence level (PENDING supervisor confirmation)
