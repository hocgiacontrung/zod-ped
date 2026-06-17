# Pipeline Design & Schema

> These are working decisions, not settled doctrine. If you see a
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
  - Unit of work: the PEDESTRIAN (one track per pedestrian), NOT the window.
    Does NOT load candidate_windows.json; derives the (seq, ped) set and tracks each
    once over the full 20s clip.
  - Architecture: DETECTOR-AS-MEASUREMENT + KALMAN/RTS-AS-LINKER.
      * 3D detector (PointPillars / CenterPoint) localizes pedestrians per scan — the
        per-frame box is the measurement (independent of any motion model, so stop/start
        is captured by detection, not invented by the filter).
      * The KF/RTS linker (constant-velocity Kalman + RTS smoother) associates detections
        across scans, coasts through gaps, and smooths — it links boxes into a track but is
        NO LONGER the source of position during normal tracking.
      * 2D detector (YOLO + ByteTrack, SAM masks) runs on the camera stream for appearance
        features (crops, body orientation, occlusion) and as an independent cross-check.
  - Compensate FIRST: lift each scan's points to a single world frame
    [ pose(t) @ T_ego_lidar @ p_lidar ], interpolating the ego pose at scan time (SLERP+lerp),
    so ego motion is never mistaken for pedestrian motion. Shared by detector and linker.
  - Two-tier output (see "Two-tier labels" below):
      * GOLD — keyframe-annotated peds (verified ZOD identity + 3D box). Track is anchored
        on the keyframe box; the detector measurements are validated against it.
      * SILVER — detector-found peds with no keyframe annotation. Seeded from the first
        confident detection; flagged label_confidence_tier=low / is_in_gold_standard=false.
  - Coast on miss: predict-only when the detector returns nothing in the gate
    (occlusion/range); mark in_observation=false; terminate after N consecutive misses.
  - Output: per-pedestrian world-frame trajectory →
    data/processed/trajectories/{seq_id}_{pedestrian_id}.json
  - NOTE: position_ego_rel is per-window (relative to ego pose at that window's start),
    so it is NOT produced here — it is added during sample assembly.
  - Bring-up gates (run BEFORE building the MOT — see "Step 2 Bring-up" below):
      (1) world-frame transform validation (static object stays fixed in world frame);
      (2) detector recall vs the 1,776 keyframe boxes (go/no-go for detector-as-measurement).

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

1. **Detect** pedestrians per scan with the 3D detector (PointPillars / CenterPoint). The
   detector box is the **measurement** — it localizes independently each frame, so stop/start
   is captured by detection, not by a motion model.
2. **Compensate first**: lift every scan's points (and detections) to the world frame *before*
   association, so ego motion is removed and the pedestrian moves smoothly.
3. **Predict** the next position with a constant-velocity Kalman filter — used to **associate**
   (gate the next detection) and to **coast**, not as the source of position.
4. **Associate**: match the detection inside the predicted gate to the track. GOLD tracks are
   seeded/anchored on the keyframe box (t≈10s, `location_3d` lifted to world); SILVER tracks are
   seeded on the first confident detection. Run **forward** and **backward** from the seed.
5. **Update** Kalman; store innovation residual as `kalman_confidence`. **Coast** (predict-only,
   `in_observation=false`) when no detection falls in the gate; terminate after N consecutive misses.
6. **Smooth**: RTS backward pass over the completed track.

Design choices (debatable — see top of doc): **detector-as-measurement + KF/RTS-as-linker**
(a learned detector localizes each frame independently, fixing the constant-velocity model's lag
at the crossing decision — exactly the moments intent labeling keys off); **compensate-before-
associate**; **online associate + RTS smooth** rather than smooth-only. Pedestrian moves <15cm
between LiDAR frames (~111ms) → a small association gate suffices.

> **Considered & set aside — gated-centroid tracker.** An earlier model-based design used a
> constant-velocity Kalman gate whose centroid-of-enclosed-points was the measurement (no
> detector). Rejected as the primary tracker because the CV model invents position and lags at
> stop/start, and the centroid is biased toward the LiDAR-facing surface. The KF/RTS *linker*
> from that design is reused verbatim — only the measurement source changed (centroid → detector
> box). The gated-centroid measurement survives only as a **coast fallback** when the detector is
> empty. Implementation kept in `src/labeling/tracker.py`; the full gated-centroid run was NOT
> executed.

## Step 2 Bring-up (gates before building the MOT)
The detector pivot rests on two cheap, decisive experiments — both part of standing up the
detector, NOT a separate baseline phase:
1. **World-frame transform validation.** Track a known-static object that has `location_3d`
   (e.g. a `TrafficGuide`/`SnowMarker` pole) through the compensate-first chain; it must stay
   **pinned** in world coordinates. Validates pose interpolation + extrinsics for BOTH designs,
   with no detector needed.
2. **Detector recall on the 1,776 keyframe boxes.** Run PointPillars on the keyframe scans and
   measure recovery of the verified boxes. This is the **go/no-go** for detector-as-measurement:
   good recall → build the MOT; poor recall at range → fine-tune on the keyframe boxes or fall
   back to the gated-centroid measurement.

## Detectors: roles
No per-frame ground truth beyond the keyframe, so the keyframe boxes are the only validation set.
- **3D — PointPillars / CenterPoint**: primary per-scan measurement; validated (recall) against
  the 1,776 keyframe boxes.
- **2D — YOLO + ByteTrack + SAM**: appearance features (crops, body orientation, occlusion masks)
  for intent labeling, AND an independent cross-check (project the 3D track to camera via
  `projection.py`, compare vs the 2D track — **agreement metric, not accuracy**).

## Two-tier labels — keyframe coverage
ZOD annotates one keyframe per 20s clip, so a pedestrian present only before/after it carries no
ZOD annotation. We DO admit such pedestrians, in a separate quality tier so the verified set stays
clean (`label_confidence_tier`, `is_in_gold_standard` already in the schema):
- **GOLD** — keyframe-anchored peds: verified ZOD identity + 3D box.
- **SILVER** — detector-found peds: no human verification; flagged
  `label_confidence_tier=low`, `is_in_gold_standard=false`. Grows the set; reportable separately
  so benchmark metrics on GOLD remain uncontaminated.

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
