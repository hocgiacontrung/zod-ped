# Pipeline Design & Schema

## Pipeline Overview

```
[pedestrian_sequences.json — 130 sequences with LiDAR]
        ↓
[Step 1] Window Generation & Filtering
  - Slice each sequence into overlapping 0.5s windows (stride 0.25s)
  - Filter out windows where pedestrian is too far from ego road boundary
  - Output: data/processed/candidate_windows.json

        ↓
[Step 2] Trajectory Generation
  - Seed: keyframe 3D box from object_detection.json
  - Track: LiDAR point cloud IoU / nearest-neighbor, forward + backward from keyframe
  - Compensate: ego-motion via ego_motion.json
  - Smooth: Kalman Filter
  - Output: per-frame (x,y,z) trajectory for each pedestrian in world + ego-relative frames

        ↓
[Step 3] Intent Labeling
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
```yaml
max_distance_to_ego_m: 50.0
max_distance_to_road_m: 15.0
max_occlusion: Heavy          # no occlusion filtering; all included, flagged in metadata
min_lidar_frames_in_window: 3 # out of ~4 expected
require_lidar_detection: true
```

## Trajectory Tracking Detail
1. **Seed** at keyframe (t=10s): 3D bounding box from annotation
2. **Forward** (t+1 … t=20s): previous box → next LiDAR scan → highest IoU or nearest centroid cluster → update box
3. **Backward** (t-1 … t=0s): same logic
4. **Ego-motion compensation**: transform all positions to world frame using ego_motion.json
5. **Kalman filter**: smooth after full trajectory is generated; store innovation residual as `kalman_confidence`

Pedestrian moves <15cm between LiDAR frames (~111ms) → small search radius is sufficient.

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
  frames:
    - timestamp: str
      position_world: [x, y, z]
      position_ego_rel: [x, y, z]   # relative to ego pose at window_start; use for training
      in_observation: bool
      kalman_confidence: float
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
