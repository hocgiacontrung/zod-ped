# ZOD Pedestrian Intent & Trajectory Dataset

Internship project at Intelligent Robotics Lab, Aalto University.
Conda env: `zod-iac` | Server: `user20@aalto`

## Working Preferences
- Pipeline decisions are open to debate. If you (any session, human or AI) see a better approach,
  challenge it — propose alternatives, state the trade-off, don't just follow the docs.

## Goal
Build a multimodal pedestrian intent & trajectory prediction dataset on top of
Zenseact Open Dataset (ZOD). Key novelty: synchronized camera + LiDAR + radar,
unlike existing pedestrian intent datasets (JAAD, PIE, PSI) which are camera-only.
Schema approved. Now building the pipeline.

## Current Status (Week 2)
- [x] 358 sequences with pedestrian annotations identified → `data/pedestrian_sequences.json`
- [x] All 358 pedestrian sequences now have LiDAR on disk (non-pedestrian LiDAR pruned via
  `scripts/prune_lidar.py`; 1473 seq dirs total, 358 retain `lidar_velodyne/`) → full working set
- [x] Dataset schema approved by supervisor → `docs/PIPELINE.md`
- [x] Exploration notebook → `notebooks/01_explore_sequence.ipynb` (seq 000007)
- [x] Projection utility → `src/utils/projection.py`
- [x] Step 1 script → `scripts/01_filter_sequences.py` → `data/processed/candidate_windows.json`
- [ ] **Next: Step 2 – Trajectory generation** (`scripts/02_generate_trajectories.py`) —
  detector-as-measurement architecture (see below). Two bring-up gates first:
  (1) world-frame transform validation, (2) detector recall vs the 1,776 keyframe boxes.
- [ ] Step 3 – Proximity filter (`scripts/03_filter_by_trajectory.py`)
- [ ] Step 4 – Intent labeling (`scripts/04_label_intent.py`)

## Key Findings from Exploration (seq 000007)
- **`location_3d` is in the LiDAR sensor frame**, not the vehicle ego frame — despite ZOD
  docs saying "sensor/ego frame". Verified: projecting via `inv(cam_ext) @ lid_ext` places
  centroids within ±35px of annotated 2D bboxes.
- **`ego_road.json` is in image pixel coordinates** — cannot be directly compared to
  `location_3d`. Road proximity checks require projecting 3D points to image space first.
- **LiDAR `.npy` is a structured array** — use `cloud['x']` not `cloud[:,0]`.
  Per-point `timestamp` is a relative µs offset from the filename timestamp, not absolute UTC.
- **Some pedestrians are 2D-only** (no `location_3d`). Always guard:
  `"location_3d" in p["properties"]` before accessing any 3D field.
- **Nearest LiDAR scan gap**: 37.4ms for seq 000007 — well within 55ms limit.

## Step 1 Output: candidate_windows.json
Full-set results (358 sequences, 2,159 total pedestrians):
- 296 skipped (2D-only), 87 skipped (no LiDAR detection at keyframe)
- 1,776 pedestrians → **140,072 candidate windows**
- Output written **compact** (no indent) → 241 MB; load with a streaming parser in Step 2
- distance_to_ego_m and distance_to_road_m stored as metadata only — NOT filtered here
- Proximity filtering deferred to Step 3 (needs tracked positions)
- Note: `MIN_LIDAR_IN_WINDOW=3` guard never fires in practice (~5 scans/0.5s window)
- Run: ~19s for all 358; continue-on-error with a `failures` list in the output JSON

## Step 2 Approach (see docs/PIPELINE.md)
**Architecture: detector-as-measurement + KF/RTS-as-linker.** Decided 2026-06-17 over the earlier
gated-centroid design (CV model lags at stop/start — the crossing decision — and centroid is
surface-biased). Going straight to the detector architecture; the gated-centroid **full run was
dropped** (kept only as a coast fallback). See [[pipeline-status]].
- **Unit = pedestrian, not window**: track each ped once over the full 20s clip. Step 2 does NOT
  load the 241 MB candidate_windows.json.
- **3D detector (PointPillars / CenterPoint) = per-scan measurement** — localizes independently
  each frame, so stop/start is captured by detection, not invented by a motion model.
- **KF/RTS = linker**: constant-velocity Kalman associates detections (gate) + coasts gaps; RTS
  backward smooth. The linker machinery is reused verbatim from `src/labeling/tracker.py`; only
  the measurement source changed (gated centroid → detector box). Coast on miss → `in_observation=false`.
- **Compensate-before-associate**: lift each scan to world frame via interpolated ego pose first.
- **Two tiers**: GOLD = keyframe-anchored peds (verified box+identity); SILVER = detector-found
  peds (flagged `label_confidence_tier=low`, `is_in_gold_standard=false`) — grows the set without
  contaminating the GOLD benchmark.
- **2D — YOLO+ByteTrack+SAM**: appearance features (crops, body orientation, occlusion) for intent
  labeling + independent cross-check (agreement metric, not accuracy — no per-frame GT).
- **Bring-up gates (do first):** (1) world-frame transform validation (static pole stays pinned);
  (2) detector recall vs the 1,776 keyframe boxes (go/no-go for detector-as-measurement).
- Output: per-ped `data/processed/trajectories/{seq_id}_{pedestrian_id}.json` (world frame).
  `position_ego_rel` is per-window → added at sample assembly, not Step 2.

## Key Constraints
- LiDAR files named by UTC timestamp, not frame index — always match by timestamp
- Annotations exist only at 1 keyframe per sequence (central frame of 20s clip)
- `pedestrian_sequences.json` entries: `{seq_id, num_pedestrians, lidar_batch}`
- `pedestrian_sequences.json` is at `data/pedestrian_sequences.json` (not `data/splits/`)
- Do NOT use ZodSequences API for full dataset — load JSON files directly
  (trainval-sequences-full.json not available for partial downloads)
- ZOD devkit IS installed at `zod-iac` conda env — safe to read source and use low-level
  utilities (e.g. `project_3d_to_2d_kannala`), just not the full `ZodSequences` loader
- No budget for VLMs — use Gemini 1.5 Flash free tier or local Llama 3.2 Vision 11B

## Project Layout
```
zod-ped/
├── data/
│   ├── raw/sequences/XXXXXX/     ← ZOD data (annotations, lidar, images, etc.)
│   ├── processed/                ← pipeline outputs
│   │   └── candidate_windows.json   ← Step 1 output (140,072 windows, 241 MB compact)
│   ├── annotations/              ← generated pseudo-labels (Step 4 output)
│   └── pedestrian_sequences.json    ← 358 sequences with pedestrian annotations
├── src/
│   ├── dataset/                  ← ZOD loading & data structures
│   ├── labeling/                 ← filtering, tracking, intent labeling
│   ├── utils/
│   │   └── projection.py         ← Kannala projection, road polygon check
│   └── visualization/
├── scripts/
│   ├── prune_lidar.py            ← delete lidar_velodyne/ for non-pedestrian seqs (run per batch)
│   ├── 01_filter_sequences.py    ← Step 1: window generation
│   ├── 02_generate_trajectories.py   ← Step 2: TODO
│   ├── 03_filter_by_trajectory.py    ← Step 3: TODO
│   └── 04_label_intent.py            ← Step 4: TODO
├── notebooks/
│   └── 01_explore_sequence.ipynb    ← seq 000007 exploration (verified)
├── configs/                      ← pipeline parameters (YAML)
└── docs/
    ├── DATA_FORMAT.md            ← sensor specs, file formats, annotation fields
    └── PIPELINE.md               ← pipeline steps, schema, labeling strategy
```

## Reference Docs
- Sensor specs, file formats → `docs/DATA_FORMAT.md`
- Pipeline design, schema summary → `docs/PIPELINE.md`
- Full schema spec (source of truth) → `docs/dataset_schema_v0_1.yaml`
