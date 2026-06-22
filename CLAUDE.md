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
- [x] Step 2 bring-up gates done → drove a **DIRECTION PIVOT** (2026-06-18, supervisor review):
  off-the-shelf 3D detectors (PointPillars/OpenPCDet) dropped for a severe ZOD domain gap; now
  **2D-first (Detectron2) + frustum lift to 3D**. See "Step 2 Approach" below + `docs/EXPERIMENTS_LOG.md`.
- [~] **Step 2 – Trajectory generation** (`scripts/02_generate_trajectories.py`) — **GOLD tier
  BUILT** (2026-06-22). Measurement = 2D (YOLO11x) → frustum lift to world; KF/RTS linker reused
  from `src/labeling/tracker.py` (`track_pedestrian_from_detections`); frustum extracted to
  `src/labeling/frustum.py`; BEV review viewer `scripts/viz_trajectories.py`. Verified on 6 seqs
  (36 tracks, 0 failures, 0 false-association jumps). Detector stays YOLO (Detectron2 not built —
  no torch-2.2 wheel; swap deferred, `make_detector` handles rtdetr*). **TODO:** full 358 run
  (~4.7h), manual review, then SILVER tier (detector-discovered peds + track birth/dedup).
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

## Step 2 Approach — DIRECTION PIVOT 2026-06-18 (see docs/PIPELINE.md "Direction & open options")
**Architecture: detector-as-measurement + KF/RTS-as-linker.** The LINKER is settled; the
MEASUREMENT SOURCE changed after the bring-up gates. Off-the-shelf 3D detectors
(PointPillars/OpenPCDet/CenterPoint) were **rejected** — severe ZOD domain gap (recall 0.11–0.49,
collapses >40m where ~52% of GT lives; see `docs/EXPERIMENTS_LOG.md`, code at tag
`experiments/3d-detectors`).

**Current direction (building):**
- **Unit = pedestrian, not window**: track each ped once over the full 20s clip. Step 2 does NOT
  load the 241 MB candidate_windows.json.
- **Measurement = 2D-first → frustum lift to 3D**: strong 2D detector (YOLO → **Detectron2**) box
  + ZOD calibration/projection + in-frustum LiDAR depth → per-scan 3D position. Best gate (recall
  0.585, ~15cm median, zero training). Generate good-enough tracks → The maintainer manually reviews.
- **KF/RTS = linker** (unchanged): CV Kalman associates measurements (gate) + coasts gaps; RTS
  backward smooth. Reused verbatim from `src/labeling/tracker.py`; only the measurement source
  changed. Coast on miss → `in_observation=false`.
- **Compensate-before-associate**: lift each scan to world frame via interpolated ego pose first.
- **Two tiers**: GOLD = keyframe-anchored peds (verified box+identity); SILVER = detector-found
  peds (flagged `label_confidence_tier=low`, `is_in_gold_standard=false`) — grows the set without
  contaminating the GOLD benchmark.

**Open options still on the table (decide as we go — NOT yet committed):**
1. **Modern 3D detector** (e.g. **SAM4D**, 2025) to add to/replace the frustum's 3D step — never
   the old PointPillars/CenterPoint family again.
2. **Fine-tune on a few hand-annotated ZOD sequences** if 2D→3D quality is inadequate.
3. ~~Switch dataset~~ — **RESOLVED 2026-06-18: STAYING ON ZOD.** Candidates rejected — KITScenes
   (no pedestrian GT), Waymo Interaction Pred. (no raw sensors), NVIDIA PhysicalAI-AV (133 TB, no GT
   ped tracks), nuScenes (old + intent already taken by PePScenes). MAN TruckScenes (GT tracks + 4D
   radar) close but truck/highway = sparse pedestrians. ZOD novelty intact; auto-labeling validated.

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
