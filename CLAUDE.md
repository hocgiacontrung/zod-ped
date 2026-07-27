# ZOD Pedestrian Intent & Trajectory Dataset

Internship project at Intelligent Robotics Lab, Aalto University.
Conda env: `zod-iac` | Server: `user20@aalto`

## Working Preferences
- Pipeline decisions are open to debate. If you see a better approach, challenge it — propose alternatives, state the trade-off.

## Goal
Build a multimodal pedestrian intent & trajectory prediction dataset on top of Zenseact Open Dataset (ZOD). Key novelty: synchronized camera + LiDAR + radar,
unlike existing pedestrian intent datasets (JAAD, PIE, PSI) which are camera-only.

## Current Status (Week 5)
- [x] 358 sequences with pedestrian annotations identified → `data/pedestrian_sequences.json`
- [x] All 358 pedestrian sequences now have LiDAR on disk (non-pedestrian LiDAR pruned; 1473 seq dirs total, 358 retain `lidar_velodyne/`) → full working set
- [x] Dataset schema approved → `docs/PIPELINE.md`
- [x] Exploration notebook → `notebooks/01_explore_sequence.ipynb` (seq 000007)
- [x] Projection utility → `src/zodped/utils/projection.py`
- [x] Trajectory bring-up gates done → **DIRECTION PIVOT** (2026-06-18): off-the-shelf 3D detectors (PointPillars/OpenPCDet) dropped for domain gap; now **2D-first + frustum lift to 3D**. See "Trajectory Approach" below + `docs/EXPERIMENTS_LOG.md`.
- [x] **Step 1 – Trajectory generation** (`scripts/01_generate_trajectories.py`)
  **GOLD tier BUILT + full 358 run DONE** (re-run 2026-06-29 w/ per-frame boxes): 358 seqs → 1,863 tracks, 0 failures (`data/processed/reports/trajectories_run_report.json`). **Step 0** caches 2D boxes (`scripts/00_detect.py` → `data/processed/detections/`, all 358 cached; `src/zodped/labeling/detection_cache.py`). Measurement = 2D (YOLO11x) → frustum lift to world (`frustum.py`); KF/RTS linker (`tracker.py: track_pedestrian_from_detections`); per-frame 3D **box** = tracked centre + rigid keyframe extent + velocity yaw (`src/zodped/labeling/boxes.py`). QC scorer built: `scripts/qc_trajectories.py` (ranked review queue + flags + occlusion summary). 
  Demo/QC viz: `viz_render_video.py` (multi-ped MP4), `viz_trajectories.py` (BEV + QC table), `viz_find_demo_pedestrians.py`. 
  Linker bridges occlusions only up to `max_consecutive_misses`≈5 frames (~0.44s). This is **Step 1a GOLD** (anchor-seeded). **Step 1b SILVER** (detector-birth, `scripts/01b_generate_silver.py`): full 358-seq pass RAN (9,394 tracks, `data/processed/reports/silver_run_report.json`) — **pending QC**, don't treat as final. **TODO:** SILVER QC + the manual review pass.
- [~] **Step 2 – Action labeling** (`scripts/02_label_action.py`). Per full track: crossing ACTION = `crosses_ego_road` (feet on the `ego_road` polygon; project each world point through the keyframe camera → point-in-polygon) + `crossing_frame_timestamp` (`t_c`, first frame on the road). EMPTY tracks → `undetermined` (kept+flagged, never forced). Feet-on-road is the **acting** label only — the model-consensus ACTION labeler is validated against **human curated labels** (the anchor batch showed geometry ~30% wrong vs a human, so geometry is retired as the yardstick). Committee (→ `docs/PIPELINE.md` "Action label source"): **PV-LSTM member #1** — human-anchor gate 2026-07-27 (`scripts/02b_committee_gate.py`, truth = `curation_worksheet.csv`) AUC **0.74** (detector boxes), up from the old geometry-graded 0.55; PedGraph+ next. JAAD lives at `data/external/JAAD`. **Corridor BENCHED (2026-07-08)** (re-derivable as an aux feature). **Action ≠ intent** — keep separate.
- [~] **Step 3 – Sample assembly + intent labeling** (`scripts/03_assemble_samples.py`, `src/zodped/labeling/samples.py`). **BUILT + full GOLD run DONE (2026-07-15)**: 1,087 samples / 933 peds, 0 failures (`data/processed/reports/samples_run_report.json`; per-sample JSON + Parquet index → `data/annotations/`). TTE-anchored windows from `t_c` (242 crosser windows; 845 comparison windows at the closest **observed** road approach, ≥0.7 real-detection fraction — the raw closest-approach rule filled the negative class with coasted segments, a "smooth motion ⇒ not_crossing" shortcut; fixed same day, classes now matched 0.93 vs 0.92 observed_frac) + per-window filters (≤50m ego / ≤15m road at midpoint, ≥3 LiDAR scans; biggest cut = distance_to_ego, 460 windows) + `position_ego_rel` + projected-3D-box **bbox sequences** per camera frame (JAAD/PIE-model input; ~10% null = out of fisheye FOV) + ego context (speed/heading/turn-indicator, `src/zodped/utils/vehicle_data.py`). Crossing ratio by horizon: 8.1% @1.0s / 15.5% @1.5s / **22.3% @2.0s** (target band 20–30%). Intent = forward-looking PREDICTION (`t_c ∈ [window_end, window_end+h]`) — **not** "crosses inside the window"; v1 ROUGH intent derives straight from Step-2 `t_c`; **behavioral intent labeling** (pose/attention/social cues; faces blurred) is its own later step. **TODO:** supervisor sign-off on counts/ratio (+ whether to relax the 50m gate for crosser windows — positives are scarce: 88 @1.0s), split assignment (Step 4), SILVER pour after SILVER QC.
- [~] **Step 4 – Dataset packaging + QA** — **Step 4a split assignment DONE (2026-07-16)**: `scripts/04_assign_splits.py` + `src/zodped/dataset/splits.py`; sequence-level stratified deal (crosser-window strata, best of 64 seeds) over all 358 working-set seqs, **FROZEN** at `data/processed/splits/sequence_splits.json` (re-deal = `--force`, forbidden once any result is reported on it). Achieved 775/99/213 samples (ratio 70/10/20), crossing ratio @2.0s 0.222/0.232/0.221 (corpus 0.223); Step 3 re-runs + SILVER pour inherit the mapping automatically. **TODO:** reference baseline on GOLD as a label sanity check (dataset QA, not the product); manual review queue (built).
- **Tiers live only in Step 1**; Steps 2–4 are tier-agnostic (carry `is_in_gold_standard`). Build vertically on GOLD first, then pour SILVER through the same Steps 2–4. Full workflow → `docs/PIPELINE.md` "Pipeline Overview".

## Key Gotchas (verified, seq 000007)
**`location_3d` is in the LiDAR sensor frame** (so `ego_road.json` is image-pixel, not 3D-comparable —> project first).
Full set — frame conventions, structured-array `.npy`, per-point µs timestamp offset, 2D-only guard, 55ms scan-gap limit → `docs/DATA_FORMAT.md`.

## Trajectory Approach (pivot 2026-06-18)
Measurement pivoted from off-the-shelf 3D detectors to **2D detector → frustum lift to 3D**, fed into **KF/RTS linker** (CV-Kalman associate + coast, RTS smooth; compensate-before-associate). Unit = pedestrian (tracked once over the full clip). Two tiers: GOLD (keyframe-anchored, verified) / SILVER (detector-found, flagged `is_in_gold_standard=false`).
Output: per-ped `data/processed/trajectories/{seq_id}_{pedestrian_id}.json` (world frame); `position_ego_rel` is per-window, added in Step 3 (sample assembly).
→ Architecture, open options (SAM4D / fine-tune), dataset decision, evidence: `docs/PIPELINE.md` "Direction & open options" + `docs/EXPERIMENTS_LOG.md`.

## Key Constraints
- LiDAR files named by UTC timestamp, not frame index — always match by timestamp
- Annotations exist only at 1 keyframe per sequence (central frame of 20s clip)
- `data/pedestrian_sequences.json` entries: `{seq_id, num_pedestrians, lidar_batch}`
- Do NOT use ZodSequences API for full dataset — load JSON files directly (trainval-sequences-full.json not available for partial downloads)
- ZOD devkit IS installed at `zod-iac` conda env — safe to read source and use low-level utilities (e.g. `project_3d_to_2d_kannala`), just not the full `ZodSequences` loader
- No budget for VLMs — use local open-source model

## Project Layout
```
zod-ped/
├── data/
│   ├── raw/sequences/XXXXXX/     ← ZOD data (annotations, lidar, images, etc.)
│   ├── processed/                ← pipeline outputs (subdirs only; no loose files)
│   │   ├── detections/          ← Step 0 output: cached 2D person boxes per seq ({seq}.json)
│   │   ├── trajectories/         ← Step 1 output: per-ped world-frame tracks ONLY ({seq}_{ped}.json)
│   │   ├── actions/             ← Step 2 output: per-track action records ({seq}_{ped}.json)
│   │   ├── reports/              ← run reports (trajectories_run_report.json, detector/frustum gates)
│   │   ├── review/              ← generated manual-review artifacts (BEV + overlay PNGs)
│   │   └── splits/               ← Step 4a output: FROZEN sequence_splits.json (seq_id → train/val/test)
│   ├── annotations/              ← generated per-sample intent labels + index (Step 3 output)
│   ├── external/JAAD/            ← ykotseruba/JAAD clone (MIT: annotations + jaad_data.py loader) + JAAD_clips/ (346 MP4s, 2.9GB) inside it — the layout every JAAD-consuming repo expects. NEVER run split_clips_to_frames.sh (169GB; disk won't fit — decode frames on the fly instead)
│   └── pedestrian_sequences.json    ← 358 sequences with pedestrian annotations
├── pyproject.toml               ← installable package config; `pip install -e . --no-deps` (then `import zodped` works everywhere — NO sys.path hacks)
├── src/zodped/                  ← the importable library (src-layout package)
│   ├── dataset/                 ← keyframe.py (ZOD loading & data structures), splits.py (Step 4a deal logic)
│   ├── labeling/                ← detector.py (make_detector), detection_cache.py, frustum.py, tracker.py, boxes.py
│   └── utils/                   ← projection.py (Kannala fisheye projection), ego_motion.py
├── scripts/                     ← runnable entry-points (thin: argparse + I/O + calls into zodped)
│   ├── _common.py                ← shared CLI plumbing: paths, frustum-pool arg group (keeps 1a/1b in sync), --tier filter
│   ├── prune_lidar.py            ← delete lidar_velodyne/ for non-pedestrian seqs (run per batch)
│   ├── 00_detect.py                  ← Step 0: cache 2D detections once (run before Step 1)
│   ├── qc_trajectories.py            ← Step 1 QC: score tracks + rank manual-review queue
│   ├── 01_generate_trajectories.py   ← Step 1a: GOLD trajectory generation (anchor-seeded)
│   ├── 01b_generate_silver.py        ← Step 1b: SILVER trajectories (detector-birth; ran, pending QC)
│   ├── 02_label_action.py            ← Step 2: action labeling (track-level, geometric)
│   ├── 03_assemble_samples.py        ← Step 3: sample assembly + intent labeling (TTE-anchored)
│   └── 04_assign_splits.py           ← Step 4a: frozen sequence-level split deal + stamping
├── notebooks/
│   ├── 01_explore_sequence.ipynb    ← seq 000007 exploration (verified)
│   └── 02_bringup_gates.ipynb       ← detector-recall + frustum-POC gate results (pivot evidence)
├── configs/                      ← pipeline parameters (YAML)
└── docs/
    ├── DATA_FORMAT.md            ← sensor specs, file formats, annotation fields
    ├── PIPELINE.md               ← pipeline steps, schema, labeling strategy
    └── EXPERIMENTS_LOG.md        ← detector bring-up gates + pivot evidence
```

## Reference Docs
- Sensor specs, file formats, frame conventions → `docs/DATA_FORMAT.md`
- Pipeline design, schema summary, open options → `docs/PIPELINE.md`
- JAAD/PIE label taxonomy mapping + open Step-3/4 decisions → `docs/JAAD_PIE_ALIGNMENT.md`
- Detector bring-up evidence (pivot rationale) → `docs/EXPERIMENTS_LOG.md`
- Full schema spec (live source of truth) → `configs/dataset_schema_v0.2.yaml`
  (`v0.1` = frozen supervisor-approved Week-1 snapshot)
