# Pipeline Design & Schema

> These are working decisions, not settled doctrine. If you see a better approach, push back — alternatives and disagreement are welcome. State the trade-off and
> make the case; don't silently follow the doc if something looks wrong.

## Pipeline Overview

```
[pedestrian_sequences.json — sequences with LiDAR on disk]
        ↓
[Step 1] Trajectory Generation  (scripts/01_generate_trajectories.py)
  - Unit of work: the PEDESTRIAN (one track per pedestrian), NOT the window. Derives the (seq, ped) set directly from the keyframe annotations (src/dataset/keyframe.py) and tracks each once over the full 20s clip.
  - Measurement = 2D detector box → frustum-lifted to 3D; linker = KF/RTS (associate + coast + smooth); every scan ego-motion-compensated to world frame first. Mechanics → "Architecture".
  - Two tiers: GOLD (keyframe-anchored) / SILVER (detector-found). See "Two-tier labels".
  - Output: per-pedestrian world-frame trajectory → data/processed/trajectories/{seq_id}_{pedestrian_id}.json (position_ego_rel is per-window → added later)
  - Bring-up gates ran BEFORE this was built — see "Bring-up gates".

        ↓
[Step 2] Sample Assembly + Filter  (scripts/02_assemble_samples.py)
  - Now that per-frame tracked positions exist, MATERIALISE the per-window samples
    (windows are born here, only for tracked peds that pass — never pre-enumerated):
      * enumerate the deterministic window grid: 0.5s windows at 0.25s stride over the clip
      * data-availability filter: ≥ min LiDAR scans in window, camera frame present
      * proximity filter, from the TRACKED position at the window midpoint:
            distance_to_ego_m ≤ 50.0
            distance_to_road_m ≤ 15.0   (projected to the ego_road polygon)
      * attach per-window geometry: position_ego_rel (relative to ego at window_start) and the multimodal frame pointers
  - Output: data/processed/samples.json

        ↓
[Step 3] Intent Labeling  (scripts/03_label_intent.py)
  - Rule-based (~75–80%): trajectory crosses ego road centerline within horizon?
  - Pose estimation (~10–15%): body orientation for ambiguous cases
  - VLM (~5%): Gemini 1.5 Flash (free tier) or Llama 3.2 Vision 11B (local, ~7GB VRAM) always provide trajectory coordinates in prompt
  - Manual (~5–10%): gold standard validation subset
  - Output: data/annotations/{sample_id}.json + data/annotations/dataset_index.parquet
```

## Architecture — detector-as-measurement + KF/RTS linker

Pedestrians are localized per scan by a **detector** (the measurement); a **Kalman/RTS smoother** only links those measurements into a track (the linker). The linker is settled; the measurement source changed on 2026-06-18 (see "Direction & open options").

**Measurement — 2D → frustum lift to 3D.** A strong 2D detector (YOLO/Detectron2) boxes pedestrians on the camera stream; each box is lifted to a 3D position via ZOD calibration + in-frustum LiDAR depth (2D box → in-frustum points → nearest-depth slab). This localizes each frame *independently* of any motion model, so stop/start is captured by detection, not invented by the filter. The 2D detector also supplies appearance features (crops, body orientation, occlusion) for intent labeling. **Cross-check:** projecting the 3D track back to camera via `projection.py` and comparing to the 2D track is an *agreement* metric instead of accuracy — no per-frame GT exists beyond the keyframe, so the keyframe boxes are the only validation set.

**Linker — constant-velocity Kalman + RTS.** Associates measurements across scans (gating), coasts through gaps, and smooths. It links boxes into a track but is NOT the source of position during normal tracking.

**Compensate first.** Every scan's points (and detections) are lifted to a single world frame `p_world = pose(t) @ T_ego_lidar @ p_lidar`, where `T_ego_lidar = calib["FC"]["lidar_extrinsics"]` and `pose(t)` is interpolated (SLERP + lerp) from `ego_motion["poses"]` (T[world←ego], origin = ego at clip start) at the scan timestamp — so ego motion is never mistaken for pedestrian motion.

Per-scan procedure:
1. **Detect + lift**: 2D box → frustum 3D position = the measurement.
2. **Compensate**: lift points/detections to the world frame *before* association.
3. **Predict** the next position with the CV Kalman — used to associate (gate the next detection) and to coast, not as the source of position.
4. **Associate**: match the in-gate detection to the track. GOLD tracks are anchored on the keyframe box (t≈10s, `location_3d` lifted to world); SILVER tracks seed on the first confident detection. Run **forward** and **backward** from the seed.
5. **Update** Kalman; store innovation residual as `kalman_confidence`. **Coast** (predict-only, `in_observation=false`) when no detection falls in the gate; terminate after N consecutive misses.
6. **Smooth**: RTS backward pass over the completed track.

Design choices (debatable): a learned detector localizes each frame independently, fixing the CV model's lag at the crossing decision — exactly the moments intent labeling keys off; compensate-before-associate; online associate + RTS smooth rather than smooth-only. A pedestrian moves <15 cm between LiDAR frames (~111 ms), so a small association gate suffices.

> **Considered & set aside — gated-centroid tracker.** An earlier model-based design used a CV
> Kalman gate whose centroid-of-enclosed-points was the measurement (no detector). Rejected as the
> primary tracker because the CV model invents position and lags at stop/start, and the centroid is
> biased toward the LiDAR-facing surface. The KF/RTS *linker* from that design is reused verbatim —
> only the measurement source changed (centroid → detector box). The gated-centroid measurement
> survives only as a **coast fallback** when the detector is empty. Kept in
> `src/labeling/tracker.py`; the full gated-centroid run was NOT executed.

## Direction & open options (updated 2026-06-18, supervisor review)

The measurement source changed here. The original plan localized pedestrians with an off-the-shelf 3D detector (PointPillars / CenterPoint); the bring-up gates exposed a severe ZOD domain gap (off-the-shelf PointPillars recall 0.11–0.49, collapsing to ~0.01–0.24 beyond 40 m where ~52 % of GT lives — see `docs/EXPERIMENTS_LOG.md`). The 2026-06-18 supervisor meeting redirected to the 2D→frustum measurement now described in "Architecture" (recall 0.585, ~15 cm median, zero training): generate good-enough tracks → The maintainer reviews/corrects (3D via rerun / SUSTechPOINTS, 2D via CVAT / Label Studio).

**Open options still on the table (decide as we go):**
1. **Modern 3D detector** to add to or replace the frustum's 3D step — if a 3D model is wanted, use a *new* one (e.g. **SAM4D**), NOT the old PointPillars/CenterPoint family.
2. **Fine-tune on a few hand-annotated ZOD sequences** if off-the-shelf 2D→3D quality is inadequate (supervisor-endorsed fallback; 2D expected good, 3D expected weaker).
3. ~~Change dataset~~ — **RESOLVED 2026-06-18: STAYING ON ZOD.** Re-evaluated alternatives:
   KITScenes (HD-maps, no pedestrian GT), Waymo Perception (ready to label intent, switch to only if tracking ZOD can't work), NVIDIA PhysicalAI-AV (machine-labels, no GT ped tracks) all rejected; nuScenes dropped (2019, and pedestrian-intent already taken by PePScenes); MAN TruckScenes (GT tracks + 4D radar) but truck/highway → sparse pedestrians. ZOD keeps its cam+LiDAR+radar novelty; auto-labeling is a sound, validated methodology. Trade-off accepted: no GT trajectories → auto-generate via 2D→frustum (+ fine-tune a few sequences if needed) + manual review.

**Rejected:** off-the-shelf PointPillars / OpenPCDet as the 3D measurement — domain gap too large (`docs/EXPERIMENTS_LOG.md`; code recoverable at git tag `experiments/3d-detectors`).

## Bring-up gates (DONE — drove the 2026-06-18 pivot)
Two experiments run before building the MOT (full numbers in `docs/EXPERIMENTS_LOG.md`):
1. **World-frame transform validation — PASSED.** A known-static object (`TrafficGuide`/`SnowMarker` pole) with `location_3d`, tracked through the compensate-first chain stayed **pinned** in world coordinates (seq 000007: world-frame std 0.027 m vs 0.959 m uncompensated). Validates pose interpolation + extrinsics for every design; no detector needed.
2. **Detector recall on the keyframe boxes — DONE, go/no-go answered: NO-GO for off-the-shelf 3D.** 
Off-the-shelf PointPillars recovered only 0.11 (KITTI)/0.49 (nuScenes) of the boxes and collapsed at range; the **frustum** (2D→3D) reached 0.585 with no training. → dropped off-the-shelf 3D detectors, adopted the 2D→frustum measurement (see "Direction & open options"). Remaining open lever: fine-tune, or bring in a modern 3D detector (SAM4D).

## Two-tier labels — keyframe coverage
ZOD annotates one keyframe per 20s clip, so a pedestrian present only before/after it carries no ZOD annotation. We DO admit such pedestrians, in a separate quality tier so the verified set stays clean (`label_confidence_tier`, `is_in_gold_standard` already in the schema):
- **GOLD** — keyframe-anchored peds: verified ZOD identity + 3D box.
- **SILVER** — detector-found peds: no human verification; flagged `label_confidence_tier=low`, `is_in_gold_standard=false`. Grows the set; reportable separately so benchmark metrics on GOLD remain uncontaminated.

## Schema, parameters & filters

Full, authoritative field-by-field spec → **`configs/dataset_schema_v0.2.yaml`** (live source of truth; `v0.1` is the frozen supervisor-approved Week-1 snapshot). Summary:

- **Windowing**: 0.5 s observation window, 0.25 s stride, prediction horizons [1.0, 1.5, 2.0] s.
- **Sample unit**: per (pedestrian, window); `sample_id = {seq}_{ped}_{window_start_ms}`. Carries the intent label (+ soft label), the per-frame trajectory (world + ego-relative), multimodal file pointers, ego/pedestrian context, and metadata (incl. the GOLD/SILVER tier flags).
- **Filters by stage**: Step 1 selects pedestrians with a 3D keyframe box (`require_3d_annotation`); Step 2 applies the per-window gates from TRACKED positions — proximity (`distance_to_ego ≤ 50 m`, `distance_to_road ≤ 15 m`, at the window midpoint) and data-availability (`min_lidar_frames_in_window`). Proximity is deferred to Step 2 because the single keyframe annotation is a poor proxy for windows up to ~14 m away; a keyframe LiDAR-in-box check is *not* a hard filter (GOLD seeds geometrically).
- **Output**: one JSON per sample + a Parquet index of all scalar fields (fast filtering).
- **Scale (target, PENDING)**: ~10–15 windows/seq after filtering across the 358-seq working set; total sample count + crossing ratio to be recomputed after Step 2/3 and confirmed with supervisor.
