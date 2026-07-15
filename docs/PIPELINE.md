# Pipeline Design & Schema

> These are not settled doctrine. If you see a better approach, push back — alternatives and disagreement are welcome. Don't silently follow the doc if something looks wrong, state the trade-off and make the case.

## Pipeline Overview

> **Action ≠ intent.** Two distinct labels live at two levels — never conflate them. The **action** (did this pedestrian cross the road, and *when*) is a verdict about the **whole track** — Step 2: computed geometrically today (feet on the ego road), with **model consensus** as the decided next label source and geometry as its validation anchor (see "Action label source"). The **intent** is **per-window** and a separate later step: v1 is a rough, TTE-anchored label *derived* straight from the action (will t_c fall within the horizon after window_end — Step 3); behavioral intent (pose / attention / social cues — the PIE `intention_prob` sense) is a planned enrichment after that. Labeling each window by what happens *inside* it would collapse intent prediction into action detection and break comparability with JAAD/PIE.

```
[pedestrian_sequences.json — sequences with LiDAR on disk]
        ↓
[Step 0] Detection Cache  (scripts/00_detect.py → data/processed/detections/)         [DONE]
  - Run the 2D detector (YOLO11x) over every camera image ONCE; cache the person boxes per seq.
  - Detects EVERYONE (GOLD + SILVER + false positives).
        ↓
[Step 1] Trajectory Generation  (scripts/01_generate_trajectories.py)                 [DONE]
  - Unit of work: the PEDESTRIAN (one track per pedestrian). Tracks each over the full 20s clip.
  - Measurement = 2D detector box → frustum-lifted to 3D; linker = KF/RTS (associate + coast + smooth); every scan ego-motion-compensated to world frame first.
  - 3D boxes: the shipped per-frame box = tracked centre + rigid keyframe extent + velocity heading (zodped.labeling.boxes.assemble_track_boxes). Clustering is not in the product (see docs/EXPERIMENTS_LOG.md "Boxfit cluster experiment"); box size/yaw come from the keyframe anchor + motion.
  - Two tiers (see "Two-tier labels"):
      * 1a GOLD  (DONE) — anchor-seeded: seed at verified keyframe box, thread through the pool. 358 seqs → 1,863 tracks.
      * 1b SILVER (RAN — pending QC) — birth-seeded (`scripts/01b_generate_silver.py`): births tracks from pool candidates no GOLD track claimed (residual pool → online MOT → support/duration confirmation + a size PRIOR). Full 358-seq pass ran: 9,394 tracks (`silver_run_report.json`) — NOT final until QC'd. Flagged.
  - Output: per-pedestrian world-frame trajectory → data/processed/trajectories/{seq_id}_{pedestrian_id}.json (per frame: position + oriented 3D box; position_ego_rel is per-window → added later)

        ↓   (Steps 2–4 are same code for GOLD and SILVER tracks)
[Step 2] Action Labeling  (scripts/02_label_action.py — track-level, geometric ANCHOR)
  - For each FULL track, compute the crossing ACTION (feet on the ego road) + when it happens. Pure geometry over the whole trajectory.
      * crosses_ego_road : bool — does the track put its feet on the ego_road drivable-surface polygon (project the world point through the KEYFRAME camera → point-in-polygon; FOV/range-limited). The JAAD/PIE "crossing the roadway" notion. This is the geometric GROUND-TRUTH ANCHOR for the model-consensus ACTION labeler (same track-level verdict, decided next label source — see "Action label source"); geometry stays the acting label until that labeler passes its GOLD gate.
      * crossing_frame_timestamp (t_c) — the crossing-onset timestamp (first frame on the road).
  - BENCHED (2026-07-08): the ego-corridor swept-path label (`crosses_ego_corridor` + `ego_distance_at_crossing_m`) was the old primary. The label is now feet-on-road; corridor needs no per-frame road. Its computation is kept dormant in `zodped.labeling.corridor`, re-derivable as a Step-4 aux feature (ego-relevance / metric range-to-crossing). See EXPERIMENTS_LOG.
  - EMPTY / no-real-motion tracks (real_frac=0; the "track" is Kalman coasting from a single anchor) → action = undetermined. 
    NEVER forced to crossing/not_crossing — kept + flagged (a verified ped we could not track).
  - Output: per-track action record (data/processed/actions/{seq_id}_{pedestrian_id}.json).

        ↓
[Step 3] Sample Assembly + Intent Labeling  (scripts/03_assemble_samples.py — per (ped, window))   [BUILT — v1 GOLD run 2026-07-15]
  - Materialise the samples and attach the forward-looking INTENT label, derived from the Step-2 action:
      * TTE-anchored windows: for a crosser, observation windows ENDING at t_c − TTE (the model sees motion BEFORE the event); comparison windows for non-crossers. (Not a dense stride grid — that re-introduces trivial in/post-crossing windows.)
      * per-window filters from the TRACKED position at the window midpoint: distance_to_ego_m ≤ 50.0, distance_to_road_m ≤ 15.0; data-availability ≥ min LiDAR scans, camera frame present.
      * per-window geometry: position_ego_rel (relative to ego at window_start) + multimodal frame pointers.
      * ego context at window_start from vehicle_data.hdf5: ego_speed, turn_indicator (as PIE uses them).
      * intent label per horizon h ∈ {1.0, 1.5, 2.0}s: will the ped start crossing within [window_end, window_end + h]? (forward-looking, NOT "crosses inside this window").
        v1 ROUGH intent = derived straight from the Step-2 action's t_c (TTE derivation). Behavioral intent labeling (pose / attention / social cues) is its own later step. Model consensus labels the ACTION (Step 2), NOT intent — see "Action label source".
  - Output: data/annotations/{sample_id}.json + data/annotations/dataset_index.parquet.
  - v1 GOLD run (2026-07-15, samples_run_report.json): 1,087 samples / 933 peds, 0 failures.
    242 TTE-anchored + 845 comparison; crossing ratio 8.1% @1.0s / 15.5% @1.5s / 22.3% @2.0s.
    Biggest filter cut: distance_to_ego (460 windows — many peds cross >50 m ahead of the ego).
    Also emits per-camera-frame bbox_xyxy (projected 3D box — the JAAD/PIE-family model input,
    defined on coasted frames too).
  - CONFOUND FIXED (same day): comparison windows anchor on the closest OBSERVED road approach
    and require ≥0.7 real-detection fraction (`min_observed_fraction_comparison`). Before: the
    negative class averaged 0.42 observed_frac vs 0.92 for positives — a label-correlated
    smoothness artifact (coasting ⇒ not_crossing) models would shortcut on. After: 0.93 vs 0.92 —
    classes matched on track quality. Cost: 141 non-crossers with no clean observed window
    produce no sample (counted: comparison_window_unobserved).

        ↓
[Step 4] Dataset Packaging + QA
  - Manual review queue (scripts/qc_trajectories.py — built): ranked track review + flags.
  - Reference baseline on GOLD = a label SANITY CHECK (are labels learnable / balanced / JAAD-PIE-comparable), QA only — the dataset is the product, not the model.
  - Split assignment at SEQUENCE level (no window leakage between overlapping windows of one seq).
```

**Sequencing.** Build vertically on GOLD first: prove Steps 2–3 against verified, keyframe-anchored tracks (so a wrong label is a labeling bug, not track noise), 
confirm the crossing-rate distribution and a baseline number, *then* build 1b SILVER and pour it through the same proven Steps 2–4. 
GOLD alone is already ~1,300 usable pedestrians (≈ JAAD/PIE scale), so scale is not the blocker — end-to-end validation is.

## Architecture — detector-as-measurement + KF/RTS linker

Pedestrians are localized per scan by a **detector** (the measurement); a **Kalman/RTS smoother** only links those measurements into a track (the linker). The linker is settled; the measurement source changed on 2026-06-18 (see "Direction & open options").

**Measurement — 2D → frustum lift to 3D.** A strong 2D detector (YOLO/Detectron2) boxes pedestrians on the camera stream; each box is lifted to a 3D position via ZOD calibration + in-frustum LiDAR depth (2D box → in-frustum points → nearest-depth slab). This localizes each frame *independently* of any motion model, so stop/start is captured by detection. The 2D detector also supplies appearance features (crops, body orientation, occlusion) for intent labeling. 
**Cross-check:** projecting the 3D track back to camera via `projection.py` and comparing to the 2D track is an *agreement* metric instead of accuracy — the keyframe boxes are the only validation set.

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

> **Considered & set aside — gated-centroid tracker.** An earlier design used a CV Kalman gate whose centroid-of-enclosed-points was the measurement (no detector). Rejected because the CV model invents position and lags at stop/start, and the centroid is biased toward the LiDAR-facing surface. The gated-centroid measurement code was **removed** from `src/zodped/labeling/tracker.py` (2026-06-24). It is recoverable from git history if a real centroid fallback is ever wanted.

## Direction & open options (updated 2026-06-18, supervisor review)

The measurement source changed here. The original plan localized pedestrians with an off-the-shelf 3D detector (PointPillars / CenterPoint); the bring-up gates exposed a severe ZOD domain gap (off-the-shelf PointPillars recall 0.11–0.49, collapsing to ~0.01–0.24 beyond 40 m where ~52 % of GT lives — see `docs/EXPERIMENTS_LOG.md`). Redirected to the 2D→frustum measurement described in "Architecture": generate good-enough tracks → the maintainer reviews/corrects (3D via rerun / SUSTechPOINTS, 2D via CVAT / Label Studio).

**Open options still on the table (decide as we go):**
1. **Modern 3D detector** to add to or replace the frustum's 3D step — if a 3D model is wanted, use a *new* one (e.g. **SAM4D**).
2. **Fine-tune on a few hand-annotated ZOD sequences** if off-the-shelf 2D→3D quality is inadequate (supervisor-endorsed fallback; 2D expected good, 3D expected weaker).
3. ~~Change dataset~~ — **RESOLVED 2026-06-18: STAYING ON ZOD.** Re-evaluated alternatives: KITScenes, Waymo Perception, NVIDIA PhysicalAI-AV; nuScenes. ZOD keeps its cam+LiDAR+radar novelty; auto-labeling is a sound, validated methodology. Trade-off accepted: no GT trajectories → auto-generate (+ fine-tune + manual review).

4. **2D-first tracking (associate → *then* lift)** — quality lever for the linker. Today the linker
   lifts each 2D box to 3D *first* and associates identities in the world frame, i.e. it decides
   identity on the frustum-noisy **depth** axis and discards the reliable signal (the 2D box) at the
   lift. 2D-first inverts the order: associate in the image (IoU + optional appearance ReID, where the
   box is reliable), then frustum-lift the finished tracklet. Fixes maneuver / pass-by identity swaps
   and is **required for SILVER** (detector-birth has no anchor → a 2D-MOT problem). *Not* a universal
   win: it does not fix frustum depth-jumps (measurement unchanged), can regress anchor-threaded GOLD
   if a generic tracker fragments the seed, and `camera_front_blur` weakens ReID.  Evidence → `docs/EXPERIMENTS_LOG.md`.

**Rejected:** off-the-shelf PointPillars / OpenPCDet as 3D measurement (`docs/EXPERIMENTS_LOG.md`; recoverable at git tag `experiments/3d-detectors`).

## Bring-up gates (DONE — drove the 2026-06-18 pivot)
Two experiments run before building the MOT (full in `docs/EXPERIMENTS_LOG.md`):
1. **World-frame transform validation — PASSED.** A known-static object (`TrafficGuide`/`SnowMarker` pole) with `location_3d`, tracked through the compensate-first chain stayed **pinned** in world coordinates. Validates pose interpolation + extrinsics for every design.
2. **Detector recall on the keyframe boxes — DONE -> NO-GO for off-the-shelf 3D.** 
Off-the-shelf PointPillars recovered only 0.11 (KITTI)/0.49 (nuScenes) of the boxes and collapsed at range; the **frustum** (2D→3D) reached 0.585 with no training. → dropped off-the-shelf 3D detectors, adopted the 2D→frustum measurement ("Direction & open options"). Remaining open lever: fine-tune, or bring in a modern 3D detector (SAM4D).

## Two-tier labels — keyframe coverage
ZOD annotates one keyframe per 20s clip, so a pedestrian present only before/after it carries no ZOD annotation. We DO admit such pedestrians, in a separate quality tier so the verified set stays clean:
- **GOLD** — keyframe-anchored peds: verified ZOD identity + 3D box.
- **SILVER** — detector-found peds: no human verification; flagged `label_confidence_tier=medium`, `is_in_gold_standard=false` (`low` is reserved for a possible future demoted tier).

## Action label source — model-consensus labeler (decided 2026-07-15)

The track-level ACTION verdict (did this pedestrian cross the road, and when) moves from pure
geometry to the **agreement of three local models**. Geometry (the Step-2 feet-on-road test) is the
**ground-truth ANCHOR**: consensus is validated against it on GOLD, and disagreement (model↔model or
model↔anchor) routes a track to manual review. Rationale: the geometric label is FOV/range-limited
(the `ego_road` polygon exists only at the keyframe camera), while models can judge crossing from
the video anywhere on the track. This is ACTION labeling — the per-window intent label is a separate
later step (v1 rough intent derives straight from the action's `t_c`; behavioral intent after that)
and does NOT use this consensus.

**The three models** — `github.com/lindgrenkalle/pip-thesis` (thesis repo; models compared
individually there — the consensus/voting layer is OURS to build):
1. **LSTM-LSTM** encoder-decoder baseline (from `vita-epfl/bounding-box-prediction`).
2. **LSTM-diffusion** — LSTM encoder + anchor-based truncated-diffusion decoder (DiffusionDrive-style).
3. **Transformer-diffusion** — HuggingFace ViT encoder + the same diffusion decoder.

Each consumes bbox position+velocity, ego motion, 224×224 image crops over an observation window and
emits `pred_intent` (crossing logits) + a predicted trajectory (diffusion variants: K samples). The
trajectory head doubles as a candidate for the Step-4 reference baseline. All inputs are derivable
from our pipeline (Step-0 2D boxes / projected Step-1 tracks, `ego_motion`/`vehicle_data`, camera
crops). **[PENDING] adaptation design:** the three are TTE-style *window predictors*, so as ACTION
labelers they are swept per-window over the full track and their votes aggregated into a track
verdict + `t_c` (alternative: add a per-frame crossing head); the voting rule and how consensus
combines with the geometric `t_c` are open — confirm with supervisor.

**Operational facts (checked 2026-07-15):**
- **No pretrained weights ship** — we train all three on JAAD/PIE first (small models; feasible on
  the RTX 4080; requires downloading JAAD + PIE).
- **No LICENSE file** — default all-rights-reserved; contact the author before reusing code wholesale
  (re-implementing against our own data interface is the fallback).
- **Domain gap to validate through** (the PointPillars lesson applies): JAAD/PIE are 30 fps,
  unblurred dashcam; ZOD is 10.1 Hz with anonymisation blur and a 120° lens. The 0.5 s observation
  window is ~5 ZOD frames vs their default 15 (config-exposed in the repo — retrain with the matched
  length, don't naively resample). The image/appearance path (ViT, crops) is the most blur-exposed.
- **GATE before scale:** consensus-vs-anchor agreement on GOLD must pass before the consensus labels
  anything at scale (thresholds PENDING — see schema `action.method`).

## Schema, parameters & filters

Full, authoritative field-by-field spec → **`configs/dataset_schema_v0.2.yaml`** (live source of truth; `v0.1` is the frozen supervisor-approved Week-1 snapshot). Summary:

- **Windowing**: 0.5 s observation window, prediction horizons [1.0, 1.5, 2.0] s, TTE-anchored on the Step-2 crossing onset (not a dense stride grid).
- **Sample unit**: per (pedestrian, window); `sample_id = {seq}_{ped}_{window_start_ms}`. Carries the forward-looking intent label (+ soft label), the per-frame trajectory (world + ego-relative), multimodal file pointers, ego/pedestrian context, and metadata (incl. the GOLD/SILVER tier flags).
- **Action vs intent**: Step 2 emits the track-level **action** (`crosses_ego_road` + `crossing_frame_timestamp` (`t_c`); EMPTY tracks → `undetermined`) — **action label source DECIDED (2026-07-15): model consensus**, with the geometric feet-on-road test as the validation anchor (see "Action label source" above). Step 3 derives the per-window forward-looking **rough intent** from that action (`t_c` × horizon); **behavioral intent labeling** (pose / attention / social cues) follows as its own step (see `docs/JAAD_PIE_ALIGNMENT.md` for the surrounding taxonomy debate). The two never share a field.
- **Filters by stage**: Step 1 selects pedestrians with a 3D keyframe box (`require_3d_annotation`); Step 2 (action) applies **no** filter — it labels every track; Step 3 (assembly) applies the per-window gates from TRACKED positions — proximity (`distance_to_ego ≤ 50 m`, `distance_to_road ≤ 15 m`, at the window midpoint) and data-availability (`min_lidar_frames_in_window`). Proximity is deferred to Step 3 because the single keyframe annotation is a poor proxy for windows up to ~14 m away; a keyframe LiDAR-in-box check is *not* a hard filter (GOLD seeds geometrically).
- **Output**: one JSON per sample + a Parquet index of all scalar fields (fast filtering).
- **Scale (target, PENDING)**: ~10–15 windows/seq after filtering across the 358-seq working set; total sample count + crossing ratio to be recomputed after Step 2/3 and confirmed with supervisor.
