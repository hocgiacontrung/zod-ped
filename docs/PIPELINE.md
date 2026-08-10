# Pipeline Design & Schema

What the pipeline *is* and why it's shaped this way. Dated results and evidence live in
`docs/EXPERIMENTS_LOG.md`; current step-by-step state lives in `CLAUDE.md`.

> Not settled doctrine. If something looks wrong, say so and make the case — don't silently follow.

## Pipeline Overview

**Action ≠ intent.** The **action** is a verdict about the *whole track*: did this pedestrian cross
the ego road, and at what time `t_c` (Step 2). The **intent** is *per window* and forward-looking:
will crossing start within the horizon *after* the window ends (Step 3). Labeling a window by what
happens *inside* it would collapse intent prediction into action detection and break comparability
with JAAD/PIE.

```
[pedestrian_sequences.json]
   ↓
[Step 0] Detection cache — YOLO11x person boxes over every camera image, once, per seq.
   ↓      Detects EVERYONE: GOLD + SILVER + false positives.
[Step 1] Trajectory generation — unit of work is the PEDESTRIAN, tracked over the full 20s clip.
   ↓      Measurement = 2D box → frustum-lifted to 3D. Linker = KF/RTS. World frame throughout.
   ↓      Shipped per-frame 3D box = tracked centre + rigid keyframe extent + velocity yaw.
   ↓      1a GOLD  — anchor-seeded from the verified keyframe box.
   ↓      1b SILVER — birth-seeded from pool candidates no GOLD track claimed, + a size prior.
   ↓      → data/processed/trajectories/{seq}_{ped}.json
   ↓
   ↓   ——— Steps 2–4 are the same code for both tiers ———
   ↓
[Step 2] Action labeling — per full track, geometric.
   ↓      crosses_ego_road : do the feet land on the ego_road polygon? (project each world point
   ↓        through the KEYFRAME camera → point-in-polygon; FOV/range-limited by construction)
   ↓      crossing_frame_timestamp (t_c) : first frame on the road.
   ↓      EMPTY tracks (Kalman coasting from a single anchor, real_frac=0) → undetermined.
   ↓        Kept and flagged, NEVER forced to a class.
   ↓      → data/processed/actions/{seq}_{ped}.json
   ↓
[Step 2e] Human merge (GOLD only) — human verdicts written over geometry's two fields.
   ↓      A separate output dir, never a mutation: Step 2 stays reproducible from geometry alone
   ↓      and the human layer stays independently auditable.
   ↓      → data/processed/actions_verified/{seq}_{ped}.json   ← what Step 3 should read
   ↓
[Step 3] Sample assembly + intent labeling — per (pedestrian, window).
   ↓      TTE-anchored windows: for a crosser, observation windows ENDING at t_c − TTE, so the model
   ↓        sees motion BEFORE the event. Not a dense stride grid (that re-introduces trivial
   ↓        in-crossing and post-crossing windows).
   ↓      Comparison windows for non-crossers, at the closest OBSERVED road approach.
   ↓      Per-window filters from the TRACKED position at the window midpoint.
   ↓      Per-window geometry, bbox sequences, ego context.
   ↓      → data/annotations/{sample_id}.json + dataset_index.parquet
   ↓
[Step 4] Packaging + QA
          4a splits — SEQUENCE-level stratified deal, so overlapping windows of one sequence can
             never straddle a split. FROZEN; re-deal only via --force, and never after a result has
             been reported on it. Sequences with no samples yet are dealt too, so the SILVER pour
             inherits splits instead of re-dealing.
          4b reference baseline on GOLD = a label SANITY CHECK (are the labels learnable, balanced,
             JAAD/PIE-comparable). QA only — the dataset is the product, not the model.
          4c snapshot — annotations + frozen splits + schema + docs into one checksummed,
             manifested bundle, whose README is the generated LABEL & TRACKING SUMMARY. INTERNAL:
             it pins reported numbers to one exact state of the data, it is not a release. No raw
             ZOD frames — samples ship relative pointers, so it stays megabytes and re-distributes
             none of ZOD's CC BY-SA data.
             → data/snapshots/zod-ped-v{version}/   (read with zodped.dataset.loader)
```

**One counting path.** `zodped.dataset.stats` computes the funnel, the composition, and the
per-stage provenance; `scripts/dataset_stats.py`, the snapshot manifest and `docs/LABEL_SUMMARY.md`
all render *that*, so no reported number can disagree with the artifacts.

**Consumer trap worth knowing.** A sample's `trajectory.frames` spans
`[window_start, window_end + max_horizon]` — most rows are the FUTURE, i.e. the trajectory-prediction
target. `loader.positions_array` defaults to `part="observed"` so the naive call cannot leak the
label; the raw JSON offers no such guard. (`in_observation` is unrelated — it flags a real detection
versus a coasted frame.)

**Sequencing.** Build vertically on GOLD first — against verified, keyframe-anchored tracks, a wrong
label is a labeling bug rather than track noise. Then pour SILVER through the same proven steps.
GOLD alone is already ≈JAAD/PIE scale, so scale is not the blocker; end-to-end validation is.

## Architecture — detector-as-measurement + KF/RTS linker

A **detector** localizes pedestrians per scan (the measurement); a **Kalman/RTS smoother** only links
those measurements into a track (the linker). The linker is settled; the measurement source pivoted
on 2026-06-18 (see "Direction & open options").

**Measurement — 2D → frustum lift.** A 2D detector boxes pedestrians on the camera stream; each box
is lifted to 3D via ZOD calibration + in-frustum LiDAR depth (2D box → in-frustum points →
nearest-depth slab). This localizes each frame *independently of any motion model*, so stop/start is
captured by detection rather than invented by the filter — exactly the moments intent labeling keys
off. The detector also supplies appearance features for later intent work.

**Linker — constant-velocity Kalman + RTS.** Associates measurements across scans (gating), coasts
through gaps, smooths backward. It links boxes into a track but is NOT the source of position during
normal tracking.

**Compensate first.** Every scan's points and detections are lifted to one world frame,
`p_world = pose(t) @ T_ego_lidar @ p_lidar`, where `T_ego_lidar = calib["FC"]["lidar_extrinsics"]`
and `pose(t)` is interpolated (SLERP + lerp) from `ego_motion["poses"]` at the scan timestamp — so
ego motion is never mistaken for pedestrian motion.

Per-scan: **detect + lift** → **compensate** → **predict** (to gate and to coast, not as position) →
**associate** (GOLD anchors on the keyframe box; SILVER seeds on the first confident detection; run
forward *and* backward from the seed) → **update** (store innovation as `kalman_confidence`) or
**coast** (`in_observation=false`, terminate after N misses) → **RTS smooth**.

A pedestrian moves <15 cm between LiDAR frames (~111 ms), so a small association gate suffices.

**Cross-check.** Projecting the 3D track back to camera and comparing to the 2D track is an
*agreement* metric, not accuracy — the keyframe boxes are the only validation set we have.

> **Set aside — gated-centroid tracker.** An earlier design used a CV Kalman gate whose
> centroid-of-enclosed-points was the measurement (no detector). Rejected: the CV model invents
> position and lags at stop/start, and the centroid is biased toward the LiDAR-facing surface.
> Removed from `tracker.py` 2026-06-24; recoverable from git history.

## Two-tier labels — keyframe coverage

ZOD annotates one keyframe per 20s clip, so a pedestrian present only before or after it carries no
ZOD annotation. We admit those pedestrians, in a separate quality tier so the verified set stays clean:

- **GOLD** — keyframe-anchored: verified ZOD identity + 3D box.
- **SILVER** — detector-found: no human verification. `is_in_gold_standard=false`,
  `label_confidence_tier=medium` (`low` reserved for a possible future demoted tier).

Tiers are set at Step 1 only. Everything downstream is tier-agnostic and carries the flag.

## Action label source

Geometry (feet-on-road) is the **acting** Step-2 label, not the truth. It is FOV/range-limited by
construction — the `ego_road` polygon exists only at the keyframe camera — and measured against a
human it is wrong on roughly a quarter of the crossings it declares, in both directions. So:

- **GOLD** — a human reviewed essentially every geometry-declared crosser. GOLD's labels are
  **human** (`02e_merge_human_labels.py`, precedence human > geometry). The question is closed there.
- **SILVER** — nobody will watch ~9,400 tracks, so SILVER keeps **geometry** labels and ships as
  *weak-labeled training bulk*: flagged `is_in_gold_standard=false`, `pv_disputed` retained as a
  counted quantity, and the measured GOLD error rate (28% false-positive on declared crossings)
  documented as expected label noise. **Never an evaluation set** — GOLD is the eval set.

**The model committee was tried and dropped (2026-08-06).** The plan was geometry + PV-LSTM,
auto-accepting where they agree. Zero-shot, PV-LSTM scores AUC **0.52** on the 217 human-verified
tracks — a coin flip on exactly the boundary cases a tie-breaker exists to arbitrate. Retrained on
ZOD it ranks far better (0.76), but **ranking is not labeling**: it declares crossings at 0.29
precision against geometry's 0.72, so relabeling with it would make SILVER worse. Dropped on both
counts. Evidence → EXPERIMENTS_LOG.

PV-LSTM (`vita-epfl/bounding-box-prediction`) remains useful for what it demonstrably does: **RANKING
which tracks a human should watch**, which is how the crosser review package was built. Two lessons
worth keeping: its intention head must be consumed as a SCORER (the released checkpoint is
conservative, so its 0.5 argmax is degenerate), and a model's AUC on a broad sample can be carried
almost entirely by easy negatives — always measure on the slice where the decision is actually hard.
PedGraph+ was evaluated and benched; TAMformer was never explored.

**Never fine-tune a member on our geometric labels** — that is circular, and the committee only has
value as an independent judge. Fine-tuning on *human-verified* tracks is legitimate, but must respect
the frozen splits (train-split sequences only).

**Domain gap to validate through** (the PointPillars lesson): JAAD/PIE are 30 fps unblurred dashcam;
ZOD is 10.1 Hz with anonymisation blur and a 120° lens. A 0.5s observation window is ~5 ZOD frames vs
their 15. Validate every checkpoint on its own dataset's test split before trusting it on ZOD.

## Direction & open options

The measurement source pivoted on 2026-06-18: off-the-shelf 3D detectors (PointPillars / OpenPCDet)
showed a severe ZOD domain gap and were dropped for the 2D→frustum measurement above. Evidence and
the recall numbers → EXPERIMENTS_LOG. Rejected code recoverable at git tag `experiments/3d-detectors`.

Still open:

1. **Modern 3D detector** (e.g. SAM4D) to add to or replace the frustum's 3D step.
2. **Fine-tune on hand-annotated ZOD sequences** if 2D→3D quality proves inadequate
   (supervisor-endorsed fallback; 2D expected good, 3D weaker).
3. **2D-first tracking (associate → *then* lift).** Today the linker lifts to 3D first, so identity
   is decided on the frustum-noisy *depth* axis while the reliable 2D box is discarded at the lift.
   2D-first associates in the image (IoU + optional appearance ReID), then lifts the finished
   tracklet. Fixes pass-by identity swaps and is **required for SILVER** (detector-birth is a 2D-MOT
   problem). Not a universal win: it doesn't fix frustum depth-jumps, can fragment anchor-threaded
   GOLD, and `camera_front_blur` weakens ReID.
4. **`ego_road` polygon extent** — whether opposing lanes and separated bike lanes count. A real
   share of geometry-vs-human disagreement is *definitional*, not perceptual. Cheaper to fix than
   any model change, and it shifts the crossing rate.

**Resolved: staying on ZOD** (2026-06-18). KITScenes, Waymo, nuScenes and NVIDIA PhysicalAI-AV were
re-evaluated; ZOD keeps its cam+LiDAR+radar novelty, and auto-labeling is a sound validated
methodology. Trade-off accepted: no GT trajectories → auto-generate + manual review.

## Schema, parameters & filters

Authoritative field-by-field spec → **`configs/dataset_schema_v0.2.yaml`**. Summary:

- **Windowing** — 0.5s observation window; prediction horizons 1.0 / 1.5 / 2.0s; TTE-anchored on `t_c`.
- **Sample unit** — per (pedestrian, window); `sample_id = {seq}_{ped}_{window_start_ms}`. Carries the
  forward-looking intent label, per-frame trajectory (world + ego-relative), multimodal file pointers,
  ego and pedestrian context, and the tier flags.
- **Filters by stage** — Step 1 selects pedestrians with a 3D keyframe box. Step 2 applies **no**
  filter (it labels every track). Step 3 applies the per-window gates from tracked positions:
  proximity (`distance_to_ego ≤ 50m`, `distance_to_road ≤ 15m`, at the window midpoint) and data
  availability (`min_lidar_frames_in_window`, camera frame present). Proximity is deferred to Step 3
  because the single keyframe annotation is a poor proxy for windows seconds away.
- **Output** — one JSON per sample + a Parquet index of all scalar fields for fast filtering.
