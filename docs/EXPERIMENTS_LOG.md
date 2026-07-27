# Experiments Log — trajectory detector bring-up gates

Durable record of the detector experiments run during Week 2 (2026-06-17/18). The raw
reports live under `data/processed/*.json` (gitignored, kept on disk); this file is the
committed summary. The dead 3D-detector code (OpenPCDet / PointPillars) was removed from
the working tree on 2026-06-18 — recover it from the `experiments/3d-detectors` tag.

## Why this log exists
After review (2026-06-18) the project **pivots away from off-the-shelf 3D detectors**. These numbers are the evidence for that decision. New direction:
1. Drop the old 3D detectors (PointPillars / OpenPCDet). If a 3D model is needed, use a modern one (e.g. SAM4D, 2025).
2. Keep the 2D+3D combination. Improve the **2D** front-end (e.g. Detectron2), then use ZOD calibration/projection to lift detections to 3D (the **frustum** approach below).
3. Expect 2D good, 3D weaker; if 3D is inadequate, hand-annotate a few sequences to fine-tune.

## Common protocol
- Same **1,830** keyframe 3D GT boxes (358 pedestrian seqs, 352 evaluated, 6 failed) for all 3D/frustum gates; 2D gate uses **2,159** GT 2D boxes (includes the 296 "2D-only" peds).
- 3D match: BEV centre, 2 m gate. 2D match: IoU ≥ 0.5. Score/conf threshold 0.1.
- "Compensated" = LiDAR motion-compensated to the camera keyframe instant (see `src/zodped/dataset/keyframe.py::motion_compensate_to_keyframe`); fixes a ~5 cm median/ ~19 cm p90 ego-motion bias from the +37 ms scan offset.

## Results

| Gate | Source | Recall | Precision | Notes |
|---|---|---:|---:|---|
| **3D — nuScenes PointPillars** | OpenPCDet PP-MultiHead, raw intensity 0-255, uncompensated | **0.485** | 0.085 | 888 TP / 9598 FP / 942 FN. <40 m 0.72–0.81; **>40 m → 0.24**. |
| **3D — KITTI PointPillars** | zhulf0804, `epoch_160`, intensity ×1/255, compensated | **0.114** | 0.229 | 208 TP / 700 FP / 1622 FN. 0–10 m 0.55 → **>40 m 0.008**; <40 m only 0.227. |
| **2D — YOLO11x** | imgsz 2560, conf 0.1, IoU≥0.5 | **0.629** | 0.645 | Occl: None 0.88 / Light 0.79 / Med 0.51 / Heavy 0.16. Height <40 px 0.33 → >320 px 0.86. 2D-only 0.30. |
| **Frustum (2D→3D)** | YOLO box → ZOD projection → LiDAR depth slab, compensated | **0.585** | 0.533 | Loc err median **0.147 m** / mean 0.254 / p90 0.515. Range 0–10 m 0.93 → >40 m 0.44. |

Frustum, uncompensated (kept for reference): recall 0.574 / prec 0.523 / loc median 0.199 m
/ p90 0.703 → compensation cut median −26 %, p90 −27 %.

## Conclusions
- **Off-the-shelf PointPillars has a severe ZOD domain gap**, worst beyond 40 m where **~52 % (949/1830) of GT lives**. KITTI weights (narrow HDL-64E front-FOV domain) are far worse than nuScenes weights. This is a transfer/domain-gap result.
- **Frustum (2D-driven) beats every off-the-shelf 3D detector** on recall, precision, and range — with zero training — and yields accurate 3D positions (~15 cm median). It is the realisation of the "good 2D → calibration/projection → 3D" direction and is **retained** as the Step 1 measurement front-end.
- 2D's main limiter is occlusion (single-frame; recoverable by the tracker/linker) and small far peds. Upgrading the 2D detector can directly lift the frustum ceiling.

## Pointers
- Surviving code: `scripts/bringup_frustum_poc.py`, `scripts/bringup_eval_detector2d_recall.py`, `scripts/bringup_validate_world_frame.py`, `src/zodped/dataset/keyframe.py`, `src/zodped/utils/projection.py`.
- Result notebook kept: `notebooks/02_bringup_gates.ipynb` (Gate A 2D recall + Gate B frustum POC).
- Removed (in git history @ `experiments/3d-detectors`): `scripts/eval_detector_recall.py`, `notebooks/02_detector_recall_results.ipynb`, `notebooks/05_pointpillars_zhu_results.ipynb`, and the external `~/OpenPCDet` / `~/PointPillars` clones.

---

## Boxfit cluster experiment (2026-06-26) — REJECTED, kept slab + anchor box

**Question.** Update #1 made the per-frame 3D box a required deliverable. Two open questions: (a)
should a LiDAR-cluster centroid replace the nearest-depth slab as the tracking measurement, and (b)
can a cluster supply the box size/orientation? Approach tried: in-frustum points → local ground removal → DBSCAN → nearest qualifying cluster → fit an oriented box; option
to feed the cluster centre back into the KF/RTS linker.

**Method.** A keyframe GT-box gate (the keyframe is the only frame with a GT 3D box) compared, per
GOLD pedestrian, the slab vs cluster centroid against GT, plus the fitted cluster box's 3D IoU /
height / yaw vs GT. 150 sequences, 384 cluster matches.

| metric | slab (baseline) | cluster |
|---|---:|---:|
| centre err xy — median | **0.146 m** | 0.154 m |
| centre err xy — mean | **0.232 m** | 1.304 m |
| centre err xy — p90 | **0.383 m** | 1.17 m |
| closer to GT | 51.6 % | 48.4 % |

Cluster box vs GT: 3D IoU median **0.098** (7 % ≥0.25, 0 % ≥0.5); height error median **−0.81 m**;
PCA yaw error median **63°**.

**Conclusion — REJECTED.** The cluster ties the slab on median but has heavy failure tail: for a sparse/occluded pedestrian the body fails to form a qualifying cluster
while a background blob in the frustum cone does, so "nearest cluster" jumps onto it. The raw cluster box is
unshippable — sparse points underestimate height by ~0.8 m and a pedestrian's round cross-section
makes PCA yaw ~random. So **the shipped GOLD box = tracked (slab) centre + keyframe anchor size +
velocity yaw** (`zodped.labeling.boxes`), and clustering is not in the product. The tail is a
fixable cluster-*selection* problem (anchor the pick to the slab depth), but probably not worth it. Revisit only for the SILVER tier, which has
no keyframe anchor and will need a size *prior* (measured extent alone is too short).

**Removed (recover from git history):** `src/zodped/labeling/boxfit.py`,
`scripts/bringup_boxfit_gate.py`, `scripts/viz_boxfit_diagnostic.py`. Kept and now in the product:
the Step 0 detection cache (`scripts/00_detect.py`, `src/zodped/labeling/detection_cache.py`) and
the box assembly (`src/zodped/labeling/boxes.py`).

**Note — the rejection is scoped to Regime A (detector HAS a box).** Two regimes: A) detector box →
slab wins (above); B) detector MISSES (occlusion) → no box, handled by KF coast + RTS smooth, not
LiDAR. Don't reach for clustering in B either: camera/LiDAR are co-mounted, so a hard occluder hides
the ped from both — the cone holds the *occluder*, and "nearest cluster" lands on it. Lesson:
robustness is the motion prior + gate, not the primitive. SILVER options, cheapest first:
re-acquisition (gate widen + appearance ReID, cf. OC-SORT), then gated cluster/scene-flow
segmentation (motion, not density; cf. FlowNet3D) — viable only because the track selects it.

## Step 2 corridor: straight strip → curved swept path (2026-07-01)

**Evidence.** seq 000041 (a 90° junction turn, ego heading −89° over the 8 s window): the straight
strip flagged **2/2** determined tracks as crossers; both are people standing on the pavement. The
curved swept path (ped world position projected onto the ego trajectory polyline; in-corridor when
the foot is `0..50 m` of arc-length ahead and within the half-width) flags **0/2**.

**Fix (SHIPPED).** `actions.py::_corridor_action` now uses the curved swept path. Dataset-wide the
corridor crossing count fell **185 → 38** (≈11.5 % → 2.4 % of determined) — the ~147 lost were
turn/heading artifacts. The 38 survivors are clean: median min-lateral **0.05 m** (all ≤ 1.5 m),
32/38 with a real detection at `t_c` and a left↔right side change (true traversal). A near-stationary
ego (`path < 2 m`) yields no swept path → no crossing. 

**Open follow-up.** 2.4 % is a small positive class; the broader `ego_road` set is 9.1 %. Confirm
with supervisor whether the crossing-prediction benchmark uses the strict corridor alone or unions
in `ego_road` (the union option was considered and deferred).

## Step 2 corridor BENCHED — crossing action = feet on ego road (2026-07-08)

**Decision** The crossing ACTION is redefined as *feet on the ego road*
(`crosses_ego_road`), not entry into the ego swept path (`crosses_ego_corridor`). Two consequences:

1. **Corridor is no longer a label.** We only have the `ego_road` polygon at the keyframe (image-
   pixel), so a road-membership test is FOV/range-limited — but that IS the
   agreed definition, and a camera model can emit it per-frame without us ever having per-frame road.
   The corridor's value (metric range-to-crossing, ego-relevance) is real but is a *feature*, not the label.
2. **Labeling pivots to model consensus.** The per-window crossing-action label (Step 3) moves from
   pure geometry to a local-model consensus labeler (geometry stays as one voter / the GT anchor).
   See `docs/JAAD_PIE_ALIGNMENT.md` and PIPELINE.md.
   *Correction 2026-07-15: this sentence conflated the two label levels — the consensus labels the
   track-level ACTION (crossed / not + when), not a per-window label; per-window intent is a
   separate later step (See PIPELINE.md "Action label source").*

**Change (SHIPPED).** `actions.py` is road-only; the corridor computation is preserved in `zodped.labeling.corridor` (benched). `02_label_action.py` and the QC viz drop corridor. Re-ran full GOLD: 1,863 tracks, 1,602 determined, **146 `ego_road` crossings (9.11 %)**,
0 failures — road numbers unchanged from before (only the corridor fields were removed). Record schema bumped `action/v0.2 → action/v0.3`.

## Tracker robustness: association in 3D vs 2D-first (2026-07-01, exploratory)

**Symptom.** The linker (`tracker.py`) lifts boxes to 3D then associates in the world frame, so
identity is decided on the *depth* axis (frustum-noisy) while the reliable 2D box position is thrown
away at the lift. Implausible speed "kicks" (>4 m/s, faster than a sprint = a filter kick, not real
motion) appear in **34/1545 (2 %)** GOLD tracks, concentrated in crowds; at the kick the nearest
*other* GOLD track is ≈0 m away = identity swap.

**Single-sequence A/B — 2D-IoU pre-gate** (before the existing 3D gate, keep only candidates whose 2D
box is IoU-consistent with the target's last-seen box):

| seq | scene | peak speed base→gate | recall | verdict |
|-----|-------|----------------------|--------|---------|
| 000903 | 5 peds, clean pass-by | 4.5 → 0.8 | 93→92 % | **fixed** |
| 000953 | 3 peds | 4.3 → 0.9 | 64→41 % | fixed kick, recall lost (stale last-box; needs box-forward predict) |
| 001396 | 4 peds | 4.1 → 4.1 | 96→96 % | unchanged — a frustum **depth-jump**, not a swap (needs anisotropic R) |
| 000603 | 34-ped crowd | 20→17 kicks | hurt | insufficient — boxes overlap each other → needs appearance **ReID** |

**Reading.** Association is the right lever for pass-bys; the *correct* architecture is **2D-first
(associate → lift)**, see PIPELINE "Direction & open options" #4. The pre-gate is a patch on the
current (backwards) ordering; the GOLD-appropriate form is anchor-seeded 2D-first. Depth-jumps and
crowds are separate problems (measurement noise; ReID). Full anchor-seeded 2D-first A/B on these four
regimes is pending before committing to a rerun (Step 1 rerun is minutes off the detector cache;
Step 2 is tier-agnostic and regenerates in minutes).

## Committee gate v1: zero-shot PV-LSTM on ZOD GOLD — FAILED (2026-07-16)

**Setup.** Released JAAD-trained PV-LSTM ckpt (validated on JAAD same day: intention AUC 0.77,
trajectory head at published ballpark — formatting proven). Swept over all 1,863 GOLD tracks on a
virtual 30 Hz timeline (RTS track sampled + 3D box projected per tick; ZOD px ×0.5 ≈ JAAD scale;
16-frame windows, stride 5). Track verdict vs Step-2 geometric anchor, threshold calibrated on the
frozen train split. `scripts/02b_committee_gate.py` → `reports/committee_gate_pvlstm.json`.

**Result: no usable zero-shot signal.** Track-score AUC vs anchor 0.55/0.53/0.60 (train/val/test;
was 0.77 on JAAD). Event-locked profile FLAT-to-inverted: crosser windows average 0.063–0.087
p(cross) approaching t_c vs 0.085 for non-crossers — the score does not rise before real crossings.
Not an aggregation artifact (max / mean / top-5 / p90 all AUC 0.42–0.56).

**Mechanism evidence.** Window score anti-correlates with box height (r=−0.22; 120px+ boxes —
77% crosser windows — mean score 0.039 vs 0.099 for <30px). The ckpt scores low on exactly the
big near-range boxes where ZOD crossings live. Points at input-distribution mismatch beyond lens
geometry alone: projected-3D boxes (looser, FOV-clipped) vs JAAD's tight detector boxes, fisheye
lateral geometry, ZOD image-velocity regime.

**Consequences.** Geometry remains the acting Step-2 label (as gated). Next iterations, cheapest
first: (a) feed Step-0 DETECTOR boxes (observed frames) instead of projected-3D boxes — closest
match to JAAD's box statistics; (b) undistort via Kannala→virtual-pinhole reprojection before
boxing; (c) parallel evidence from PedGraph+ (pose cue transfers differently); (d) retrain on JAAD
at matched 10 Hz/5-frame protocol (plan-B, doesn't fix geometry mismatch by itself). Gate harness
itself is DONE and cheap (~7 min full GOLD sweep) — iterate through it.

### Gate iteration (a): detector boxes — box style ACQUITTED (2026-07-16)

Same gate, windows built from matched Step-0 YOLO boxes (per-frame match: project track point,
pick containing detection; lerp 10 Hz→30 Hz, bridge ≤0.35 s; `--boxes detector`,
`reports/committee_gate_pvlstm_detector.json`). Detector boxes ARE visibly different (e.g. width
19 px vs projected 39 px on the same ped) and the score SCALE responds (max-F1 threshold 0.343 vs
0.101), but ranking does not: test AUC 0.587 vs 0.597, event-locked profile still flat-to-inverted
(0.084 at −3 s → 0.060 at t_c vs 0.085 non-crosser baseline). Also checked: track score vs ego
speed corr −0.05 — ego-motion streaming is NOT the confound. Cost: 423 mostly-far/occluded tracks
lose all windows to detector gaps (1,440 scored vs 1,863).

**Updated suspect list:** box style and ego-motion regime acquitted; remaining: (1) fisheye
lateral geometry, (2) POPULATION/LABEL-SEMANTICS mismatch — JAAD's cue is curbside near-range
"will step off the curb"; our anchor includes crossings far ahead and tracks already on the road —
the learned signature may not exist in our windows at all. Next probes ranked by information:
PedGraph+ second judge (different cue family; if pose ALSO fails ⇒ population mismatch, not model
choice), fisheye undistortion (half-day, lens-only), matched-protocol retrain (supervisor fork).

## Manual anchor batch: geometry vs human crossing labels (2026-07-27)

First human-labeled anchor over the frozen 20-sequence curated batch
(`data/processed/review/curation_worksheet.csv`, 87 tracks). Human crossing labels follow JAAD
"crossing the roadway" semantics (see `docs/JAAD_PIE_ALIGNMENT.md` §5). Compared against the Step-2
geometric `crosses_ego_road` label on the **35 GOLD kept tracks** (SILVER has no action records —
Step 2 ran GOLD-only). This is the reciprocal of the committee gate: instead of "does a model match
geometry," it asks "does geometry match a human."

**Agreement.** Confusion (human, machine): no/no 19, yes/yes 6, no/yes 6, yes/no 4.
- Agreement 25/35 = **71%**.
- Machine crossing **precision ~50%** (6 real of 12 called crossing), **recall ~60%** (6 of 10 human
  crossers caught).
- Crossing rate among kept, decided tracks: 11 yes / 41 no = **21%** (inside the JAAD/PIE band).

**Fails in both directions — the key result.**
- **6 false positives.** Four sequences tagged `crosser` (000940, 001084, 000035, 000067) contained
  *no* human-confirmed crossing at all. Geometry invented them.
- **4 false negatives.** Sequences 000952 and 000953, tagged `negative`, each held two real crossers
  geometry never saw.

**Root causes (two, partly separable).**
1. **FOV / range limit of the keyframe polygon.** `ego_road` exists only at the keyframe camera, so
   crossings far ahead or outside its view are invisible — the 952/953 misses. (Same limitation cited
   in PIPELINE.md "Action label source" as the reason for the model-consensus move.) Frustum depth
   smear is the likely FP mechanism: a sidewalk walker's lifted feet land on the road polygon.
2. **Road-extent / "lane" definition mismatch (annotator testimony).** A substantial share of the
   disagreement is *definitional*, not perceptual: geometry sometimes counts only the ego's own
   carriageway (not the opposing-direction lanes), and treats a separated bike lane inconsistently
   with the human. So yes/no and onset timing diverge on where "the road" begins, independent of
   whether the ped was seen clearly. **This part is fixable by redefining the `ego_road` polygon
   extent (opposing lanes in/out; explicit bike-lane policy) — separate from, and cheaper than, the
   committee.** It will also shift the crossing rate.

**t_c comparison: deferred.** Human onset values recorded (range −5.5 s to +6.3 s rel. keyframe, plus
`pre_obs` for already-crossing at first sight). Machine t_c extraction bugged in this pass (action
record keyframe key) — redo before quoting onset error.

**SILVER precision.** SILVER fate: 20 keep / 11 merge / 20 drop of 51 → only **39% survive as their
own new pedestrian**. GOLD survives 35/36. Cyclists and people-inside-cars (YOLO `person` FPs) are a
**SILVER-only** contaminant — GOLD is anchored on verified keyframe boxes, so they cannot enter it.
Implication for the pour: a cheap pre-filter (drop detections whose box lies inside a vehicle box;
drop cyclist overlaps) raises SILVER precision before QC and cuts manual load.

**Consequences.**
- Geometry is unfit as the *final* Step-2 label (~30% wrong vs human, both directions) — confirms the
  gate-v1 conclusion from the label side. It stays the acting label only.
- Do the road-extent redefinition (cause #2) before attributing more of the gap to the model — a
  chunk of the "committee vs geometry" disagreement may be this same definitional issue, not model
  skill. This double-attests the committee gate's "POPULATION/LABEL-SEMANTICS mismatch" suspect.
- Anchor positives are thin (11). Grow to ~30–50 hand-confirmed crossers — but NOT by seeding from
  the machine's `crosser` tag (4 of 8 crosser-tagged seqs held no real crossing).

### Human-anchor gate: PV-LSTM vs the curated labels (2026-07-27)

`scripts/02c_committee_gate_human.py` — same PV-LSTM scorer + window machinery as 02b, but truth =
the HUMAN `crossed_yes_no` in the curation worksheet (25 seqs: original 20 + 5 grow), BOTH tiers,
kept tracks only. AUC is the headline (pooled; ~20 positives is too few for a held-out split, and
AUC needs no threshold). Motivation: 02b's model-vs-geometry gate conflates model error with
geometry's ~30% error — checking against the human removes that.

**Result (83 kept+labeled tracks, 21 crossers / 62 non):**
| box source | n | crossers | AUC vs HUMAN | mean score cross/non |
|------------|----|----------|--------------|----------------------|
| projected3d | 83 | 21 | **0.678** | 0.227 / 0.164 |
| detector    | 77 | 20 | **0.736** | 0.228 / 0.148 |

Reference: PV-LSTM on JAAD's own test 0.77; the old ZOD model-vs-geometry gate 0.55.

**Two earlier conclusions corrected — both were artifacts of grading against geometry, not truth:**
1. **PV-LSTM is NOT near-random on ZOD.** Against the human it ranks a real crosser above a
   non-crosser ~68-74% of the time (0.55 → 0.68/0.74). Most of the "committee gate v1 FAILED"
   signal was geometry being wrong, not the model.
2. **Detector boxes DO help.** 02b found detector≈projected vs geometry (0.587 vs 0.597); against
   the human, detector boxes are clearly better (0.736 vs 0.678) — they match JAAD's box statistics,
   and that only shows up once the truth is correct. (Cost: 6 short tracks yield no detector window.)

Geometry-vs-human on the shared gold tracks reproduces the anchor batch: ~0.72-0.74 agreement.

**Honest caveat:** ~20 crossers, pooled, optimistic (same-set) threshold → the AUC is a rough,
encouraging read with a wide error bar, NOT a validated pass. Direction is trustworthy (consistent
across both box types, clean mean-score separation); the absolute number is not precise. Next levers
unchanged (undistortion, PedGraph second judge, matched retrain), but detector boxes are now the
default input for PV-LSTM.
