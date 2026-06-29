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
4. Not locked to ZOD — evaluate other datasets if better/easier/better-annotated.

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
- **Off-the-shelf PointPillars (either weights) has a severe ZOD domain gap**, worst beyond 40 m where **~52 % (949/1830) of GT lives**. KITTI weights (narrow HDL-64E front-FOV domain) are far worse than nuScenes weights. This is a transfer/domain-gap result.
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

**Conclusion — REJECTED.** The cluster ties the slab on the median but has a heavy failure tail
(mean/p90 blow up): for a sparse/occluded pedestrian the body fails to form a qualifying cluster
while a background blob in the frustum cone does, so "nearest cluster" jumps onto it. The raw cluster box is
unshippable — sparse points underestimate height by ~0.8 m and a pedestrian's round cross-section
makes PCA yaw ~random. So **the shipped GOLD box = tracked (slab) centre + keyframe anchor size +
velocity yaw** (`zodped.labeling.boxes`), and clustering is not in the product. The tail is a
fixable cluster-*selection* problem (anchor the pick to the slab depth), but even fixed the cluster
only ties the slab, so it was not worth the complexity. Revisit only for the SILVER tier, which has
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
