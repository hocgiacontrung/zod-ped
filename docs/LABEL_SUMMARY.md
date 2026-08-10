# zod-ped — Label & Tracking Summary  (snapshot v0.2)

> **INTERNAL — NOT FOR DISTRIBUTION.** The labels are auto-generated and only partially
> human-verified. This snapshot exists to pin a set of numbers to one exact state of the data, so
> results stay reproducible while the work continues. It is not a public dataset release, and the
> limitations below are the reason.

Multimodal pedestrian intent & trajectory dataset built on the Zenseact Open Dataset (ZOD). Camera + LiDAR + radar, where JAAD/PIE/PSI are camera-only.

Built 2026-08-08T19:06:28+00:00 from commit `7dca4f395f8c` (tree dirty at build time).
Schema: `schema/dataset_schema_v0.2.yaml`.

## Read this first — what state the labels are actually in

1. **Only 217 tracks of 9247 have been watched by a human.** Everything else carries a geometric rule's verdict.
2. **On those reviewed tracks, human review flipped 56 labels (26%)** — 41 declared crossings that were not, and 15 crossings the rule missed. That is the measured error rate of the automatic labels, and SILVER carries it entirely.
3. **492 GOLD tracks flagged as disputed were never reviewed.**
4. **SILVER's accuracy has never been measured directly** — too few SILVER tracks carry a human label. Its error rate is GOLD's, carried over on the assumption the rule behaves the same; SILVER's tracks are farther and noisier, so that assumption is optimistic.
5. **The evaluation set is small** — GOLD test holds 35 positive windows out of 200. Differences of a few points between models mean nothing at that size.
6. **The labels are contested even among humans** — two reviewers agreed on ~86% of the verified tracks, so this task has no clean ceiling.

Nothing here is a surprise or a regression; it is the honest accounting, and it is why this is a
snapshot rather than a release.

## What was done to the data

| stage | in | out | what we know about the error |
|---|---|---|---|
| 1 · tracking | 358 sequences | 11257 tracks (1863 GOLD + 9394 SILVER) | 74% clean, 17% marginal, 9% bad |
| 2 · action (geometry) | 9247 pedestrians | 367 declared crossers (GOLD 9.1%, SILVER 3.0%) | 261 GOLD tracks undetermined (kept, never forced) |
| 2e · human review (GOLD) | 217 tracks watched | 119 crossing / 97 not | **56 labels flipped** (41 cross→no, 15 no→cross) = geometry's measured error |
| 3 · intent windows | 4285 pedestrians | 4449 samples (270 TTE-anchored, 4179 comparison) | labels forward-looking; filters applied per window |
| 4b · sanity check | 740 train windows | GOLD test AUC **0.7635** ± 0.0251 | AP 0.3753 vs chance 0.1759 — labels learnable |

Design rationale for each stage → `docs/PIPELINE.md`. Dated evidence and the rejected
alternatives → `docs/EXPERIMENTS_LOG.md`. Detector and frustum bring-up numbers live only in the
log, deliberately, so they have exactly one home.

## What a sample is

One sample is a **(pedestrian, time window)** pair: a 0.5s observation window over one tracked
pedestrian, carrying a forward-looking intent label.

**Action is not intent.** `action.crosses_ego_road` is a verdict about the *whole track* — did
this person cross the ego road, and when. `intent.labels_by_horizon` is *per window* and looks
*forward* — will crossing start within the horizon **after** the window ends. Training on the
action field turns intent prediction into action detection and breaks comparability with JAAD/PIE.

Labels are provided at three prediction horizons: 1.0, 1.5, 2.0 seconds.

## Contents

```
annotations/     4449 sample JSONs + dataset_index.parquet (one row per sample)
splits/          FROZEN sequence-level train/val/test mapping
schema/          authoritative field-by-field spec
docs/            data format, pipeline design, experiments log, JAAD/PIE alignment
manifest.json    build provenance + full composition at every horizon
CHECKSUMS.sha256  SHA-256 of every file above
```

**Raw sensor data is not included.** Samples point at ZOD camera/LiDAR/radar files by path
relative to the sequence directory. Download ZOD separately and resolve them with
`zodped.dataset.loader.media_paths`.

## Composition (horizon 2.0s)

4449 samples over 4285 pedestrians in 286 sequences.

| tier | split | samples | pedestrians | crossers | crossing ratio |
|---|---|---:|---:|---:|---:|
| GOLD | test | 200 | 178 | 35 | 0.175 |
| GOLD | train | 743 | 662 | 127 | 0.171 |
| GOLD | val | 100 | 89 | 18 | 0.180 |
| SILVER | test | 683 | 679 | 10 | 0.015 |
| SILVER | train | 2331 | 2297 | 59 | 0.025 |
| SILVER | val | 392 | 380 | 21 | 0.054 |

### Read this before quoting a number

**GOLD and SILVER are reported separately, always.**

| tier | what it is | samples | crossing ratio |
|---|---|---:|---:|
| **GOLD** | human-verified labels — **the evaluation set** | 1043 | **0.173** |
| **SILVER** | geometry labels, weak — training bulk, never evaluation | 3406 | 0.026 |

SILVER's crossing rate is far lower by construction, not by error: ZOD annotates only pedestrians
a human judged relevant, and relevant overwhelmingly means near the road, while the detector finds
everyone else — far, peripheral, on the pavement. A combined ratio therefore describes neither
tier. Every index row carries `is_in_gold_standard`, so the two are always separable.

## Loading

```python
from zodped.dataset.loader import (load_index, select, load_sample,
                                   boxes_array, positions_array, intent_label)

index = load_index("annotations")
train = select(index, split="train", tier="gold")

doc = load_sample(train.sample_id.iloc[0], "annotations")
boxes = boxes_array(doc)                       # (T, 4) pixel xyxy, NaN where unobserved
past  = positions_array(doc)                   # (T, 3) world xyz INSIDE the window  -> input
future = positions_array(doc, part="future")   # after the window                    -> target
label = intent_label(doc, horizon="2.0")

# Raw sensors, if you have ZOD on disk:
from zodped.dataset.loader import media_paths, load_radar_window
paths = media_paths(doc, "data/raw/sequences")
radar = load_radar_window(doc, "data/raw/sequences")   # returns inside this window
```

> **`trajectory.frames` is not the observation window.** It spans
> `[window_start, window_end + max_horizon]`, so most of its rows lie *after* the window — they
> are the trajectory-prediction target. Passing them to a model as input leaks the answer.
> `positions_array` defaults to `part="observed"` for exactly this reason; the raw JSON gives you
> no such protection. (The per-frame `in_observation` flag is unrelated — it means the tracker had
> a real detection rather than coasting.) `boxes_array` is window-only and always safe.

Splits are **sequence-level and frozen** — windows from one sequence share frames, pedestrians and
scene, so a sample-level split would leak near-duplicates into test. Use the shipped `split`
column; do not re-deal it.

## Where the pedestrians went

Funnel: 11257 track files →
9247 pedestrians after cut + stitch →
4449 samples. Step 3 dropped the rest through explicit per-window gates:

| gate | windows dropped | share |
|---|---:|---:|
| `distance_to_ego` | 2201 | 40% |
| `distance_to_road` | 1165 | 21% |
| `comparison_window_unobserved` | 1133 | 21% |
| `window_outside_track` | 675 | 12% |
| `undetermined_action` | 274 | 5% |
| `no_track_frames` | 6 | 0% |
| **total** | **5454** | |

The `distance_to_ego` gate is the largest single filter and is deliberate — it is a class-balance
filter, not a data-availability one. Relaxing it makes the crossing ratio *worse*, because the
pedestrians it admits are far ones who never cross. Measured; see EXPERIMENTS_LOG.

## Other caveats

Beyond the label-quality accounting at the top:

1. **Geometry is FOV- and range-limited by construction.** The `ego_road` polygon exists only at
   the keyframe camera, so crossings outside that view cannot be seen by the rule at all.
2. **Radar is per-sequence, not per-window.** ZOD ships one structured `.npy` per sequence holding
   every sweep; `radar_path` points at that blob. Use `loader.load_radar_window` to slice it.
3. **`num_pedestrians_in_scene` / `is_key_pedestrian` are population-dependent.** The values here
   come from the full-population pass and are the accurate ones, but re-running the pipeline over
   a single tier would recompute them differently. Known wart, not yet fixed.
4. **The model committee was tried and dropped.** PV-LSTM ranks review candidates well but decides
   badly (0.29 precision against geometry's 0.72), so it labels nothing. Evidence in
   EXPERIMENTS_LOG.

## Provenance and licence

Underlying sensor data is the **Zenseact Open Dataset**, CC BY-SA 4.0, obtained separately from
<https://zod.zenseact.com/> under its own terms. This snapshot redistributes no ZOD data — only
annotations and relative pointers. Code: MIT.

Produced at the Intelligent Robotics Lab, Aalto University.
