# ZOD Pedestrian Intent & Trajectory Dataset

Internship project at Intelligent Robotics Lab, Aalto University.
Conda env: `zod-iac` | Server: `user20@aalto`

## Working Preferences
- Pipeline decisions are open to debate. If you see a better approach, challenge it — propose alternatives, state the trade-off.
- **Don't duplicate.** One fact lives in one place. Before adding a script or an output file, check what exists and extend it instead. Docs cross-reference; they don't repeat.

## Goal
A multimodal pedestrian intent & trajectory dataset on ZOD. Novelty: synchronized camera + LiDAR +
radar, where JAAD/PIE/PSI are camera-only.

**Action ≠ intent.** Action = did this pedestrian cross, and when (whole track, Step 2). Intent =
will they start crossing within the horizon (per window, Step 3). Never conflate them.

## Current Status (Week 8, 2026-08-06)

Working set: **358 sequences** with pedestrian annotations + LiDAR on disk.
Two tiers, set at Step 1 only; Steps 2–4 are tier-agnostic and carry `is_in_gold_standard`.

| step | script | state |
|---|---|---|
| 0 · detection cache | `00_detect.py` | DONE — all 358 seqs |
| 1a · GOLD trajectories | `01_generate_trajectories.py` | DONE — **1,863 tracks** |
| 1b · SILVER trajectories | `01b_generate_silver.py` | DONE — 9,394 tracks → **7,384 peds** after cut+stitch |
| 2 · action labels (geometry) | `02_label_action.py` | DONE — **9,247 pedestrians** (`--stitch` for SILVER) |
| 2e · human merge (GOLD) | `02e_merge_human_labels.py` | DONE — 217 human-verified → `actions_verified/` |
| 3 · samples + intent | `03_assemble_samples.py` | DONE both tiers — **4,449 samples / 4,285 peds** |
| 4a · splits | `04_assign_splits.py` | **FROZEN** — inherited by the pour (3,074/492/883) |
| 4b · reference baseline | `04b_train_baseline.py` | DONE — GOLD test **AUC 0.76**, labels learnable |
| 4c · snapshot + summary | `05_package_snapshot.py` | DONE — checksummed INTERNAL snapshot; README = `docs/LABEL_SUMMARY.md` |

**Next:** release notes + final presentation. The label sanity check PASSED — a PV-LSTM retrained on
our data scores **AUC 0.763 ± 0.025 / AP 0.375** on human-verified GOLD test (chance 0.50 / 0.18),
up from 0.52 zero-shot. Headline is the GOLD-only arm: adding SILVER to training measured +0.036 AUC,
which is inside the noise (3/5 seeds) — SILVER does not poison, but it is not shown to help.

**Numbers: run `python scripts/dataset_stats.py`.** It reads the artifacts, so it is never stale.
Any doc that disagrees with it is wrong, including this one.

**Report the tiers SEPARATELY — never a combined crossing ratio.** GOLD is the eval set (1,043
samples, **17.3%** @2.0s, human-verified). SILVER is training bulk (3,406 samples, **2.6%**, geometry
labels, ~28% expected noise). SILVER's rate is 6.6× lower because ZOD only annotates pedestrians a
human judged relevant, and relevant means near the road; the detector finds everyone else. Every
index row carries `is_in_gold_standard`, so the split is always available.

**Committee is OFF (2026-08-06), and training does not revive it.** Zero-shot PV-LSTM is a coin flip
on the hard cases (AUC 0.52 on the 217 verified tracks). Retrained on ZOD it ranks well (0.76) — but
**ranking is not labeling**: it declares crossings at 0.29 precision against geometry's 0.72, so
relabeling with it would make SILVER worse. SILVER keeps geometry labels, flagged weak, never used
for eval. PV-LSTM stays a review RANKER only. Evidence → EXPERIMENTS_LOG.

**Open:** GOLD's 17.3% @2.0s sits below the 20–30% target band — the human pass removed geometry's
false crossings, so this is the honest number, and counts were approved 2026-08-06. Two things NOT to
try: relaxing the 50m `distance_to_ego` gate (it is a class-balance filter — opening it makes the
ratio *worse*), and pouring more SILVER (it dilutes, 2.6%). Both measured — EXPERIMENTS_LOG. The one
untried lever is chasing geometry's misses in the geo=no/PV=no slice nobody has sampled.

**Known wart:** `num_pedestrians_in_scene` / `is_key_pedestrian` depend on which `--tier` a Step-3 run
loads. Shipped values are from the `--tier all` pass (the accurate ones); make them
population-independent before anyone re-runs a single tier.

Details and evidence for every line above → `docs/EXPERIMENTS_LOG.md`.

## Key Gotchas (verified, seq 000007)
**`location_3d` is in the LiDAR sensor frame** — so `ego_road.json` (image-pixel) is not
3D-comparable; project first. Full set — frame conventions, structured-array `.npy`, per-point µs
timestamp offset, 2D-only guard, 55ms scan-gap limit → `docs/DATA_FORMAT.md`.

## Key Constraints
- LiDAR files are named by UTC timestamp, not frame index — always match by timestamp
- ZOD annotates only 1 keyframe per sequence (central frame of the 20s clip)
- Do NOT use the `ZodSequences` loader — read the JSON files directly (the full trainval index is
  unavailable for partial downloads). The rest of the devkit is installed and fine to use
  (e.g. `project_3d_to_2d_kannala`)
- No budget for VLMs — local open-source models only
- Never run JAAD's `split_clips_to_frames.sh` (169GB; disk won't fit — decode on the fly)

## Project Layout
```
zod-ped/
├── data/
│   ├── raw/sequences/XXXXXX/   ← ZOD data (annotations, lidar, images)
│   ├── processed/              ← pipeline outputs (subdirs only; no loose files)
│   │   ├── detections/  pose/         ← Step 0
│   │   ├── trajectories/              ← Step 1 (GOLD + SILVER, one dir)
│   │   ├── actions/                   ← Step 2 geometry labels (both tiers)
│   │   ├── actions_verified/          ← Step 2e human layer (GOLD; Step 3 reads this)
│   │   ├── review/                    ← worksheets, queues, focus clips
│   │   ├── reports/                   ← run reports
│   │   └── splits/                    ← Step 4a FROZEN sequence_splits.json
│   ├── annotations/            ← Step 3 per-sample JSON + dataset_index.parquet
│   ├── snapshots/              ← Step 4c bundles (gitignored; rebuild with 05_package_snapshot.py)
│   ├── external/JAAD/          ← ykotseruba/JAAD clone + JAAD_clips/
│   └── pedestrian_sequences.json
├── src/zodped/                 ← the importable library (`pip install -e . --no-deps`)
│   ├── dataset/                ← keyframe.py, splits.py, stats.py, loader.py, packaging.py
│   ├── labeling/               ← detector, detection_cache, frustum, tracker, boxes, committee
│   └── utils/                  ← projection.py, ego_motion.py, vehicle_data.py, video.py
├── scripts/                    ← entry-points (thin: argparse + I/O + calls into zodped)
│   └── _common.py              ← shared paths, frustum-pool args, --tier filter
├── notebooks/  configs/  docs/
```

## Reference Docs
| doc | what lives there |
|---|---|
| `docs/DATA_FORMAT.md` | sensor specs, file formats, frame conventions |
| `docs/PIPELINE.md` | pipeline design, architecture, schema summary, open options |
| `docs/EXPERIMENTS_LOG.md` | dated evidence — every result and decision, with numbers |
| `docs/JAAD_PIE_ALIGNMENT.md` | JAAD/PIE taxonomy, curation rules, open naming decisions |
| `docs/LABEL_SUMMARY.md` | GENERATED per-stage label & tracking summary — the report backbone |
| `configs/dataset_schema_v0.2.yaml` | authoritative field-by-field spec (the only one; `v0.1` retired) |
