# ZOD Pedestrian Intent & Trajectory Dataset

> Internship project — Intelligent Robotics Lab, Aalto University

## Overview

A multimodal pedestrian intent and trajectory prediction dataset built on top of the
[Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/). Unlike existing pedestrian
intent datasets (JAAD, PIE, PSI) which are camera-only, this dataset includes
synchronized **camera + LiDAR + radar** data across 14 European countries.

**Status:** both tiers built — 4,449 samples over 4,285 pedestrians, GOLD human-verified,
splits frozen, reference baseline passing (Week 8). Current numbers:
`python scripts/dataset_stats.py`. Step-by-step state → `CLAUDE.md`.

## Dataset Foundation

Built on ZOD Sequences (1,473 × ~20s clips). ZOD annotates one keyframe per sequence;
this project generates pseudo-labels for pedestrian trajectory and crossing intent across
all frames, in two quality tiers. Working set: 358 sequences with pedestrian annotations
+ LiDAR on disk.

## Using the dataset

```bash
python scripts/05_package_snapshot.py --summary-to docs/LABEL_SUMMARY.md
python scripts/05_package_snapshot.py --verify data/snapshots/zod-ped-v0.2
```

**Not a public release** — the labels are auto-generated and only partially human-verified. The
snapshot pins reported numbers to one exact, checkable state of the data. What each stage did, what
is known about its error, and how to read the bundle → `docs/LABEL_SUMMARY.md`, which is also the
bundle's own README. Load it with `zodped.dataset.loader`, which works on the bundle or on
`data/annotations/` unchanged.

## Setup

```bash
conda activate zod-iac
pip install -r requirements.txt
pip install -e . --no-deps    # register the `zodped` library (editable; no sys.path hacks)
```

Data must be downloaded separately via the ZOD CLI. See `docs/DATA_FORMAT.md`.

New to the repo? → `docs/HANDOVER.md` (read order, exact commands to re-run the pipeline).

## Project Structure

```
src/zodped/   importable library (pip install -e .): dataset, labeling, utils
scripts/      runnable entry-points: pipeline steps, QC, visualisation
notebooks/    exploration and analysis
configs/      pipeline parameters + dataset schema
docs/         handover guide, data format, pipeline design, experiments log, label summary
```

## License

Code: MIT  
Dataset (ZOD): CC BY-SA 4.0
