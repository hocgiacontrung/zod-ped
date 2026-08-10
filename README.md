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

**Not a public release.** The labels are auto-generated and only partially human-verified; the
snapshot exists to pin reported numbers to one exact, checkable state of the data. Its README is
the generated **label & tracking summary** — what each stage did and what is known about its error
(`docs/LABEL_SUMMARY.md`).

The bundle carries annotations, frozen splits, schema, reference docs, a manifest, and SHA-256 over
every file. Raw ZOD frames are never copied — samples hold relative pointers, resolved with
`zodped.dataset.loader.media_paths`. Read it with `zodped.dataset.loader`, which works on the
bundle or on `data/annotations/` unchanged.

## Setup

```bash
conda activate zod-iac
pip install -r requirements.txt
pip install -e . --no-deps    # register the `zodped` library (editable; no sys.path hacks)
```

Data must be downloaded separately via the ZOD CLI. See `docs/DATA_FORMAT.md`.

## Project Structure

```
src/zodped/   importable library (pip install -e .): dataset, labeling, utils
scripts/      runnable entry-points (pipeline steps, viz/demo, bring-up gates)
notebooks/    exploration and analysis
configs/      pipeline parameters + dataset schema
docs/         data format, pipeline design, experiments log, JAAD/PIE alignment
```

## License

Code: MIT  
Dataset (ZOD): CC BY-SA 4.0
