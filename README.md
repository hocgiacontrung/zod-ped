# ZOD Pedestrian Intent & Trajectory Dataset

> Internship project — Intelligent Robotics Lab, Aalto University

## Overview

A multimodal pedestrian intent and trajectory prediction dataset built on top of the
[Zenseact Open Dataset (ZOD)](https://zod.zenseact.com/). Unlike existing pedestrian
intent datasets (JAAD, PIE, PSI) which are camera-only, this dataset includes
synchronized **camera + LiDAR + radar** data across 14 European countries.

**Status:** Pipeline in development (Week 2)

## Dataset Foundation

Built on ZOD Sequences (1,473 × ~20s clips). Annotations exist at one keyframe per
sequence; this project generates pseudo-labels for pedestrian intent and trajectory
across all frames. Working set: 130 sequences with pedestrian annotations + LiDAR.

## Setup

```bash
conda activate zod-iac
pip install -r requirements.txt
```

Data must be downloaded separately via the ZOD CLI. See `docs/DATA_FORMAT.md`.

## Project Structure

```
src/          pipeline source code
scripts/      one-off utility scripts
notebooks/    exploration and analysis
configs/      pipeline parameters
docs/         data format reference, pipeline design, schema
```

## License

Code: MIT  
Dataset (ZOD): CC BY-SA 4.0
