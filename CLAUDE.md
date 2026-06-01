# ZOD Pedestrian Intent & Trajectory Dataset

Internship project at Intelligent Robotics Lab, Aalto University.
Conda env: `zod-iac` | Server: `user20@aalto`

## Goal
Build a multimodal pedestrian intent & trajectory prediction dataset on top of
Zenseact Open Dataset (ZOD). Key novelty: synchronized camera + LiDAR + radar,
unlike existing pedestrian intent datasets (JAAD, PIE, PSI) which are camera-only.
Schema approved. Now building the pipeline.

## Current Status (Week 2)
- [x] 358 sequences with pedestrian annotations identified → `data/splits/pedestrian_sequences.json`
- [x] 130 of those have LiDAR available (seq 000000–000479) → working set
- [x] Dataset schema approved by supervisor → `docs/PIPELINE.md`
- [ ] **Next: exploration notebook on sequence 000007** (see Current Task below)
- [ ] Filtering script (Step 1)
- [ ] Trajectory generation (Step 2)
- [ ] Intent labeling (Step 3)

## Current Task
Write `notebooks/01_explore_sequence.ipynb` on sequence `000007`.
Must verify before writing any pipeline code:
1. Load `object_detection.json` → confirm pedestrian fields and coordinate frame
2. Timestamp matching: find nearest LiDAR file to keyframe timestamp (gap ≤55ms)
3. LiDAR point cloud: load `.npy`, confirm points land inside pedestrian 3D box
4. `ego_road.json`: confirm polygon coordinate frame matches pedestrian location_3d
5. Bird's-eye plot: LiDAR scan + pedestrian centroid + road polygon in same frame

If points don't land in the bounding box → coordinate frame mismatch → check
`calibration.json` or use the ZOD devkit (`from zod import ZodSequences`).

## Key Constraints
- LiDAR files named by UTC timestamp, not frame index — always match by timestamp
- Annotations exist only at 1 keyframe per sequence (central frame of 20s clip)
- `pedestrian_sequences.json` entries: `{seq_id, num_pedestrians, lidar_batch}`
- Do NOT use ZodSequences API for full dataset — load JSON files directly
  (trainval-sequences-full.json not available for partial downloads)
- No budget for VLMs — use Gemini 1.5 Flash free tier or local Llama 3.2 Vision 11B

## Project Layout
```
zod-ped/
├── data/
│   ├── raw/sequences/XXXXXX/     ← ZOD data (annotations, lidar, images, etc.)
│   ├── processed/                ← pipeline outputs
│   ├── annotations/              ← generated pseudo-labels
│   └── splits/
│       └── pedestrian_sequences.json
├── src/
│   ├── dataset/                  ← ZOD loading & data structures
│   ├── labeling/                 ← filtering, tracking, intent labeling
│   ├── utils/                    ← geometry, transforms, I/O helpers
│   └── visualization/
├── scripts/
├── notebooks/
├── configs/                      ← pipeline parameters (YAML)
└── docs/
    ├── DATA_FORMAT.md            ← sensor specs, file formats, annotation fields
    └── PIPELINE.md               ← pipeline steps, schema, labeling strategy
```

## Reference Docs
- Sensor specs, file formats → `docs/DATA_FORMAT.md`
- Pipeline design, schema summary → `docs/PIPELINE.md`
- Full schema spec (source of truth) → `docs/dataset_schema_v0_1.yaml`
