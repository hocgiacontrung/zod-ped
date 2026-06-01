# Data Format Reference

## Sensor Overview

| Sensor   | Freq     | Format              | Notes |
|----------|----------|---------------------|-------|
| Camera   | 10.1 Hz  | `.jpg` (blur/)      | 8MP, 120° FOV, faces/plates blurred |
| LiDAR    | ~9 Hz    | `.npy` per scan     | 3 sensors merged, ~254k pts, up to 245m |
| Radar    | ~17 Hz   | `.npy`              | Range rate (Doppler velocity) available |
| GNSS/IMU | 100 Hz   | `.hdf5`             | 0.01m accuracy, ego-motion source |

## Sequence Directory Layout
```
data/raw/sequences/XXXXXX/
├── annotations/
│   ├── object_detection.json   ← pedestrian 3D boxes (keyframe only)
│   ├── ego_road.json           ← road polygon
│   ├── lane_marking.json
│   └── traffic_signs.json
├── lidar_velodyne/             ← .npy files named by UTC timestamp
├── images_blur/                ← .jpg camera frames
├── calibration.json            ← sensor extrinsics/intrinsics
├── ego_motion.json             ← vehicle poses over time
├── info.json
└── metadata.json
```

## LiDAR Files

**Naming:** UTC timestamp, not frame index
```
000007_quebec_2022-02-14T13:23:32.251875Z.npy
```
Parse timestamp: strip `.npy`, replace trailing `Z` with `+00:00`, use `datetime.fromisoformat()`.

**Array columns:** `[timestamp, x, y, z, intensity, diode_index]`
- `diode_index` 0–127: VLS128 main sensor
- `diode_index` 128–143: left VLP16
- `diode_index` 144–159: right VLP16
- Coordinates in **sensor/ego frame** (meters)

**Synchronization:** For each camera frame, find nearest LiDAR scan by timestamp.
Max acceptable gap: **55ms**. Use `ego_motion.json` to compensate movement between frames.

## object_detection.json

```json
[
  {
    "geometry": {
      "type": "MultiPoint",
      "coordinates": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  // 2D image pixel bbox
    },
    "properties": {
      "annotation_uuid": "...",
      "class": "Pedestrian",          // capital P; also Vehicle, VulnerableVehicle, Animal
      "occlusion_ratio": "None | Light | Heavy",
      "orientation_3d_qw": 0.99,
      "orientation_3d_qx": 0.0,
      "orientation_3d_qy": 0.0,
      "orientation_3d_qz": 0.1,
      "size_3d_length": 0.8,          // meters
      "size_3d_width":  0.5,
      "size_3d_height": 1.8,
      "location_3d": {
        "type": "Point",
        "coordinates": [x, y, z]      // sensor/ego frame at keyframe timestamp
      }
    }
  }
]
```
Filter pedestrians: `a["properties"]["class"] == "Pedestrian"`

## ego_motion.json
Contains vehicle poses over the full 20s sequence. Used for:
- Getting the keyframe timestamp
- Ego-motion compensation when transforming LiDAR points to world frame

## calibration.json
Sensor extrinsic/intrinsic transforms. Required to verify LiDAR points land in
3D bounding boxes. If axis-aligned box check returns 0 points, apply the
LiDAR→annotation frame transform from this file before checking.

## pedestrian_sequences.json
```json
[
  {"seq_id": "000007", "num_pedestrians": 10, "lidar_batch": "lidar_velodyne_000000_000039.tar.gz"},
  ...
]
```
358 total entries. LiDAR available for seq 000000–000479 (130 sequences with pedestrians).
Working set = entries where `lidar_batch` is one of the 12 downloaded batches.
