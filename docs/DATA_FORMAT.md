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
├── camera_front_blur/          ← .jpg camera frames
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

**Format:** structured numpy array (named fields, not positional columns). Load with:
```python
cloud = np.load("scan.npy")
x, y, z = cloud["x"], cloud["y"], cloud["z"]   # NOT cloud[:, 1] etc.
```

**Fields:** `('x', 'y', 'z', 'timestamp', 'intensity', 'diode_index')`
- `x`, `y`, `z`: float32, ego/sensor frame (metres). x forward, y left, z up.
- `timestamp`: int64, **relative microsecond offset** from the scan's center time
  (encoded in the filename). To get absolute UTC per point:
  `abs_ts = filename_unix_ts + cloud["timestamp"] / 1e6`
- `intensity`: uint8, 0–255
- `diode_index`: uint8 — 1–127: VLS128 main sensor; 128–143: left VLP16; 144–159: right VLP16

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

**Coordinate frame of `location_3d`:** **LiDAR sensor frame** (same as `.npy` point cloud
coordinates), NOT the vehicle ego frame. Despite the ZOD docs saying "sensor/ego frame",
these are meaningfully different due to the LiDAR mount transform (~1.75m above vehicle,
slight rotation). Verified on seq 000007: projecting via `inv(cam_ext) @ lid_ext` puts
centroids within ±35px of annotated 2D bbox centers.

**Note:** Not all pedestrian annotations have a 3D box. Some are 2D-only (no `location_3d`,
no size/orientation fields). Always guard: `"location_3d" in a["properties"]` before
accessing any 3D field.

## ego_motion.json
Contains vehicle poses over the full 20s sequence. Used for:
- Getting the keyframe timestamp
- Ego-motion compensation when transforming LiDAR points to world frame

## ego_road.json
Road polygon annotations. **Coordinate frame: image pixel space** (not ego/3D).
Polygon vertices are (u, v) pixel coordinates in the front camera image (3848×2168).
To check whether a pedestrian is on the road, project `location_3d` into image space
first using the FC camera model, then test containment in the polygon.

## calibration.json
Sensor extrinsic/intrinsic transforms. Top-level key is `"FC"` (front camera).
Relevant sub-keys:
- `intrinsics`: 3×4 matrix (Kannala fisheye)
- `extrinsics`: 4×4 ego→camera transform
- `lidar_extrinsics`: 4×4 LiDAR→camera transform
- `distortion` / `undistortion`: Kannala distortion coefficients
- `image_dimensions`: [3848, 2168]

LiDAR points and `location_3d` are confirmed to be in the same **LiDAR sensor frame**
(verified on seq 000007) — no extra transform needed between them.
To project either into the image, use `T_cam_lidar = inv(extrinsics) @ lidar_extrinsics`.
See `src/utils/projection.py`.

## pedestrian_sequences.json
```json
[
  {"seq_id": "000007", "num_pedestrians": 10, "lidar_batch": "lidar_velodyne_000000_000039.tar.gz"},
  ...
]
```
358 total entries. LiDAR available for seq 000000–000479 (130 sequences with pedestrians).
Working set = entries where `lidar_batch` is one of the 12 downloaded batches.
