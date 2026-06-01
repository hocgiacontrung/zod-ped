import json
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/home/user20/projects/zod-ped/data/raw/sequences")

results = []

seq_dirs = sorted(DATA_ROOT.iterdir())
print(f"Scanning {len(seq_dirs)} sequences...")

for seq_dir in seq_dirs:
    if not seq_dir.is_dir():
        continue

    obj_file = seq_dir / "annotations" / "object_detection.json"
    if not obj_file.exists():
        continue

    with open(obj_file) as f:
        annotations = json.load(f)

    # Count pedestrians - field is properties.class
    peds = [
        ann for ann in annotations
        if ann.get("properties", {}).get("class") == "Pedestrian"
    ]

    if peds:
        seq_id = int(seq_dir.name)
        lidar_batch_start = (seq_id // 40) * 40
        lidar_file = f"lidar_velodyne_{lidar_batch_start:06d}_{lidar_batch_start+39:06d}.tar.gz"
        results.append({
            "seq_id": seq_dir.name,
            "num_pedestrians": len(peds),
            "lidar_batch": lidar_file,
        })

# Summary
print(f"\nSequences with pedestrians: {len(results)} / {len(seq_dirs)}")

# Group by LiDAR batch
batches = defaultdict(list)
for r in results:
    batches[r["lidar_batch"]].append(r["seq_id"])

print(f"\nLiDAR batches needed: {len(batches)}")
for batch, seqs in sorted(batches.items()):
    print(f"  {batch}  →  {len(seqs)} sequences: {seqs[:5]}{'...' if len(seqs)>5 else ''}")

# Save result
out_file = Path("/home/user20/projects/zod-ped/data") / "pedestrian_sequences.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_file}")