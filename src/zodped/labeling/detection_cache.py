"""Persisted 2D-detection cache — run YOLO once, reuse forever (Step 0).

The detector is the only expensive part of Step 1 (~170 ms/image at imgsz=2560 on the 4080);
everything downstream — frustum lift, the KF/RTS linker, box assembly — is cheap LiDAR geometry we
re-run repeatedly while tuning. This module persists the per-image 2D person boxes so that geometry
can iterate in minutes instead of paying for YOLO each time.

The cache is keyed by image FILENAME (stable across runs). Invalidate it — re-run
`scripts/00_detect.py` — only when the detector GEOMETRY changes (model / imgsz), since those change
the boxes themselves. `conf` is NOT an invalidator: cache once at a low floor and a run raises the
threshold for free via `decode_detections(min_conf=...)`; geometry/tracking parameters do not change
the boxes at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from zodped.labeling.detector import Detection2D

SCHEMA = "detections/v1"


def cache_path(detections_dir: Path, seq_id: str) -> Path:
    """Per-sequence cache file path."""
    return detections_dir / f"{seq_id}.json"


def write_detections(path: Path, meta: dict, per_image: Dict[str, List[Detection2D]]) -> None:
    """Persist one sequence's detections.

    `per_image` maps image filename -> its `Detection2D` list. Boxes are stored as flat
    [x1, y1, x2, y2, score] rows to keep the file compact.
    """
    images = {
        fn: [[float(d.xyxy[0]), float(d.xyxy[1]), float(d.xyxy[2]), float(d.xyxy[3]), float(d.score)]
             for d in dets]
        for fn, dets in per_image.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SCHEMA, "meta": meta, "images": images}))


def read_doc(path: Path) -> dict:
    """Load one sequence's raw cache document (schema / meta / images) without decoding boxes."""
    return json.loads(Path(path).read_text())


def decode_detections(doc: dict, min_conf: float = 0.0) -> Dict[str, List[Detection2D]]:
    """Decode a cache document to {filename: [Detection2D, ...]}, dropping boxes below `min_conf`.

    `min_conf` is the run's confidence threshold applied at LOAD time: the cache is built once at a
    low floor (``meta["conf"]``) so a run can raise the threshold for free. Requesting a threshold
    below the cached floor cannot recover boxes that were never stored.
    """
    return {
        fn: [Detection2D(xyxy=np.asarray(r[:4], dtype=np.float64), score=float(r[4]))
             for r in rows if r[4] >= min_conf]
        for fn, rows in doc["images"].items()
    }


def load_detections(path: Path, min_conf: float = 0.0) -> Dict[str, List[Detection2D]]:
    """Read + decode a per-sequence cache in one call (read_doc + decode_detections)."""
    return decode_detections(read_doc(path), min_conf)


def cached_detector(per_image: Dict[str, List[Detection2D]]) -> Callable[[object], List[Detection2D]]:
    """Wrap a loaded cache as a `(image_path) -> List[Detection2D]` closure, keyed by filename.

    Drop-in for `make_detector`'s closure, so `build_candidate_pool` is detector-source agnostic.
    """
    def detector(image_path) -> List[Detection2D]:
        return per_image.get(Path(image_path).name, [])

    return detector
