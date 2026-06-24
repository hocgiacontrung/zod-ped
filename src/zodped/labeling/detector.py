"""COCO 2D pedestrian-detector factory — the Step 1 measurement source.

Builds a YOLO or RT-DETR detector once and returns a per-image closure yielding `Detection2D`
boxes. The trajectory pipeline (frustum lift consumes these boxes), the demo video, and the
bring-up gates all import from here, so the detector lives in exactly one place and every caller
is detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

COCO_PERSON_CLASS = 0


@dataclass
class Detection2D:
    """A predicted 2D box in image pixels (x1, y1, x2, y2)."""

    xyxy: np.ndarray
    score: float


def make_detector(model_name: str = "yolo11x.pt", conf: float = 0.25, imgsz: int = 1280,
                  person_class: int = COCO_PERSON_CLASS):
    """Build a COCO detector once and return a per-image `(path) -> List[Detection2D]` closure.

    Dispatches on `model_name`: a `rtdetr*` name loads ultralytics' RT-DETR, anything else loads
    YOLO. Both share the same `.predict(...)` / `.boxes` interface, so callers stay detector-agnostic
    and the bring-up gate is an apples-to-apples comparison.

    `imgsz=1280`+ matters: ZOD images are 3848x2168, so distant pedestrians are tiny and get lost at
    low input resolution.
    """
    is_rtdetr = Path(model_name).name.lower().startswith("rtdetr")
    try:
        from ultralytics import RTDETR, YOLO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not import ultralytics. `pip install ultralytics`. {exc!r}") from exc

    model = (RTDETR if is_rtdetr else YOLO)(model_name)

    def detector(image_path) -> List[Detection2D]:
        res = model.predict(source=str(image_path), imgsz=imgsz, conf=conf,
                            classes=[person_class], verbose=False, device=0)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        return [Detection2D(xyxy=b, score=float(s)) for b, s in zip(xyxy, scores)]

    return detector
