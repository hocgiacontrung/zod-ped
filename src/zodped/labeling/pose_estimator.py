"""COCO 2D-pose estimator factory — the PedGraph+ measurement source (Step 0b).

Builds a YOLO-pose model once and returns a per-image closure yielding `PersonPose` skeletons.
Sibling of `detector.py`: same one-place-only, caller-agnostic pattern, but the payload is 17 COCO
keypoints per person instead of a single box. The pose cache (`pose_cache.py`) and the PedGraph
scorer both import from here, so the pose model lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

COCO_PERSON_CLASS = 0
NUM_COCO_KEYPOINTS = 17


@dataclass
class PersonPose:
    """One predicted person: 2D box (x1, y1, x2, y2) plus 17 COCO keypoints as (x, y, conf)."""

    xyxy: np.ndarray        # (4,) person box in image pixels
    score: float            # person detection confidence
    keypoints: np.ndarray   # (17, 3) COCO keypoints: x, y, per-joint confidence


def make_pose_estimator(model_name: str = "yolo11x-pose.pt", conf: float = 0.1, imgsz: int = 2560,
                        person_class: int = COCO_PERSON_CLASS):
    """Build a YOLO-pose model once and return a per-image `(path) -> List[PersonPose]` closure.

    `imgsz=2560`+ matters for the same reason as the detector: ZOD images are 3848x2168, so distant
    pedestrians are tiny and their skeletons collapse at low input resolution. Cache at a low `conf`
    floor — a run raises the person threshold for free at load time (see `pose_cache.decode_poses`).
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not import ultralytics. `pip install ultralytics`. {exc!r}") from exc

    model = YOLO(model_name)

    def estimator(image_path) -> List[PersonPose]:
        res = model.predict(source=str(image_path), imgsz=imgsz, conf=conf,
                            classes=[person_class], verbose=False, device=0)[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        return [PersonPose(xyxy=b, score=float(s), keypoints=k)
                for b, s, k in zip(xyxy, scores, kpts)]

    return estimator
