"""Read the shipped dataset — the consumer-facing API.

Everything in this module works identically against the live pipeline output
(`data/annotations/`) and against an unpacked release bundle, because the packager keeps the
same layout. A consumer never needs to know which one they have.

The split of responsibility:
  * `load_index` / `iter_samples` — choose a population (split, tier) from the Parquet index.
  * `load_sample` — the full per-sample JSON, including the trajectory and the sensor pointers.
  * `boxes_array` / `positions_array` — the two arrays a model actually trains on.
  * `media_paths` / `load_radar_window` — resolve sensor pointers against the raw ZOD tree.

**The one trap worth knowing.** A sample's `trajectory.frames` runs past the observation window
into the prediction horizon, so reading it whole leaks the label. `positions_array` defends
against this by default — see `_part_indices` for the rule.

Sensor pointers are RELATIVE to the sequence directory (`camera_front_blur/....jpg`), so
resolving them needs the ZOD sequences root.

Quickstart:

    from zodped.dataset.loader import load_index, load_sample, boxes_array

    index = load_index("data/annotations")
    gold_train = index[index.is_in_gold_standard & (index.split == "train")]

    doc = load_sample(gold_train.sample_id.iloc[0], "data/annotations")
    boxes = boxes_array(doc)                 # (T, 4) pixel xyxy, NaN where unobserved
    label = doc["intent"]["labels_by_horizon"]["2.0"]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from zodped.dataset.keyframe import parse_zod_ts
from zodped.dataset.stats import DEFAULT_HORIZON, horizon_column

INDEX_NAME = "dataset_index.parquet"


def load_index(root: Path | str) -> pd.DataFrame:
    """Load the Parquet index: one row per sample, all scalar fields, for fast filtering."""
    path = Path(root) / INDEX_NAME
    if not path.exists():
        raise FileNotFoundError(f"no dataset index at {path} — is {root} a dataset directory?")
    return pd.read_parquet(path)


def select(
    index: pd.DataFrame,
    split: Optional[str] = None,
    tier: Optional[str] = None,
    horizon: str = DEFAULT_HORIZON,
    crossing_only: bool = False,
) -> pd.DataFrame:
    """Filter the index by split and tier.

    `tier` is ``"gold"`` or ``"silver"``. Evaluate on GOLD only: SILVER carries geometry labels
    with ~28% measured false-positive rate on declared crossings, so a score against SILVER
    measures agreement with the geometry rule rather than crossing behaviour.
    """
    out = index
    if split is not None:
        out = out[out.split == split]
    if tier is not None:
        if tier not in ("gold", "silver"):
            raise ValueError(f"tier must be 'gold' or 'silver', got {tier!r}")
        out = out[out.is_in_gold_standard == (tier == "gold")]
    if crossing_only:
        out = out[out[horizon_column(horizon)] == "crossing"]
    return out


def load_sample(sample_id: str, root: Path | str) -> dict:
    """Load one sample's full JSON document by id."""
    path = Path(root) / f"{sample_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no sample {sample_id!r} at {path}")
    return json.loads(path.read_text())


def iter_samples(
    root: Path | str,
    split: Optional[str] = None,
    tier: Optional[str] = None,
    index: Optional[pd.DataFrame] = None,
) -> Iterator[dict]:
    """Yield full sample documents for a chosen population, in index order.

    Pass a pre-filtered `index` to iterate any selection `select` cannot express.
    """
    root = Path(root)
    frame = load_index(root) if index is None else index
    for sample_id in select(frame, split=split, tier=tier).sample_id:
        yield load_sample(sample_id, root)


# ---------------------------------------------------------------------------
# Per-sample arrays
# ---------------------------------------------------------------------------

def intent_label(doc: dict, horizon: str = DEFAULT_HORIZON) -> str:
    """The forward-looking intent label at one horizon: ``crossing`` or ``not_crossing``.

    Intent is per WINDOW and looks forward — will crossing start within `horizon` seconds AFTER
    the window ends. It is not `doc["action"]["crosses_ego_road"]`, which is a verdict about the
    whole track. Conflating the two turns intent prediction into action detection.
    """
    return doc["intent"]["labels_by_horizon"][horizon]


def boxes_array(doc: dict) -> np.ndarray:
    """Camera boxes over the window as ``(T, 4)`` pixel xyxy, NaN-filled where unobserved.

    NaN rather than zeros: a zero box is a confident claim about a pedestrian who was outside the
    frame. Callers must decide explicitly whether to drop or impute such a window.
    """
    frames = doc["multimodal"]["camera_frames"]
    out = np.full((len(frames), 4), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame.get("bbox_xyxy") is not None:
            out[i] = frame["bbox_xyxy"]
    return out


def window_bounds(doc: dict) -> tuple[float, float]:
    """``(start, end)`` of the OBSERVATION window in Unix seconds."""
    return parse_zod_ts(doc["window_start_timestamp"]), parse_zod_ts(doc["window_end_timestamp"])


PARTS = ("observed", "future", "all")


def _part_indices(doc: dict, part: str) -> np.ndarray:
    """Indices of the trajectory frames belonging to `part`.

    **`trajectory.frames` is not the observation window.** It spans
    ``[window_start, window_end + max_horizon]`` (`zodped.labeling.samples._trajectory_block`), so
    typically most of its rows lie AFTER the window — they are the trajectory-prediction target,
    and feeding them to a model as input leaks the answer. Hence `part`:

      * ``observed`` — frames inside the observation window. **Model input.** The default.
      * ``future``   — frames after it. The prediction TARGET; never an input feature.
      * ``all``      — everything, for plotting a whole trajectory.

    Do not use the per-frame `in_observation` flag for this: it means the tracker had a real
    detection that frame rather than coasting, which is a different question entirely.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, got {part!r}")
    if part == "all":
        return np.arange(len(doc["trajectory"]["frames"]))
    _, end = window_bounds(doc)
    ts = np.asarray([parse_zod_ts(f["timestamp"]) for f in doc["trajectory"]["frames"]])
    return np.where(ts <= end if part == "observed" else ts > end)[0]


def positions_array(doc: dict, frame: str = "world", part: str = "observed") -> np.ndarray:
    """Tracked 3D positions as ``(T, 3)``.

    `frame` is ``"world"`` (motion-compensated, ego motion removed — use this for trajectory
    geometry) or ``"ego_rel"`` (relative to the ego vehicle at window start — use this when the
    ego's own motion should be visible).

    `part` selects observation window / future / both, and defaults to ``observed``; see
    `_part_indices`.
    """
    key = {"world": "position_world", "ego_rel": "position_ego_rel"}.get(frame)
    if key is None:
        raise ValueError(f"frame must be 'world' or 'ego_rel', got {frame!r}")
    frames = doc["trajectory"]["frames"]
    idx = _part_indices(doc, part)
    return np.asarray([frames[i][key] for i in idx], dtype=np.float32).reshape(len(idx), 3)


def frame_timestamps(doc: dict, part: str = "observed") -> np.ndarray:
    """Trajectory frame timestamps as Unix seconds, ``(T,)``, aligned with `positions_array`."""
    frames = doc["trajectory"]["frames"]
    return np.asarray([parse_zod_ts(frames[i]["timestamp"]) for i in _part_indices(doc, part)])


# ---------------------------------------------------------------------------
# Raw sensor resolution
# ---------------------------------------------------------------------------

def media_paths(doc: dict, sequences_root: Path | str) -> dict:
    """Resolve a sample's sensor pointers to absolute paths in the raw ZOD tree.

    Returns ``{"camera": [Path, ...], "lidar": [Path, ...], "radar": Path | None}``. Paths are
    returned whether or not the file is on disk — a partial ZOD download is normal, and the
    caller decides what a missing modality means for them.
    """
    seq_dir = Path(sequences_root) / doc["sequence_id"]
    radar = doc["multimodal"].get("radar_path")
    return {
        "camera": [seq_dir / f["path"] for f in doc["multimodal"]["camera_frames"]],
        "lidar": [seq_dir / s["path"] for s in doc["multimodal"]["lidar_scans"]],
        "radar": (seq_dir / radar) if radar else None,
    }


def load_radar_window(doc: dict, sequences_root: Path | str, pad_s: float = 0.0) -> np.ndarray:
    """Radar returns falling inside the sample's observation window.

    ZOD ships radar as ONE structured `.npy` per sequence holding every sweep of the ~20s clip
    (~45k returns), so `radar_path` alone is a whole-sequence blob rather than a window. This
    slices it to the window on the per-return `timestamp` field (nanoseconds), which is the step
    a consumer would otherwise have to rediscover.

    `pad_s` widens the window symmetrically — radar runs at ~17 Hz, so a 0.5s window holds only
    ~8 sweeps and a little padding is often wanted for context.
    """
    path = media_paths(doc, sequences_root)["radar"]
    if path is None or not path.exists():
        return np.empty(0)
    returns = np.load(path)
    start, end = window_bounds(doc)
    ts_s = returns["timestamp"] / 1e9
    return returns[(ts_s >= start - pad_s) & (ts_s <= end + pad_s)]
