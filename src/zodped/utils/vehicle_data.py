"""Ego-vehicle signals from a sequence's ``vehicle_data.hdf5`` (Step-3 sample context).

Three signal groups are read (each with its own timestamp base, nanoseconds → Unix seconds):

  ego_vehicle_data/lon_vel_data      — longitudinal speed [m/s]        (~2 kHz)
  ego_vehicle_controls/turn_indicator_status — state: 0 off, 1 left, 2 right  (~4 kHz)
  satellite/heading                  — GNSS heading [deg]              (~0.5 kHz)

Speed is linearly interpolated; turn indicator and heading use the nearest sample (a discrete
state and a circular quantity respectively — interpolating either would fabricate values).
Missing/empty signal groups degrade to ``None`` rather than raising: a sample with unknown ego
speed is still a sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

# zod.data_classes.vehicle_data: 0 = off, 1 = left, 2 = right
_TURN_STATES = {0: "none", 1: "left", 2: "right"}


def _nearest(ts: np.ndarray, values: np.ndarray, t: float):
    return values[int(np.argmin(np.abs(ts - t)))]


class VehicleSignals:
    """Time-indexed access to the ego signals Step 3 attaches to each sample."""

    def __init__(
        self,
        speed_ts: np.ndarray,
        speed: np.ndarray,
        controls_ts: np.ndarray,
        turn_state: np.ndarray,
        sat_ts: np.ndarray,
        heading_deg: np.ndarray,
    ) -> None:
        self._speed_ts, self._speed = speed_ts, speed
        self._controls_ts, self._turn_state = controls_ts, turn_state
        self._sat_ts, self._heading = sat_ts, heading_deg

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VehicleSignals":
        def read(f: h5py.File, key: str) -> np.ndarray:
            return f[key][:] if key in f else np.empty(0)

        with h5py.File(path, "r") as f:
            return cls(
                speed_ts=read(f, "ego_vehicle_data/timestamp/nanoseconds/value") / 1e9,
                speed=read(f, "ego_vehicle_data/lon_vel_data/velocity/meters_per_second/value"),
                controls_ts=read(f, "ego_vehicle_controls/timestamp/nanoseconds/value") / 1e9,
                turn_state=read(f, "ego_vehicle_controls/turn_indicator_status/state"),
                sat_ts=read(f, "satellite/timestamp/nanoseconds/value") / 1e9,
                heading_deg=read(f, "satellite/heading/degrees/value"),
            )

    def speed_at(self, t: float) -> Optional[float]:
        """Longitudinal ego speed [m/s] at Unix time t (linear interpolation, clamped)."""
        if len(self._speed_ts) == 0:
            return None
        return float(np.interp(t, self._speed_ts, self._speed))

    def turn_indicator_at(self, t: float) -> Optional[str]:
        """Turn-indicator state ("none" | "left" | "right") at the nearest controls sample."""
        if len(self._controls_ts) == 0:
            return None
        return _TURN_STATES.get(int(_nearest(self._controls_ts, self._turn_state, t)), "none")

    def heading_at(self, t: float) -> Optional[float]:
        """GNSS heading [deg] at the nearest satellite sample (circular — never interpolated)."""
        if len(self._sat_ts) == 0:
            return None
        return float(_nearest(self._sat_ts, self._heading, t))
