"""High-quality video writing via the system ffmpeg binary.

OpenCV's bundled ffmpeg in this environment only exposes the MPEG-4 Part 2 encoder
(`mp4v`); H.264/VP9 are unavailable, so `cv2.VideoWriter` produces visibly grainy,
low-bitrate output. The system `ffmpeg` binary, however, ships `libx264`, so we pipe
raw BGR frames to it and let it encode crisp H.264 — matching the look of the official
ZOD sample clips.

Usage mirrors ``cv2.VideoWriter``::

    writer = FFmpegWriter(out_path, fps=10, size=(w, h))
    writer.write(bgr_frame)   # HxWx3 uint8, BGR (OpenCV convention)
    writer.release()
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np


class FFmpegWriter:
    """Encode BGR frames to H.264 MP4 by streaming raw video into ffmpeg.

    libx264 requires even frame dimensions, so the target size is rounded down to even;
    frames are cropped to match before writing.
    """

    def __init__(self, path: Path, fps: int, size: Tuple[int, int], crf: int = 18) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg binary not found on PATH — required for video encoding")
        w, h = size
        self.w, self.h = w - (w % 2), h - (h % 2)        # libx264 needs even dims
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-pix_fmt", "yuv420p", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one BGR frame (cropped to the writer's even dimensions)."""
        self.proc.stdin.write(np.ascontiguousarray(frame[: self.h, : self.w]).tobytes())

    def release(self) -> None:
        """Flush and finalise the MP4."""
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
