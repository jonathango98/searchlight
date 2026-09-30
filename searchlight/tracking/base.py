"""The tracking contract: what a tracker returns and what the camera publishes.

Pure data - no OpenCV or MediaPipe - so the game, the settings and the tests can
import it without dragging in native libraries.

Coordinates are normalised image coordinates of the frame the camera publishes
(after mirroring and digital zoom): x and y in [0, 1], origin top-left. That is
the same space the square lives in, so the game never converts between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]     # left, top, width, height; normalised

MODE_FACE = "face"
MODE_UPPER_BODY = "upper_body"

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FACE_MODEL = MODELS_DIR / "blaze_face_short_range.tflite"
POSE_MODEL = MODELS_DIR / "pose_landmarker_full.task"


@dataclass(frozen=True)
class CameraOptions:
    """Everything the camera thread needs. Built by `Settings.camera_options()`."""

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mode: str = MODE_FACE
    long_range: bool = True
    zoom: float = 1.0               # 1.0 = whole frame; 2.0 = centre half
    min_confidence: float = 0.5
    smoothing: float = 0.4          # 0 = raw; higher = steadier but laggier
    mirror: bool = True


@dataclass(frozen=True)
class Target:
    """One tracked head in one frame."""

    head: Point                     # the point that must stay inside the square
    box: Box                        # outline to draw (face box, or head box from pose)
    confidence: float = 1.0


class Tracker(Protocol):
    """A detector wrapped to report at most one player per frame."""

    mode: str
    searching: bool                 # True while no player is locked

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[Target]:
        """Find the player in an RGB frame. `timestamp_ms` strictly increases."""
        ...

    def reset(self) -> None:
        """Forget the current player (e.g. between runs)."""
        ...

    def close(self) -> None:
        ...


@dataclass
class HeadDetection:
    """What the camera thread publishes: the newest frame and who is in it."""

    frame: Optional[np.ndarray] = None      # RGB, mirrored and zoomed as displayed
    head: Optional[Point] = None
    box: Optional[Box] = None
    timestamp: float = 0.0                  # time.monotonic() of capture
    ok: bool = False                        # a frame has been captured
    confidence: float = 0.0
    mode: str = MODE_FACE
    fps: float = 0.0                        # tracking frames per second
    searching: bool = True                  # no player locked (long-range scan active)
