"""Face tracking with MediaPipe's BlazeFace, stretched to reach the back of the room.

BlazeFace short-range squeezes its input to 128x128 and needs a face about
16-20 px tall there - on a whole 1280x720 frame that is a face ~180 px tall, a
player under a metre from a typical webcam. The full-range model that would fix
this only shipped with the legacy `mp.solutions` API, which mediapipe 0.10.35 no
longer has. So instead of a bigger model we feed the small one a smaller picture:

* **Tracking.** Once a player is locked, detect on a square crop about four times
  their face box. The face then fills a quarter of the 128 px input wherever
  they stand, so a face 25-30 px tall in the frame (~5 m) is still found. A
  full-frame pass runs alongside it, so someone stepping right up to the camera
  can still take over (see `geometry.choose_target`).
* **Searching.** With nobody locked, run the full frame (close players) plus
  `TILES_PER_FRAME` zoomed tiles from `geometry.search_tiles`, taken round-robin:
  three coarse tiles (~2.7x) for mid-range and a finer set (~5x) for the back
  of the room. A distant player is typically picked up within about half a
  second, after which tracking takes over.
* **Confirming.** A face found at low zoom (full frame or tile) must be found
  again on a crop around itself before it can take the lock - see `_choose`.
  That costs a call or two, but only on the frames where a lock is won.

With `long_range` off it is the plain whole-frame detector, one call per frame.

**Running mode.** The detector runs in IMAGE mode. BlazeFace carries nothing
from frame to frame, so VIDEO mode would buy no tracking - only the rule that
timestamps strictly increase per detector instance, which several crops of the
same frame would break (and a crop that changes size every frame is not a video
anyway). IMAGE mode lets one detector take any number of crops of any size.

**Confidence.** The detector is built with a low floor and `min_confidence` is
applied here, so the menu's slider changes it live without rebuilding anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .base import FACE_MODEL, MODE_FACE, Box, Target
from .geometry import (PixelRect, box_centre, box_iou, boxes_agree, challengers,
                       choose_target, crop_to_frame_box, dedupe, rect_covers, roi_rect, search_tiles)

DETECTOR_FLOOR = 0.2            # the lowest `min_confidence` the settings allow
ROI_SCALE = 4.0                 # tracking crop side, in face-box sizes
ROI_MIN_SIDE = 128              # px; tighter than the model input only adds blur
ROI_GROWTH = 1.3                # the crop widens by this per missed frame...
MAX_MISSES = 8                  # ...and after this many we go back to searching
TILES_PER_FRAME = 2             # search tiles per frame, on top of the full frame: a
                                # full sweep of distant faces in ~0.3s at 30 fps
SWITCH_RATIO = 1.8              # a face this many times bigger steals the lock
MAX_CONFIRMS = 2                # low-zoom finds re-checked per frame (1 call each)
CONFIRM_IOU = 0.3               # how well the closer look must agree
CONFIRM_SCALE = 2.0             # the closer look's crop, in box sizes: tight, so a
                                # blob resolves into the (smaller) face inside it

# Track state names, for diagnostics (`scripts/check_camera.py`).
SEARCH, TRACK, FULL = "search", "track", "full"


def _area(target: Target) -> float:
    return target.box[2] * target.box[3]


def missing_model_message(path: Path) -> str:
    return f"Model missing at {path}. Run scripts/fetch_models.sh"


class FaceTracker:
    """A `Tracker` following one face, with optional long-range assist."""

    mode = MODE_FACE

    def __init__(self, min_confidence: float = 0.5, long_range: bool = True,
                 model_path: Path = FACE_MODEL) -> None:
        if not model_path.exists():
            raise FileNotFoundError(missing_model_message(model_path))
        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min(DETECTOR_FLOOR, min_confidence),
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self.min_confidence = min_confidence
        self.long_range = long_range
        self._tiles: List[PixelRect] = []
        self._tiles_for: Tuple[int, int] = (0, 0)
        self._tile_index = 0
        self.reset()

    # ------------------------------------------------------------ protocol

    def reset(self) -> None:
        self._locked: Optional[Target] = None
        self._misses = 0
        self._roi: Optional[PixelRect] = None
        self.state = SEARCH
        self.calls = 0                          # detector calls in the last frame

    def close(self) -> None:
        self._detector.close()

    @property
    def searching(self) -> bool:
        """No player locked: the long-range scan is running."""
        return self._locked is None

    @property
    def roi(self) -> Optional[PixelRect]:
        """The tracking crop used on the last frame, if any (pixels of that frame)."""
        return self._roi

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[Target]:
        height, width = rgb.shape[:2]
        self.calls = 0
        self._roi = None
        full = (0, 0, width, height)
        locked = self._locked
        trusted: List[Target] = []              # seen through the tracking crop
        unconfirmed: List[Target] = []          # seen at low zoom: may be a blob

        if not self.long_range:
            trusted = self._detect(rgb, full)
            self.state = FULL
        elif locked is not None:
            grow = ROI_GROWTH ** self._misses
            roi = roi_rect(locked.box, width, height, ROI_SCALE * grow, ROI_MIN_SIDE)
            if rect_covers(roi, width, height):
                # The face is so big the crop would be the frame: just use it.
                trusted = self._detect(rgb, full)
            else:
                self._roi = roi
                trusted = self._detect(rgb, roi)
                # The full frame too, but only for a much bigger newcomer - never
                # to stand in for the player when the crop misses a frame.
                unconfirmed = challengers(self._detect(rgb, full), _area(locked), _area,
                                          SWITCH_RATIO)
            self.state = TRACK
        else:
            unconfirmed = self._detect(rgb, full)
            for tile in self._next_tiles(width, height):
                unconfirmed += self._detect(rgb, tile)
            self.state = SEARCH

        candidates = dedupe(trusted + sorted(unconfirmed, key=lambda t: -t.confidence),
                            lambda t: t.box)
        chosen = self._choose(rgb, candidates, trusted, locked)
        if chosen is not None:
            self._locked, self._misses = chosen, 0
        elif locked is not None:
            self._misses += 1
            if self._misses > MAX_MISSES or not self.long_range:
                self._locked, self._misses = None, 0
        return chosen

    def _choose(self, rgb: np.ndarray, candidates: List[Target], trusted: List[Target],
                locked: Optional[Target]) -> Optional[Target]:
        """The stickiness rule, with low-zoom finds confirmed before they win.

        On a whole frame BlazeFace sometimes calls a small distant person's
        entire head-and-shoulders one big "face". Such a box would win on size,
        so anything not already seen through the tracking crop is re-detected on
        a crop around itself first, and dropped if the closer look disagrees.
        """
        previous = locked.head if locked else None
        for _ in range(MAX_CONFIRMS + 1):
            chosen = choose_target(candidates, previous, size=_area,
                                   head=lambda t: t.head, switch_ratio=SWITCH_RATIO)
            if chosen is None or any(chosen is t for t in trusted):
                return chosen
            confirmed = self._confirm(rgb, chosen)
            if confirmed is not None:
                return confirmed
            candidates = [t for t in candidates if t is not chosen]
        return choose_target(trusted, previous, size=_area, head=lambda t: t.head,
                             switch_ratio=SWITCH_RATIO)

    def _confirm(self, rgb: np.ndarray, candidate: Target) -> Optional[Target]:
        """The crop's view of `candidate`, if a closer look agrees it is a face."""
        height, width = rgb.shape[:2]
        roi = roi_rect(candidate.box, width, height, CONFIRM_SCALE, ROI_MIN_SIDE)
        agreeing = [t for t in self._detect(rgb, roi)
                    if boxes_agree(t.box, candidate.box, CONFIRM_IOU)]
        return max(agreeing, key=lambda t: box_iou(t.box, candidate.box), default=None)

    # ------------------------------------------------------------- helpers

    def _next_tiles(self, width: int, height: int) -> List[PixelRect]:
        if self._tiles_for != (width, height):
            self._tiles = search_tiles(width, height)
            self._tiles_for = (width, height)
            self._tile_index = 0
        chosen = []
        for _ in range(min(TILES_PER_FRAME, len(self._tiles))):
            chosen.append(self._tiles[self._tile_index % len(self._tiles)])
            self._tile_index += 1
        return chosen

    def _detect(self, rgb: np.ndarray, rect: PixelRect) -> List[Target]:
        """Faces in `rect` of `rgb`, as Targets normalised to the whole frame."""
        x, y, w, h = rect
        height, width = rgb.shape[:2]
        crop = rgb if (w, h) == (width, height) else rgb[y:y + h, x:x + w]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        self.calls += 1
        result = self._detector.detect(image)
        found = []
        for detection in result.detections:
            score = detection.categories[0].score if detection.categories else 0.0
            if score < self.min_confidence:
                continue
            b = detection.bounding_box            # pixels of the crop
            local: Box = (b.origin_x / w, b.origin_y / h, b.width / w, b.height / h)
            box = crop_to_frame_box(local, rect, width, height)
            found.append(Target(head=box_centre(box), box=box, confidence=score))
        return found
