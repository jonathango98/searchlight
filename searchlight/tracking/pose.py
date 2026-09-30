"""Upper-body tracking with MediaPipe's PoseLandmarker.

Why offer it next to faces: the pose detector looks at the head *and* shoulders,
at a 224 px input rather than BlazeFace's 128, so on the whole frame it keeps
finding a player further back than the face detector does, and it copes with a
turned head. The head comes from the pose's face landmarks (0-10).

It is costly - one call is ~30 ms on Apple silicon whether given a frame or a
crop - so, unlike the face tracker, this one makes exactly one call per frame:

* **Searching:** alternate the full frame with one of three coarse zoomed tiles
  (`geometry.search_tiles`), which reaches players whose upper body is too small
  for the whole-frame pass.
* **Tracking:** run on a square crop around the locked player's head and
  shoulders, so a distant player is seen at close range. Every
  `CHECK_EVERY`-th frame runs the full frame instead, to let a much bigger
  newcomer take over; on that frame a player it cannot see holds its last
  position rather than dropping out.

**Running mode.** IMAGE. VIDEO mode only skips the person detector (its real
speed-up) when it is already following `num_poses` people; with `num_poses=2`
and one player it re-detects every frame anyway, and its internal tracking
assumes a fixed frame, which a moving crop is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .base import MODE_UPPER_BODY, POSE_MODEL, Target
from .face import FULL, SEARCH, TRACK, missing_model_message
from .geometry import (PixelRect, challengers, choose_target, clamp_rect,
                       crop_to_frame_point, rect_covers, search_tiles)

NOSE = 0
EYES = (2, 5)
EARS = (7, 8)
FACE = range(0, 11)
SHOULDERS = (11, 12)

VISIBLE = 0.5                   # landmark visibility/presence to trust it
DETECTOR_FLOOR = 0.2
ROI_SCALE = 3.5                 # crop side, in shoulder widths
ROI_MIN_SIDE = 224              # px: the pose detector's input size
ROI_GROWTH = 1.3
MAX_MISSES = 6
CHECK_EVERY = 8                 # tracking: every Nth frame is a full-frame look
SWITCH_RATIO = 1.8              # compared on shoulder width squared, i.e. area


@dataclass(frozen=True)
class _Pose:
    target: Target
    shoulder_px: float          # shoulder width in frame pixels
    centre: tuple               # frame pixels, between head and shoulders


def _size(pose: _Pose) -> float:
    return pose.shoulder_px ** 2


class PoseTracker:
    """A `Tracker` following one upper body."""

    mode = MODE_UPPER_BODY

    def __init__(self, min_confidence: float = 0.5, model_path: Path = POSE_MODEL) -> None:
        if not model_path.exists():
            raise FileNotFoundError(missing_model_message(model_path))
        floor = min(DETECTOR_FLOOR, min_confidence)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=floor,
            min_pose_presence_confidence=floor,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.min_confidence = min_confidence
        self.long_range = True                  # read by CameraThread; crops on/off
        self._tiles: List[PixelRect] = []
        self._tiles_for = (0, 0)
        self.reset()

    def reset(self) -> None:
        self._locked: Optional[_Pose] = None
        self._misses = 0
        self._frame = 0
        self._roi: Optional[PixelRect] = None
        self.state = SEARCH
        self.calls = 0

    def close(self) -> None:
        self._landmarker.close()

    @property
    def searching(self) -> bool:
        return self._locked is None

    @property
    def roi(self) -> Optional[PixelRect]:
        return self._roi

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[Target]:
        height, width = rgb.shape[:2]
        full = (0, 0, width, height)
        self._frame += 1
        self._roi = None
        self.calls = 1
        locked = self._locked
        holding = False

        if not self.long_range:
            rect, self.state = full, FULL
        elif locked is not None:
            side = max(ROI_SCALE * locked.shoulder_px * ROI_GROWTH ** self._misses,
                       ROI_MIN_SIDE)
            roi = clamp_rect(locked.centre[0], locked.centre[1], side, side, width, height)
            if rect_covers(roi, width, height):
                rect = full
            elif self._frame % CHECK_EVERY == 0:
                rect, holding = full, True
            else:
                rect = self._roi = roi
            self.state = TRACK
        else:
            rect = full if self._frame % 2 else self._next_tile(width, height)
            self.state = SEARCH

        poses = self._detect(rgb, rect)
        if holding and locked is not None:
            # Accept the player (near where they were) or a much bigger newcomer,
            # never a bystander standing in for a player too small to see here.
            (lx, ly) = locked.target.head
            near = [p for p in poses if math.hypot((p.target.head[0] - lx) * width,
                                                   (p.target.head[1] - ly) * height)
                    < locked.shoulder_px]
            poses = near + challengers(poses, locked.shoulder_px ** 2, _size, SWITCH_RATIO)
        chosen = choose_target(poses, locked.target.head if locked else None,
                               size=_size,
                               head=lambda p: p.target.head, switch_ratio=SWITCH_RATIO)
        if chosen is not None:
            self._locked, self._misses = chosen, 0
            return chosen.target
        if holding and locked is not None:
            # A full-frame check that could not see a distant player is not a miss.
            return locked.target
        if locked is not None:
            self._misses += 1
            if self._misses > MAX_MISSES or not self.long_range:
                self._locked, self._misses = None, 0
        return None

    # ------------------------------------------------------------- helpers

    def _next_tile(self, width: int, height: int) -> PixelRect:
        if self._tiles_for != (width, height):
            # Coarse tiles only (~2.5x): the pose detector sees whole shoulders,
            # so it needs far less zoom than the face detector.
            self._tiles = search_tiles(width, height, fine_rows=0)
            self._tiles_for = (width, height)
        return self._tiles[(self._frame // 2) % len(self._tiles)]

    def _detect(self, rgb: np.ndarray, rect: PixelRect) -> List[_Pose]:
        x, y, w, h = rect
        height, width = rgb.shape[:2]
        crop = rgb if (w, h) == (width, height) else rgb[y:y + h, x:x + w]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        result = self._landmarker.detect(image)
        poses = []
        for landmarks in result.pose_landmarks:
            pose = self._to_pose(landmarks, rect, width, height)
            if pose is not None and pose.target.confidence >= self.min_confidence:
                poses.append(pose)
        return poses

    @staticmethod
    def _to_pose(landmarks: Sequence, rect: PixelRect, width: int,
                 height: int) -> Optional[_Pose]:
        def px(i: int):
            p = crop_to_frame_point((landmarks[i].x, landmarks[i].y), rect, width, height)
            return p[0] * width, p[1] * height

        def seen(i: int) -> float:
            lm = landmarks[i]
            return min(lm.visibility or 0.0, lm.presence or 0.0)

        face_seen = [i for i in FACE if seen(i) >= VISIBLE]
        if not face_seen:
            return None
        if seen(NOSE) >= VISIBLE:
            head = px(NOSE)
        else:
            around = [i for i in EYES + EARS if seen(i) >= VISIBLE] or face_seen
            head = tuple(np.mean([px(i) for i in around], axis=0))

        points = np.array([px(i) for i in face_seen])
        spread = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
        (lx, ly), (rx, ry) = px(EARS[0]), px(EARS[1])
        ears = math.hypot(lx - rx, ly - ry)
        (sx1, sy1), (sx2, sy2) = px(SHOULDERS[0]), px(SHOULDERS[1])
        shoulders = math.hypot(sx1 - sx2, sy1 - sy2)
        # Ears span the head face-on; half the shoulders stand in when the head
        # turns side-on and the ears overlap.
        side = max(1.2 * ears, 0.5 * shoulders, 1.3 * spread, 4.0)
        box = ((head[0] - side / 2) / width, (head[1] - 0.6 * side) / height,
               side / width, 1.2 * side / height)
        confidence = float(np.mean([seen(i) for i in face_seen]))
        target = Target(head=(head[0] / width, head[1] / height), box=box,
                        confidence=confidence)
        # Crop centre: between the head and the shoulder line.
        centre = (head[0], (head[1] + (sy1 + sy2) / 2) / 2)
        return _Pose(target=target, shoulder_px=max(shoulders, side), centre=centre)
