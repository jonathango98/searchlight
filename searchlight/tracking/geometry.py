"""The arithmetic behind tracking: crops, tiles, coordinate maps, smoothing, choice.

Kept pure - no OpenCV, no MediaPipe, no camera - because this is where the bugs
hide (an off-by-one crop, a box mapped back into the wrong frame, a filter that
lags) and it is far easier to pin them down with unit tests than in front of a
webcam.

Two coordinate systems meet here:

* **pixel rects** `(x, y, w, h)` in integers - what we slice numpy frames with;
* **normalised** points and boxes (see `base.py`) - what trackers report. A
  detector run on a crop reports coordinates normalised to *that crop*; the
  `crop_to_frame_*` helpers carry them back to the frame.

Why crops at all: MediaPipe's face detector squeezes whatever it is given into a
128x128 input and needs a face about 16-20 px tall there. On a whole 1280x720
frame that is a face ~180 px tall - a player barely a metre away. Handing it a
crop around the player (tracking) or a zoomed tile (searching) makes a distant
face fill enough of those 128 pixels to be found.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import replace
from typing import Callable, Deque, List, Optional, Sequence, Tuple, TypeVar

from .base import Box, Point, Target

PixelRect = Tuple[int, int, int, int]       # x, y, w, h in pixels of some frame

T = TypeVar("T")


# ------------------------------------------------------------------- rects

def clamp_rect(cx: float, cy: float, w: float, h: float,
               frame_w: int, frame_h: int) -> PixelRect:
    """A `w` x `h` rect centred on (cx, cy), slid (not shrunk) to lie inside the frame.

    Sliding keeps the crop's size - and so the zoom it gives - constant as the
    player nears an edge. Only a rect larger than the frame is shrunk.
    """
    w = int(round(min(max(w, 1.0), frame_w)))
    h = int(round(min(max(h, 1.0), frame_h)))
    x = int(round(min(max(cx - w / 2, 0.0), frame_w - w)))
    y = int(round(min(max(cy - h / 2, 0.0), frame_h - h)))
    return x, y, w, h


def roi_rect(box: Box, frame_w: int, frame_h: int, scale: float = 4.0,
             min_side: int = 128) -> PixelRect:
    """A square crop around a normalised `box`, `scale` times its larger side.

    `min_side` stops the crop collapsing below the detector's own input size:
    cropping tighter than that only feeds it upscaled blur.
    """
    bw, bh = box[2] * frame_w, box[3] * frame_h
    side = max(scale * max(bw, bh), float(min_side))
    cx = (box[0] + box[2] / 2) * frame_w
    cy = (box[1] + box[3] / 2) * frame_h
    return clamp_rect(cx, cy, side, side, frame_w, frame_h)


def tile_row(frame_w: int, frame_h: int, side: int, count: int,
             centre_y: float) -> List[PixelRect]:
    """`count` square tiles of `side` px spread evenly across the frame's width."""
    side = min(side, frame_w, frame_h)
    y = int(round(min(max(centre_y - side / 2, 0.0), frame_h - side)))
    if count <= 1:
        return [(int(round((frame_w - side) / 2)), y, side, side)]
    step = (frame_w - side) / (count - 1)
    return [(int(round(i * step)), y, side, side) for i in range(count)]


def search_tiles(frame_w: int, frame_h: int, coarse: int = 3, overlap: float = 0.25,
                 band_centre: float = 0.4, fine_rows: int = 2) -> List[PixelRect]:
    """Overlapping square tiles over the band where heads appear, at two zoom levels.

    * **coarse** - `coarse` tiles across the width (~2/3 of the frame height
      each at 16:9): about 2.7x zoom, for players a couple of metres away.
    * **fine** - tiles half that size in `fine_rows` rows: about 5x zoom, for
      players at the back of the room.

    `overlap` is the share of a tile shared with its neighbour, so a face on a
    seam is whole in at least one tile. `band_centre` is where heads sit, as a
    fraction of the frame height: a camera at head height sees distant heads a
    little above the middle. The order interleaves the two scales so neither
    waits for the other's full cycle.
    """
    overlap = min(max(overlap, 0.0), 0.9)
    centre_y = band_centre * frame_h
    coarse_side = int(round(min(frame_h, frame_w / (coarse - (coarse - 1) * overlap))))
    coarse_tiles = tile_row(frame_w, frame_h, coarse_side, coarse, centre_y)

    fine_side = max(1, coarse_side // 2)
    per_row = max(1, math.ceil((frame_w - fine_side * overlap) / (fine_side * (1 - overlap))))
    row_step = fine_side * (1 - overlap)
    fine_tiles: List[PixelRect] = []
    for r in range(fine_rows):
        row_centre = centre_y + (r - (fine_rows - 1) / 2) * row_step
        fine_tiles.extend(tile_row(frame_w, frame_h, fine_side, per_row, row_centre))

    # Interleave: one coarse tile, then a few fine ones, so every scale comes
    # round often.
    ordered: List[PixelRect] = []
    chunk = max(1, math.ceil(len(fine_tiles) / max(1, len(coarse_tiles))))
    for i, tile in enumerate(coarse_tiles):
        ordered.append(tile)
        ordered.extend(fine_tiles[i * chunk:(i + 1) * chunk])
    ordered.extend(fine_tiles[len(coarse_tiles) * chunk:])
    return ordered


def zoom_rect(frame_w: int, frame_h: int, zoom: float) -> PixelRect:
    """The centre crop for a digital `zoom` (1.0 = whole frame), same aspect ratio."""
    zoom = max(1.0, float(zoom))
    return clamp_rect(frame_w / 2, frame_h / 2, frame_w / zoom, frame_h / zoom,
                      frame_w, frame_h)


def rect_covers(rect: PixelRect, frame_w: int, frame_h: int, share: float = 0.8) -> bool:
    """Whether `rect` spans most of the frame - then cropping to it gains nothing."""
    return rect[2] >= share * frame_w and rect[3] >= share * frame_h


# -------------------------------------------------------- coordinate maps

def crop_to_frame_point(point: Point, crop: PixelRect, frame_w: int, frame_h: int) -> Point:
    """A point normalised to `crop` -> normalised to the whole frame."""
    x, y, w, h = crop
    return ((x + point[0] * w) / frame_w, (y + point[1] * h) / frame_h)


def frame_to_crop_point(point: Point, crop: PixelRect, frame_w: int, frame_h: int) -> Point:
    x, y, w, h = crop
    return ((point[0] * frame_w - x) / w, (point[1] * frame_h - y) / h)


def crop_to_frame_box(box: Box, crop: PixelRect, frame_w: int, frame_h: int) -> Box:
    """A box normalised to `crop` -> normalised to the whole frame."""
    left, top = crop_to_frame_point((box[0], box[1]), crop, frame_w, frame_h)
    return (left, top, box[2] * crop[2] / frame_w, box[3] * crop[3] / frame_h)


def frame_to_crop_box(box: Box, crop: PixelRect, frame_w: int, frame_h: int) -> Box:
    left, top = frame_to_crop_point((box[0], box[1]), crop, frame_w, frame_h)
    return (left, top, box[2] * frame_w / crop[2], box[3] * frame_h / crop[3])


def crop_to_frame_target(target: Target, crop: PixelRect, frame_w: int,
                         frame_h: int) -> Target:
    return replace(target,
                   head=crop_to_frame_point(target.head, crop, frame_w, frame_h),
                   box=crop_to_frame_box(target.box, crop, frame_w, frame_h))


def box_centre(box: Box) -> Point:
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def boxes_agree(a: Box, b: Box, min_iou: float = 0.3, max_ratio: float = 2.0) -> bool:
    """Whether two detections are plausibly the same face.

    Either they overlap well, or - for tiny, imprecise boxes where IoU swings
    wildly - `a`'s centre lies in `b` and neither is more than `max_ratio` times
    the other's size.
    """
    if box_iou(a, b) >= min_iou:
        return True
    cx, cy = box_centre(a)
    inside = b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]
    size_a, size_b = math.sqrt(max(a[2] * a[3], 0.0)), math.sqrt(max(b[2] * b[3], 0.0))
    if not inside or size_a <= 0 or size_b <= 0:
        return False
    return 1 / max_ratio <= size_a / size_b <= max_ratio


def dedupe(items: Sequence[T], box: Callable[[T], Box], iou: float = 0.3) -> List[T]:
    """Drop items overlapping an earlier one - put the most trusted source first."""
    kept: List[T] = []
    for item in items:
        if all(box_iou(box(item), box(k)) < iou for k in kept):
            kept.append(item)
    return kept


# ---------------------------------------------------------- target choice

def choose_target(candidates: Sequence[T], previous: Optional[Point],
                  size: Callable[[T], float], head: Callable[[T], Point],
                  switch_ratio: float = 1.8) -> Optional[T]:
    """The player to follow among `candidates`.

    With nobody locked (`previous` is None) take the biggest - at a kiosk the
    player stands closest. Once locked, stay with whoever is nearest the last
    position, so a passer-by cannot steal the game; hand over only to someone
    at least `switch_ratio` times bigger (a new player stepping right up).
    """
    if not candidates:
        return None
    largest = max(candidates, key=size)
    if previous is None:
        return largest
    nearest = min(candidates, key=lambda c: _distance(head(c), previous))
    if size(largest) > switch_ratio * size(nearest):
        return largest
    return nearest


def challengers(candidates: Sequence[T], current_size: float,
                size: Callable[[T], float], switch_ratio: float = 1.8) -> List[T]:
    """Candidates big enough to take over from a player of `current_size`.

    For detections from outside the tracking crop: they may replace the player,
    but must never be mistaken for them just because the crop missed a frame.
    """
    return [c for c in candidates if size(c) > switch_ratio * current_size]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# -------------------------------------------------------------- smoothing

class OneEuroFilter:
    """Casiez et al.'s 1-euro filter: an adaptive low-pass for noisy positions.

    A plain moving average forces a choice between jitter at rest and lag in
    motion. The 1-euro filter lowers its cutoff when the signal is still (steady
    marker) and raises it with speed (the marker keeps up when the player moves).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = value, t
            return value
        dt = max(t - self._t, 1e-3)             # a repeated timestamp counts as 1 ms
        dx = (value - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (value - self._x)
        self._t = t
        return self._x


# `options.smoothing` 0..0.9 -> 1-euro min cutoff in Hz, geometric in between.
# 0.4 (the default) gives ~1.5 Hz: steady at rest, and the speed term (BETA, in
# frame-heights per second) opens it up whenever the head actually moves.
SMOOTHING_MAX = 0.9
CUTOFF_AT_MIN = 6.0
CUTOFF_AT_MAX = 0.25
BETA = 5.0


def smoothing_cutoff(strength: float) -> float:
    s = min(max(strength, 0.0), SMOOTHING_MAX) / SMOOTHING_MAX
    return CUTOFF_AT_MIN * (CUTOFF_AT_MAX / CUTOFF_AT_MIN) ** s


class TargetSmoother:
    """Smooths a Target's head and box together. `strength` 0 is a passthrough.

    A gap longer than `reset_after` seconds starts afresh, so a returning (or
    new) player's marker appears where they are instead of gliding in from where
    the last one left.
    """

    def __init__(self, strength: float = 0.4, reset_after: float = 0.5) -> None:
        self.reset_after = reset_after
        self._filters = [OneEuroFilter() for _ in range(6)]
        self._last_t: Optional[float] = None
        self.strength = strength

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._strength = min(max(float(value), 0.0), SMOOTHING_MAX)
        cutoff = smoothing_cutoff(self._strength)
        for f in self._filters:
            f.min_cutoff, f.beta = cutoff, BETA

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
        self._last_t = None

    def __call__(self, target: Optional[Target], t: float) -> Optional[Target]:
        if target is None:
            return None
        if self._last_t is not None and t - self._last_t > self.reset_after:
            self.reset()
        self._last_t = t
        if self._strength <= 0.0:
            return target
        # Smooth the box as centre + size, so it cannot turn inside out.
        cx, cy = box_centre(target.box)
        raw = (target.head[0], target.head[1], cx, cy, target.box[2], target.box[3])
        hx, hy, cx, cy, w, h = (f(v, t) for f, v in zip(self._filters, raw))
        return replace(target, head=(hx, hy), box=(cx - w / 2, cy - h / 2, w, h))


# ---------------------------------------------------------------- fps

class FpsMeter:
    """Frames per second over a sliding window of recent ticks."""

    def __init__(self, window: float = 1.0) -> None:
        self.window = window
        self._ticks: Deque[float] = deque()

    def tick(self, t: Optional[float] = None) -> float:
        t = time.monotonic() if t is None else t
        self._ticks.append(t)
        while len(self._ticks) > 2 and t - self._ticks[0] > self.window:
            self._ticks.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        self._ticks.clear()
