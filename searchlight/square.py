"""The wandering square the player has to keep their head inside.

Pure: no pygame, OpenCV or MediaPipe, so its motion can be tested by stepping it
through simulated time.

Positions are normalised image coordinates - x and y in [0, 1], origin top-left,
the same space the head tracker reports in. Sizes and speeds, though, are measured
in *frame heights*, because a camera frame is wider than it is tall: a square whose
side is 0.3 in both normalised axes would be drawn as a wide rectangle. `aspect`
(width / height) converts between the two.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SquareConfig:
    """Every difficulty knob in one place."""

    aspect: float = 16 / 9

    # Side length in frame heights. It shrinks linearly with time survived until it
    # reaches the floor - small enough to be hard, big enough that tracking jitter
    # alone cannot push a still head out of it.
    start_side: float = 0.42
    min_side: float = 0.16
    shrink_per_second: float = 0.004

    # Speed in frame heights per second, ramping the same way.
    start_speed: float = 0.10
    max_speed: float = 0.45
    speed_ramp: float = 0.006

    # How hard the heading wanders, in radians per sqrt(second). The heading takes a
    # random walk rather than jumping, so the square curves instead of zig-zagging
    # and a player can always read where it is heading next.
    turn_rate: float = 1.6

    # Keep this far (frame heights) from every edge, so the square never asks a
    # player to put their head somewhere the camera crops off.
    margin: float = 0.04


class Square:
    def __init__(self, config: Optional[SquareConfig] = None,
                 seed: Optional[int] = None) -> None:
        self.config = config or SquareConfig()
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        """Park in the middle at full size, ready for a new run."""
        self.x = 0.5
        self.y = 0.5
        self.side = self.config.start_side
        self.speed = self.config.start_speed
        self.heading = self._rng.uniform(0, 2 * math.pi)

    # ------------------------------------------------------------- geometry

    @property
    def half_w(self) -> float:
        """Half the side, in normalised x units."""
        return self.side / 2 / self.config.aspect

    @property
    def half_h(self) -> float:
        """Half the side, in normalised y units."""
        return self.side / 2

    def bounds(self) -> tuple:
        """(left, top, right, bottom) in normalised image coordinates."""
        return (self.x - self.half_w, self.y - self.half_h,
                self.x + self.half_w, self.y + self.half_h)

    def contains(self, x: float, y: float) -> bool:
        return abs(x - self.x) <= self.half_w and abs(y - self.y) <= self.half_h

    # --------------------------------------------------------------- motion

    def update(self, dt: float, elapsed: float) -> None:
        """Advance by `dt` seconds, `elapsed` seconds into the run."""
        if dt <= 0:
            return
        cfg = self.config
        self.side = max(cfg.min_side, cfg.start_side - cfg.shrink_per_second * elapsed)
        self.speed = min(cfg.max_speed, cfg.start_speed + cfg.speed_ramp * elapsed)

        # Brownian heading: variance grows with dt, so the path looks the same
        # whatever the frame rate.
        self.heading += self._rng.gauss(0.0, cfg.turn_rate * math.sqrt(dt))

        step = self.speed * dt
        self.x += math.cos(self.heading) * step / cfg.aspect
        self.y += math.sin(self.heading) * step
        self._bounce()

    def _bounce(self) -> None:
        """Reflect off the margins, and clamp in case the square just shrank."""
        cfg = self.config
        margin_x = cfg.margin / cfg.aspect
        lo_x, hi_x = margin_x + self.half_w, 1 - margin_x - self.half_w
        lo_y, hi_y = cfg.margin + self.half_h, 1 - cfg.margin - self.half_h

        dx, dy = math.cos(self.heading), math.sin(self.heading)
        if self.x < lo_x or self.x > hi_x:
            dx = abs(dx) if self.x < lo_x else -abs(dx)
            self.x = min(max(self.x, lo_x), hi_x)
        if self.y < lo_y or self.y > hi_y:
            dy = abs(dy) if self.y < lo_y else -abs(dy)
            self.y = min(max(self.y, lo_y), hi_y)
        self.heading = math.atan2(dy, dx)
