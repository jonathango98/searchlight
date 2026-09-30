"""Motion and light for the game window: tweens, confetti, transitions, the spotlight.

Everything here redraws at 60 fps over a full-window camera frame, so the rule is
*build once, blit often*: anything whose cost depends on a size (the soft hole in
the veil, a glow, a vignette) is rendered once per size and cached, and each
frame only blits it. pygame 2.6 has no blur, so softness is made either with a
numpy distance field (exact, for the spotlight edge) or by drawing small and
smoothscaling up (cheap, for glows).

Confetti lives in *height units* - x in [0, aspect], y in [0, 1] - so a burst
started before the window size is known, or across a resize, still lands right.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pygame

from .ui import theme

Color = Tuple[int, int, int]


# ------------------------------------------------------------------ tweens


def progress(start: float, duration: float, now: float) -> float:
    """How far through `duration` seconds from `start` we are, clamped to 0-1."""
    if duration <= 0:
        return 1.0
    return theme.clamp01((now - start) / duration)


def smoothstep(t: float) -> float:
    t = theme.clamp01(t)
    return t * t * (3 - 2 * t)


def approach(current: float, target: float, dt: float, rate: float) -> float:
    """Exponential ease toward `target`, frame-rate independent.

    `rate` is roughly "fraction per second" on a log scale: 10 closes ~63% of the
    gap in 0.1s. Used for anything that should glide rather than snap - the head
    marker, the lock ring - without keeping a tween object per value.
    """
    if dt <= 0:
        return current
    return target + (current - target) * math.exp(-rate * dt)


class Tween:
    """A value moving from `a` to `b` over `duration`, with an easing curve."""

    def __init__(self, a: float, b: float, duration: float, start: float,
                 ease: Callable[[float], float] = theme.ease_out_cubic) -> None:
        self.a, self.b = a, b
        self.duration = duration
        self.start = start
        self.ease = ease

    def value(self, now: float) -> float:
        return theme.lerp(self.a, self.b, self.ease(progress(self.start, self.duration, now)))

    def done(self, now: float) -> bool:
        return now - self.start >= self.duration


# --------------------------------------------------------------- confetti

CONFETTI_COLOURS: Sequence[Color] = (theme.GOLD, theme.MAGENTA, theme.CYAN, theme.GREEN,
                                     theme.INK, (255, 140, 60))


@dataclass
class _Piece:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    spin: float
    flip: float          # phase of the tumble that fakes 3D
    flip_speed: float
    sway: float
    w: float
    h: float
    colour: Color
    born: float
    life: float


class Confetti:
    """A burst of tumbling paper. Cheap: one small polygon per piece per frame."""

    GRAVITY = 0.9        # heights / s^2
    DRAG = 1.6           # 1/s - air resistance, so pieces float rather than fall
    FLUTTER = 0.10       # heights / s of side-to-side drift once they slow down

    def __init__(self, seed: Optional[int] = None) -> None:
        self.pieces: List[_Piece] = []
        self._rng = random.Random(seed)
        self._last: Optional[float] = None

    @property
    def alive(self) -> bool:
        return bool(self.pieces)

    def clear(self) -> None:
        self.pieces.clear()
        self._last = None

    def burst(self, aspect: float, now: float, count: int = 180,
              colours: Sequence[Color] = CONFETTI_COLOURS) -> None:
        """Two cannons from the bottom corners plus a curtain falling from the top."""
        rng = self._rng
        cannons = int(count * 0.6)
        for i in range(count):
            if i < cannons:
                left = i % 2 == 0
                x = 0.02 * aspect if left else 0.98 * aspect
                y = 1.02
                angle = math.radians(rng.uniform(58, 80))
                speed = rng.uniform(1.3, 2.2)
                vx = math.cos(angle) * speed * (1 if left else -1)
                vy = -math.sin(angle) * speed
                born = now + rng.uniform(0.0, 0.18)
            else:
                x = rng.uniform(0.0, aspect)
                y = rng.uniform(-0.25, -0.02)
                vx = rng.uniform(-0.15, 0.15)
                vy = rng.uniform(0.0, 0.25)
                born = now + rng.uniform(0.0, 0.6)
            self.pieces.append(_Piece(
                x=x, y=y, vx=vx, vy=vy,
                angle=rng.uniform(0, math.tau), spin=rng.uniform(-9, 9),
                flip=rng.uniform(0, math.tau), flip_speed=rng.uniform(6, 14),
                sway=rng.uniform(0, math.tau),
                w=rng.uniform(0.010, 0.016), h=rng.uniform(0.005, 0.009),
                colour=rng.choice(list(colours)), born=born,
                life=rng.uniform(3.2, 4.8)))
        if self._last is None:
            self._last = now

    def update(self, now: float) -> None:
        if self._last is None:
            self._last = now
            return
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        if dt == 0:
            return
        damp = math.exp(-self.DRAG * dt)
        keep = []
        for p in self.pieces:
            if now < p.born:
                keep.append(p)
                continue
            age = now - p.born
            if age > p.life or p.y > 1.1:
                continue
            p.vx *= damp
            p.vy = p.vy * damp + self.GRAVITY * dt
            p.x += (p.vx + math.sin(age * 3.1 + p.sway) * self.FLUTTER) * dt
            p.y += p.vy * dt
            p.angle += p.spin * dt
            p.flip += p.flip_speed * dt
            keep.append(p)
        self.pieces = keep

    def draw(self, surface: pygame.Surface, now: float) -> None:
        height = surface.get_height()
        for p in self.pieces:
            if now < p.born:
                continue
            remaining = p.life - (now - p.born)
            shrink = theme.clamp01(remaining / 0.6)          # shrink away at the end
            tumble = math.cos(p.flip)
            w = p.w * height * max(0.15, abs(tumble)) * shrink
            h = p.h * height * shrink
            if w < 0.5 or h < 0.5:
                continue
            colour = p.colour if tumble >= 0 else theme.mix(p.colour, theme.BG_SUNK, 0.35)
            c, s = math.cos(p.angle), math.sin(p.angle)
            cx, cy = p.x * height, p.y * height
            hw, hh = w / 2, h / 2
            points = [(cx + dx * c - dy * s, cy + dx * s + dy * c)
                      for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            pygame.draw.polygon(surface, colour, points)


# ------------------------------------------------------------ transitions


class Transition:
    """Cross-fade from a snapshot of the previous screen, drifting it away.

    The caller hands over a copy of what was on screen when the state changed;
    each frame the new screen is drawn first and the snapshot is laid over it,
    fading out. Surface-level alpha on an opaque surface is the cheapest blend
    SDL has, so this costs one full-window blit for a third of a second.
    """

    def __init__(self, duration: float = 0.35) -> None:
        self.duration = duration
        self.snapshot: Optional[pygame.Surface] = None
        self.started = 0.0
        self.slide = 0.0

    def begin(self, snapshot: pygame.Surface, now: float, slide: float = 0.0,
              duration: Optional[float] = None) -> None:
        self.snapshot = snapshot
        self.started = now
        self.slide = slide
        if duration is not None:
            self.duration = duration

    def active(self, now: float) -> bool:
        return self.snapshot is not None and now - self.started < self.duration

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if self.snapshot is None:
            return
        t = progress(self.started, self.duration, now)
        if t >= 1 or self.snapshot.get_size() != surface.get_size():
            self.snapshot = None
            return
        self.snapshot.set_alpha(int(255 * (1 - theme.ease_in_out_cubic(t))))
        offset = int(-self.slide * surface.get_height() * theme.ease_out_cubic(t))
        surface.blit(self.snapshot, (0, offset))


# ------------------------------------------------------------- spotlight


def _rounded_rect_distance(size: Tuple[int, int], rect_size: Tuple[int, int],
                           radius: float) -> np.ndarray:
    """Signed distance (pixels, negative inside) from each pixel of a `size` grid to
    a rounded rectangle centred in it. Indexed [x, y] to match pygame.surfarray."""
    pw, ph = size
    w, h = rect_size
    xs = np.abs(np.arange(pw, dtype=np.float32) + 0.5 - pw / 2)[:, None]
    ys = np.abs(np.arange(ph, dtype=np.float32) + 0.5 - ph / 2)[None, :]
    r = float(min(radius, w / 2, h / 2))
    qx = xs - (w / 2 - r)
    qy = ys - (h / 2 - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def _alpha_surface(alpha: np.ndarray, colour: Color) -> pygame.Surface:
    surface = pygame.Surface(alpha.shape, pygame.SRCALPHA)
    surface.fill((*colour, 0))
    pixels = pygame.surfarray.pixels_alpha(surface)
    pixels[:] = np.clip(alpha, 0, 255).astype(np.uint8)
    del pixels                                   # unlock the surface
    return surface


@lru_cache(maxsize=16)
def hole_patch(size: Tuple[int, int], radius: int, feather: int, colour: Color,
               alpha: int) -> Tuple[pygame.Surface, int]:
    """The veil around a rounded-rect hole, with an outward soft edge.

    Returns the patch and its margin: the patch is `size` grown by the margin on
    every side, fully clear inside the rect and exactly `alpha` at its own edge,
    so it butts seamlessly against the solid veil around it.
    """
    w, h = max(1, size[0]), max(1, size[1])
    margin = feather + 2
    distance = _rounded_rect_distance((w + 2 * margin, h + 2 * margin), (w, h), radius)
    t = np.clip(distance / max(1.0, float(feather)), 0.0, 1.0)
    return _alpha_surface(t * t * (3 - 2 * t) * alpha, colour), margin


@lru_cache(maxsize=16)
def border_mask(size: Tuple[int, int], radius: int, width: int) -> pygame.Surface:
    """An anti-aliased rounded-rect outline, white, lying just inside `size`.

    `pygame.draw.rect` with both a width and a radius leaves notches at the
    corners, which show badly on the one shape the player stares at all game.
    """
    pad = 2
    distance = _rounded_rect_distance((size[0] + 2 * pad, size[1] + 2 * pad), size, radius)
    coverage = np.minimum(0.5 - distance, distance + width + 0.5)
    return _alpha_surface(coverage * 255, (255, 255, 255))


_TINTS: "OrderedDict[tuple, Tuple[pygame.Surface, pygame.Surface]]" = OrderedDict()


def _tinted(mask: pygame.Surface, colour: Color, flags: int) -> pygame.Surface:
    """`mask` multiplied by `colour`, cached.

    The multiply is the expensive part of a glow (a full pass over a few hundred
    thousand pixels), and the square's colour is steady most of the time, so the
    colour is quantised a little and the last couple of dozen results are kept.
    """
    colour = tuple(min(255, (c + 4) // 8 * 8) for c in colour)  # type: ignore[assignment]
    key = (id(mask), colour, flags)
    hit = _TINTS.get(key)
    if hit is not None and hit[0] is mask:
        _TINTS.move_to_end(key)
        return hit[1]
    tinted = mask.copy()
    fill = (*colour, 255) if flags == pygame.BLEND_RGBA_MULT else colour
    tinted.fill(fill, special_flags=flags)
    _TINTS[key] = (mask, tinted)          # holding `mask` keeps its id unique
    while len(_TINTS) > 24:
        _TINTS.popitem(last=False)
    return tinted


def draw_border(surface: pygame.Surface, rect: pygame.Rect, colour: Color, radius: int,
                width: int) -> None:
    """A crisp, anti-aliased rounded outline just inside `rect`."""
    mask = border_mask(rect.size, radius, width)
    surface.blit(_tinted(mask, colour, pygame.BLEND_RGBA_MULT), (rect.left - 2, rect.top - 2))


class Veil:
    """A dark wash over the camera with a soft-edged hole where the light is.

    The solid part is one opaque surface with surface-level alpha, blitted as up
    to four strips around the hole; only the hole's neighbourhood uses per-pixel
    alpha, from `hole_patch`, cached per (rounded) size.
    """

    def __init__(self, colour: Color = theme.BG_SUNK, alpha: int = 200) -> None:
        self.colour = colour
        self.alpha = alpha
        self._solid: Optional[pygame.Surface] = None

    def _solid_for(self, size: Tuple[int, int]) -> pygame.Surface:
        if self._solid is None or self._solid.get_size() != size:
            self._solid = pygame.Surface(size).convert() if pygame.display.get_surface() \
                else pygame.Surface(size)
            self._solid.fill(self.colour)
            self._solid.set_alpha(self.alpha)
        return self._solid

    def draw(self, surface: pygame.Surface, hole: Optional[pygame.Rect] = None,
             radius: int = 0, feather: int = 0) -> None:
        screen = surface.get_rect()
        solid = self._solid_for(screen.size)
        if hole is None or hole.width <= 0 or hole.height <= 0:
            surface.blit(solid, (0, 0))
            return
        patch, margin = hole_patch(hole.size, radius, feather, self.colour, self.alpha)
        area = pygame.Rect(hole.left - margin, hole.top - margin,
                           patch.get_width(), patch.get_height())
        strips = (pygame.Rect(0, 0, screen.width, area.top),
                  pygame.Rect(0, area.bottom, screen.width, screen.height - area.bottom),
                  pygame.Rect(0, area.top, area.left, area.height),
                  pygame.Rect(area.right, area.top, screen.width - area.right, area.height))
        for strip in strips:
            strip = strip.clip(screen)
            if strip.width > 0 and strip.height > 0:
                surface.blit(solid, strip.topleft, area=strip)
        surface.blit(patch, area.topleft)


@lru_cache(maxsize=16)
def glow_mask(size: Tuple[int, int], radius: int, spread: int, line: int) -> pygame.Surface:
    """A white halo hugging a rounded-rect outline, on black, for additive blending.

    Built from the same distance field as the spotlight hole, so the halo follows
    the corners exactly. Opaque RGB on purpose: added onto the frame, black adds
    nothing and white adds full light.
    """
    pad = spread * 2
    full = (size[0] + pad * 2, size[1] + pad * 2)
    distance = _rounded_rect_distance(full, size, radius)
    away = np.abs(distance + line / 2) / max(1.0, float(spread))
    light = np.exp(-away * away * 1.6) * 255
    glow = pygame.Surface(full)
    pixels = pygame.surfarray.pixels3d(glow)
    pixels[:] = light.astype(np.uint8)[:, :, None]
    del pixels
    return glow


def draw_glow(surface: pygame.Surface, rect: pygame.Rect, colour: Color, radius: int,
              spread: int, line: int = 2, intensity: float = 1.0) -> None:
    """Add a coloured halo around `rect`. Size-bucketed so it stays cached while the
    square shrinks; the colour and intensity come from a small tint cache."""
    if rect.width <= 0 or rect.height <= 0 or intensity <= 0:
        return
    bucket = (max(4, rect.width // 4 * 4), max(4, rect.height // 4 * 4))
    mask = glow_mask(bucket, radius, spread, line)
    k = round(theme.clamp01(intensity) * 20) / 20
    tinted = _tinted(mask, tuple(int(c * k) for c in colour), pygame.BLEND_RGB_MULT)
    pad = spread * 2
    left = rect.centerx - bucket[0] // 2 - pad
    top = rect.centery - bucket[1] // 2 - pad
    surface.blit(tinted, (left, top), special_flags=pygame.BLEND_RGB_ADD)


@lru_cache(maxsize=4)
def vignette(size: Tuple[int, int], colour: Color) -> pygame.Surface:
    """Colour creeping in from the window edges - the "danger" wash."""
    w, h = max(1, size[0] // 8), max(1, size[1] // 8)
    xs = np.linspace(-1, 1, w, dtype=np.float32)[:, None]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[None, :]
    edge = np.maximum(np.abs(xs) ** 3, np.abs(ys) ** 3)
    small = pygame.Surface((w, h), pygame.SRCALPHA)
    small.fill((*colour, 0))
    alphas = pygame.surfarray.pixels_alpha(small)
    alphas[:] = (np.clip(edge * 1.15, 0, 1) * 220).astype(np.uint8)
    del alphas
    return pygame.transform.smoothscale(small, size)


def draw_vignette(surface: pygame.Surface, colour: Color, amount: float) -> None:
    if amount <= 0.01:
        return
    layer = vignette(surface.get_size(), colour)
    layer.set_alpha(int(255 * theme.clamp01(amount)))
    surface.blit(layer, (0, 0))
