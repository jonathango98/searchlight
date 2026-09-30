"""Shared look and feel for both Searchlight windows.

Arcade cabinet, not enterprise dashboard: near-black ground, a hot gold accent, and
display type heavy enough to read from across a room. Both windows pull from here
so the game screen and the leaderboard look like one machine.

Sizes are expressed as fractions of the window height rather than pixels, because
the two windows sit on different displays at different resolutions.

Fonts come from `assets/fonts/` when present (see `scripts/fetch_fonts.sh`) and
fall back to system faces otherwise, so a missing download never stops the kiosk.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pygame

Color = Tuple[int, int, int]

BG: Color = (10, 10, 18)
BG_RAISED: Color = (22, 23, 38)
BG_HOVER: Color = (34, 36, 58)
BG_SUNK: Color = (6, 6, 12)
LINE: Color = (48, 50, 76)
INK: Color = (245, 245, 250)
INK_MUTED: Color = (150, 150, 176)
INK_FAINT: Color = (80, 80, 104)

GOLD: Color = (255, 210, 48)
MAGENTA: Color = (255, 61, 139)
CYAN: Color = (86, 214, 255)
GREEN: Color = (61, 220, 145)
RED: Color = (255, 92, 92)

# Podium colours for the top three rows: gold, silver, bronze.
PODIUM: Sequence[Color] = (GOLD, (196, 202, 220), (205, 127, 50))

FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

# kind -> (bundled file, system fallback faces, fallback bold)
_FONTS = {
    "display": ("BebasNeue-Regular.ttf", "arialblack,impact,helveticaneue,arial", False),
    "body": ("Barlow-SemiBold.ttf", "avenirnext,helveticaneue,avenir,arial", True),
    "bold": ("Barlow-Bold.ttf", "avenirnext,helveticaneue,avenir,arial", True),
    "mono": ("SpaceMono-Bold.ttf", "menlo,couriernew,monaco", False),
}


# ------------------------------------------------------------------ type


@lru_cache(maxsize=256)
def font(kind: str, size: int) -> pygame.font.Font:
    """Cached font lookup. `kind` is display | body | bold | mono."""
    filename, faces, bold = _FONTS[kind]
    path = FONT_DIR / filename
    if path.exists():
        try:
            return pygame.font.Font(str(path), size)
        except (OSError, pygame.error):
            pass
    return pygame.font.SysFont(faces, size, bold=bold)


def scaled(height: int, fraction: float) -> int:
    """A size as a fraction of window height, so layouts survive a resolution change."""
    return max(10, int(height * fraction))


def text(surface: pygame.Surface, value: str, kind: str, size: int, color: Color,
         center: Optional[Tuple[int, int]] = None,
         topleft: Optional[Tuple[int, int]] = None,
         topright: Optional[Tuple[int, int]] = None,
         midleft: Optional[Tuple[int, int]] = None,
         midright: Optional[Tuple[int, int]] = None,
         midtop: Optional[Tuple[int, int]] = None,
         midbottom: Optional[Tuple[int, int]] = None,
         shadow: bool = False, alpha: int = 255,
         max_width: Optional[int] = None) -> pygame.Rect:
    """Draw a string, positioned by whichever anchor is supplied.

    `max_width` shrinks the type until it fits, for names of unknown length.
    """
    if max_width is not None:
        size = fit_size(value, kind, size, max_width)
    glyphs = font(kind, size)
    rendered = glyphs.render(value, True, color)
    rect = rendered.get_rect()
    for name, anchor in (("center", center), ("topleft", topleft),
                         ("topright", topright), ("midleft", midleft),
                         ("midright", midright), ("midtop", midtop),
                         ("midbottom", midbottom)):
        if anchor is not None:
            setattr(rect, name, anchor)
            break

    if alpha < 255:
        rendered.set_alpha(alpha)
    if shadow:
        # A hard offset shadow, not a blur: it keeps big type legible over the
        # camera feed, where the background is whatever the room happens to be.
        dark = glyphs.render(value, True, BG_SUNK)
        if alpha < 255:
            dark.set_alpha(alpha)
        offset = max(2, size // 22)
        surface.blit(dark, rect.move(offset, offset))
    surface.blit(rendered, rect)
    return rect


def fit_size(value: str, kind: str, size: int, max_width: int) -> int:
    """The largest size <= `size` at which `value` is at most `max_width` wide."""
    width = font(kind, size).size(value)[0]
    if width <= max_width or width == 0:
        return size
    return max(10, int(size * max_width / width))


# ---------------------------------------------------------------- shapes


def panel(surface: pygame.Surface, rect: pygame.Rect, color: Color = BG_RAISED,
          radius: int = 18, border: Optional[Color] = None, width: int = 2,
          alpha: int = 255) -> None:
    """A rounded box, optionally translucent and outlined."""
    if alpha >= 255:
        pygame.draw.rect(surface, color, rect, border_radius=radius)
    else:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(layer, (*color, alpha), layer.get_rect(), border_radius=radius)
        surface.blit(layer, rect.topleft)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def meter(surface: pygame.Surface, rect: pygame.Rect, fill: float, color: Color,
          track: Color = BG_SUNK, radius: Optional[int] = None) -> None:
    """A horizontal bar. `fill` is clamped to 0-1."""
    radius = rect.height // 2 if radius is None else radius
    pygame.draw.rect(surface, track, rect, border_radius=radius)
    fill = clamp01(fill)
    if fill > 0:
        filled = pygame.Rect(rect.left, rect.top, max(int(rect.width * fill), radius * 2),
                             rect.height)
        pygame.draw.rect(surface, color, filled, border_radius=radius)


def ring(surface: pygame.Surface, centre: Tuple[int, int], radius: int, width: int,
         fraction: float, color: Color, track: Optional[Color] = INK_FAINT) -> None:
    """A progress ring filling clockwise from 12 o'clock.

    Built from a polygon rather than `pygame.draw.arc`, which leaves pinholes in
    thick arcs.
    """
    if track is not None:
        pygame.draw.circle(surface, track, centre, radius, width)
    fraction = clamp01(fraction)
    if fraction <= 0:
        return
    steps = max(8, int(96 * fraction))
    outer, inner = [], []
    for i in range(steps + 1):
        angle = -math.pi / 2 + 2 * math.pi * fraction * i / steps
        c, s = math.cos(angle), math.sin(angle)
        outer.append((centre[0] + c * radius, centre[1] + s * radius))
        inner.append((centre[0] + c * (radius - width), centre[1] + s * (radius - width)))
    pygame.draw.polygon(surface, color, outer + inner[::-1])
    # Round caps.
    cap = width / 2
    for (ox, oy), (ix, iy) in ((outer[0], inner[0]), (outer[-1], inner[-1])):
        pygame.draw.circle(surface, color, (int((ox + ix) / 2), int((oy + iy) / 2)),
                           int(cap))


def glow(surface: pygame.Surface, rect: pygame.Rect, color: Color, spread: int,
         radius: int = 0, intensity: int = 120, width: int = 0) -> None:
    """A soft halo around `rect`, faked by drawing small and smoothscaling up.

    pygame 2.6 has no blur, but a shape drawn at 1/4 size and scaled back up
    comes out convincingly soft - and cheaply enough to do every frame.
    """
    if spread <= 0 or rect.width <= 0 or rect.height <= 0:
        return
    pad = spread * 2
    full = (rect.width + pad * 2, rect.height + pad * 2)
    shrink = 4
    small = pygame.Surface((max(1, full[0] // shrink), max(1, full[1] // shrink)),
                           pygame.SRCALPHA)
    inner = pygame.Rect(pad // shrink, pad // shrink,
                        max(1, rect.width // shrink), max(1, rect.height // shrink))
    grown = inner.inflate(spread // shrink, spread // shrink)
    pygame.draw.rect(small, (*color, intensity), grown,
                     width=max(1, (width + spread) // shrink) if width else 0,
                     border_radius=max(0, (radius + spread // 2) // shrink))
    soft = pygame.transform.smoothscale(small, full)
    # A second pass spreads the falloff further.
    soft = pygame.transform.smoothscale(
        pygame.transform.smoothscale(soft, (max(1, full[0] // 2), max(1, full[1] // 2))),
        full)
    surface.blit(soft, (rect.left - pad, rect.top - pad))


@lru_cache(maxsize=16)
def vertical_gradient(size: Tuple[int, int], top: Color, bottom: Color) -> pygame.Surface:
    """A cached top-to-bottom gradient surface."""
    width, height = size
    column = pygame.Surface((1, max(1, height)))
    for y in range(max(1, height)):
        column.set_at((0, y), mix(top, bottom, y / max(1, height - 1)))
    return pygame.transform.scale(column, (max(1, width), max(1, height)))


def backdrop(surface: pygame.Surface) -> None:
    """The standard window background: a faint gradient rather than flat black."""
    surface.blit(vertical_gradient(surface.get_size(), (16, 15, 30), BG_SUNK), (0, 0))


def draw_heart(surface: pygame.Surface, centre: Tuple[int, int], size: int,
               colour: Color) -> None:
    """Two circles and a triangle - no font on the kiosk is trusted to have a heart."""
    cx, cy = centre
    r = max(2, size // 4)
    pygame.draw.circle(surface, colour, (cx - r, cy - r // 2), r)
    pygame.draw.circle(surface, colour, (cx + r, cy - r // 2), r)
    pygame.draw.polygon(surface, colour, [(cx - 2 * r, cy - r // 3),
                                          (cx + 2 * r, cy - r // 3),
                                          (cx, cy + int(1.8 * r))])


# ---------------------------------------------------------------- colour


def mix(a: Color, b: Color, amount: float) -> Color:
    """Blend two colours; `amount` 0 gives a, 1 gives b."""
    amount = clamp01(amount)
    return tuple(int(x + (y - x) * amount) for x, y in zip(a, b))  # type: ignore[return-value]


def fit_surface(image: pygame.Surface, target: pygame.Rect) -> Tuple[pygame.Surface, pygame.Rect]:
    """Scale to *cover* the target rect, preserving aspect, centred (letterbox-free)."""
    scale = max(target.width / image.get_width(), target.height / image.get_height())
    size = (int(image.get_width() * scale), int(image.get_height() * scale))
    scaled_image = pygame.transform.smoothscale(image, size)
    rect = scaled_image.get_rect(center=target.center)
    return scaled_image, rect


def dim(surface: pygame.Surface, rect: pygame.Rect, alpha: int,
        color: Color = BG) -> None:
    """Lay a translucent wash over a region, to push the camera feed back."""
    veil = pygame.Surface(rect.size, pygame.SRCALPHA)
    veil.fill((*color, alpha))
    surface.blit(veil, rect.topleft)


# ---------------------------------------------------------------- motion


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Overshoots slightly then settles - for things that pop in."""
    t = clamp01(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def pulse(now: float, speed: float = 2.4) -> float:
    """0-1 sine for breathing highlights."""
    return 0.5 + 0.5 * math.sin(now * speed)
