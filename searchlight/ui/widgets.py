"""A small, reusable pygame widget kit for the Searchlight kiosk UI.

Everything here is keyboard-and-mouse first: every widget can be driven by a
click as well as by focus + keys, because the kiosk has both attached. Sizes
are never hard-coded in pixels - callers derive rects from `theme.scaled` and
hand them to `layout()`, so the same widget looks right from 1280x720 to 4K.

Two things are built on top of this module: an in-game settings overlay (rows
generated from `searchlight.settings.SETTINGS`) and a leaderboard edit mode
(a `ScrollList` of entries with inline `TextField`/`NumberField` editing and
`ConfirmDialog`s for delete/clear). Both need widgets that look alive - hover,
focus and press states ease over ~120ms using the `now` (seconds, monotonic)
passed into every `draw()` - and that degrade gracefully when disabled.

Event flow used by every widget:

    consumed = widget.handle(event)   # event is a pygame.event.Event
    widget.draw(surface, now)         # now is seconds, e.g. time.monotonic()

A container (the settings overlay, the leaderboard edit screen, ...) owns a
list of widgets plus a `FocusManager`. Per event: call `focus_manager.handle`
first (it moves focus on Tab/Shift+Tab and on a click inside a focusable
widget's rect), then offer the event to whichever widgets are visible until
one returns True. Per frame: call `draw` on each visible widget.

Nothing here touches `searchlight.store` or `searchlight.settings` directly -
`Slider`/`Choice` optionally *bind* to a `searchlight.settings.Setting` (duck
typed: anything with `.kind`, `.min`, `.max`, `.step`, `.choices`, `.format`,
`.nudge`, `.fraction`, `.from_fraction` works), but the module has no import
of that package, so it stays reusable for both windows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import pygame

from . import theme

Color = theme.Color

# Widgets ease state changes (hover, focus, press) over this many seconds.
ANIM = 0.12


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clampi(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _mouse_pos(event: pygame.event.Event) -> Tuple[int, int]:
    """The pointer position for `event`, even for events that don't carry one."""
    pos = getattr(event, "pos", None)
    if pos is not None:
        return pos
    return pygame.mouse.get_pos()


# ------------------------------------------------------------------ motion


@dataclass
class Anim:
    """A single value that eases toward a target whenever the target changes.

    Call `set(target, now)` when the target changes (e.g. hover starts) and
    `update(now)` once per draw to get the current eased value.
    """

    value: float = 0.0
    _target: float = 0.0
    _start: float = 0.0
    _since: float = 0.0

    def __post_init__(self) -> None:
        self._target = self.value
        self._start = self.value

    def set(self, target: float, now: float) -> None:
        if target == self._target:
            return
        self._start = self.value
        self._target = target
        self._since = now

    def snap(self, value: float) -> None:
        self.value = self._start = self._target = value

    def update(self, now: float, duration: float = ANIM,
               ease: Callable[[float], float] = theme.ease_out_cubic) -> float:
        t = 1.0 if duration <= 0 else theme.clamp01((now - self._since) / duration)
        self.value = theme.lerp(self._start, self._target, ease(t))
        return self.value


def focus_ring(surface: pygame.Surface, rect: pygame.Rect, amount: float,
               radius: int = 10) -> None:
    """A gold outline that fades in around `rect` as focus is gained."""
    if amount <= 0.001:
        return
    glow_rect = rect.inflate(int(6 * amount), int(6 * amount))
    # An outline halo (width > 0): a filled one would wash the whole body of a
    # large widget, such as a focused ScrollList, in gold.
    theme.glow(surface, glow_rect, theme.GOLD, spread=max(2, int(8 * amount)),
               radius=radius, intensity=int(90 * amount), width=2)
    pygame.draw.rect(surface, theme.GOLD, glow_rect, width=2, border_radius=radius)


# ------------------------------------------------------------------- layout


def split_row(rect: pygame.Rect, label_width: int, gap: int = 16) -> Tuple[pygame.Rect, pygame.Rect]:
    """`rect` split left/right into (label_rect, control_rect)."""
    label_rect = pygame.Rect(rect.left, rect.top, label_width, rect.height)
    control_rect = pygame.Rect(rect.left + label_width + gap, rect.top,
                                max(0, rect.width - label_width - gap), rect.height)
    return label_rect, control_rect


def stack_rows(rect: pygame.Rect, heights: Sequence[int], gap: int = 0) -> List[pygame.Rect]:
    """`rect` split top-to-bottom into rows of the given heights."""
    rows = []
    y = rect.top
    for h in heights:
        rows.append(pygame.Rect(rect.left, y, rect.width, h))
        y += h + gap
    return rows


def wrap_text(value: str, kind: str, size: int, max_width: int) -> List[str]:
    """`value` broken into lines that each fit `max_width` at font `size`."""
    words = value.split()
    if not words:
        return [""]
    glyphs = theme.font(kind, size)
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and glyphs.size(trial)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return lines


def split_cols(rect: pygame.Rect, widths: Sequence[int], gap: int = 0) -> List[pygame.Rect]:
    """`rect` split left-to-right into columns of the given widths."""
    cols = []
    x = rect.left
    for w in widths:
        cols.append(pygame.Rect(x, rect.top, w, rect.height))
        x += w + gap
    return cols


# -------------------------------------------------------------------- base


class Widget:
    """Base class for every interactive control.

    Subclasses override `layout`, `handle` and `draw`. `enabled=False` grays
    the widget out and makes `handle` a no-op. `focusable=False` (class or
    instance attribute) excludes a widget from `FocusManager` tabbing, for
    things like static labels wrapped as widgets.
    """

    focusable: bool = True

    def __init__(self, rect: Optional[pygame.Rect] = None) -> None:
        self.rect = pygame.Rect(rect) if rect is not None else pygame.Rect(0, 0, 0, 0)
        self.enabled = True
        self.visible = True
        self.focused = False
        self._hover = False
        self._hover_anim = Anim()
        self._focus_anim = Anim()
        self._press_anim = Anim()

    def layout(self, rect: pygame.Rect) -> None:
        self.rect = pygame.Rect(rect)

    def handle(self, event: pygame.event.Event) -> bool:  # pragma: no cover - abstract
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:  # pragma: no cover - abstract
        pass

    # -- shared animation bookkeeping, called by subclasses in handle/draw --

    def _update_state_anims(self, now: float, hovered: Optional[bool] = None,
                             pressed: bool = False) -> Tuple[float, float, float]:
        if hovered is not None:
            self._hover = hovered
        self._hover_anim.set(1.0 if (self._hover and self.enabled) else 0.0, now)
        self._focus_anim.set(1.0 if self.focused else 0.0, now)
        self._press_anim.set(1.0 if (pressed and self.enabled) else 0.0, now)
        return (self._hover_anim.update(now), self._focus_anim.update(now),
                self._press_anim.update(now))

    def _active(self, event: pygame.event.Event) -> bool:
        return self.enabled and self.visible


# ------------------------------------------------------------------ button


class Button(Widget):
    """A clickable button. `variant` is "primary" | "secondary" | "danger".

    `on_click()` fires on a left click released inside the button, or on
    Enter/Space while focused. Call `.click()` to trigger it programmatically
    (what Enter/Space do internally) - handy for tests and for a dialog's
    default button.
    """

    def __init__(self, label: str, on_click: Optional[Callable[[], None]] = None, *,
                 variant: str = "secondary", rect: Optional[pygame.Rect] = None,
                 enabled: bool = True) -> None:
        super().__init__(rect)
        self.label = label
        self.on_click = on_click
        self.variant = variant
        self.enabled = enabled
        self._down = False

    def click(self) -> None:
        if self.enabled and self.on_click is not None:
            self.on_click()

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._down = True
                return True
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_down = self._down
            self._down = False
            if was_down and self.rect.collidepoint(event.pos):
                self.click()
                return True
            return False
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.click()
                return True
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        hover, focus, press = self._update_state_anims(now, pressed=self._down)
        base = {"primary": theme.GOLD, "secondary": theme.BG_RAISED, "danger": theme.RED}[self.variant]
        ink = theme.BG if self.variant == "primary" else theme.INK
        if not self.enabled:
            base, ink = theme.BG_RAISED, theme.INK_FAINT
        fill = theme.mix(base, theme.BG_HOVER if self.variant == "secondary" else theme.mix(base, (255, 255, 255), 0.15),
                          hover * 0.6) if self.enabled else base
        rect = self.rect.move(0, int(press * 2))
        radius = max(6, self.rect.height // 3)
        # Only a resting-state hint of glow (it would read as a muddy smudge
        # under a gold button otherwise); it blooms on hover/focus/press.
        glow_amount = max(hover, focus, press)
        if self.enabled and self.variant != "secondary" and glow_amount > 0.02:
            theme.glow(surface, rect, base, spread=int(3 + 7 * glow_amount), radius=radius,
                       intensity=int(20 + 55 * glow_amount))
        theme.panel(surface, rect, fill, radius=radius,
                    border=theme.LINE if self.variant == "secondary" else None, width=2)
        focus_ring(surface, rect.inflate(4, 4), focus, radius=radius + 2)
        size = max(12, int(self.rect.height * 0.42))
        theme.text(surface, self.label, "bold", size, ink, center=rect.center, max_width=rect.width - 16)


# ------------------------------------------------------------------ toggle


class Toggle(Widget):
    """An on/off switch. `on_change(value)` fires whenever the value flips."""

    def __init__(self, value: bool = False, on_change: Optional[Callable[[bool], None]] = None,
                 *, rect: Optional[pygame.Rect] = None) -> None:
        super().__init__(rect)
        self.value = bool(value)
        self.on_change = on_change
        self._knob = Anim(1.0 if self.value else 0.0)

    def set_value(self, value: bool, notify: bool = True) -> None:
        value = bool(value)
        if value == self.value:
            return
        self.value = value
        if notify and self.on_change is not None:
            self.on_change(self.value)

    def _flip(self) -> None:
        self.set_value(not self.value)

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._flip()
                return True
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE,
                              pygame.K_LEFT, pygame.K_RIGHT):
                self._flip()
                return True
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        hover, focus, _ = self._update_state_anims(now)
        self._knob.set(1.0 if self.value else 0.0, now)
        knob_t = self._knob.update(now, duration=0.16, ease=theme.ease_out_back)
        track = self.rect
        on_color = theme.GOLD if self.enabled else theme.INK_FAINT
        off_color = theme.BG_SUNK
        fill = theme.mix(off_color, on_color, theme.clamp01(knob_t))
        pygame.draw.rect(surface, fill, track, border_radius=track.height // 2)
        pygame.draw.rect(surface, theme.mix(theme.LINE, theme.INK, hover * 0.3), track,
                          width=2, border_radius=track.height // 2)
        r = track.height // 2 - 3
        cx = int(theme.lerp(track.left + track.height // 2, track.right - track.height // 2,
                             theme.clamp01(knob_t)))
        pygame.draw.circle(surface, theme.INK, (cx, track.centery), r)
        focus_ring(surface, track.inflate(6, 6), focus, radius=track.height // 2 + 3)


# ----------------------------------------------------------------- choice


class Choice(Widget):
    """A segmented control. `options` is ((stored_value, label), ...).

    Left/Right cycle through options when focused (wrapping); a click on a
    segment jumps straight to it. `on_change(value)` fires with the stored
    value, not the label.
    """

    def __init__(self, options: Sequence[Tuple[str, str]], value: str,
                 on_change: Optional[Callable[[str], None]] = None, *,
                 rect: Optional[pygame.Rect] = None) -> None:
        super().__init__(rect)
        self.options = list(options)
        self.value = value
        self.on_change = on_change

    def set_value(self, value: str, notify: bool = True) -> None:
        if value == self.value:
            return
        self.value = value
        if notify and self.on_change is not None:
            self.on_change(self.value)

    def _index(self) -> int:
        for i, (v, _) in enumerate(self.options):
            if v == self.value:
                return i
        return 0

    def _cycle(self, steps: int) -> None:
        if not self.options:
            return
        i = (self._index() + steps) % len(self.options)
        self.set_value(self.options[i][0])

    def _segments(self) -> List[pygame.Rect]:
        n = max(1, len(self.options))
        w = self.rect.width / n
        return [pygame.Rect(int(self.rect.left + i * w), self.rect.top, int(w), self.rect.height)
                for i in range(n)]

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, seg in enumerate(self._segments()):
                if seg.collidepoint(event.pos):
                    self.set_value(self.options[i][0])
                    return True
            return False
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_LEFT:
                self._cycle(-1)
                return True
            if event.key == pygame.K_RIGHT:
                self._cycle(1)
                return True
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        _, focus, _ = self._update_state_anims(now)
        radius = max(6, self.rect.height // 3)
        theme.panel(surface, self.rect, theme.BG_SUNK, radius=radius, border=theme.LINE, width=2)
        active = self._index()
        segs = self._segments()
        if segs:
            pad = 3
            active_rect = segs[active].inflate(-pad * 2, -pad * 2)
            theme.panel(surface, active_rect, theme.GOLD if self.enabled else theme.INK_FAINT,
                        radius=max(4, radius - pad))
        for i, (seg, (_, label)) in enumerate(zip(segs, self.options)):
            ink = theme.BG if (i == active and self.enabled) else theme.INK_MUTED
            theme.text(surface, label, "bold", max(13, int(self.rect.height * 0.62)), ink,
                       center=seg.center, max_width=seg.width - 8)
        focus_ring(surface, self.rect.inflate(4, 4), focus, radius=radius + 2)


# ------------------------------------------------------------------ slider


class Slider(Widget):
    """A numeric slider: drag, click-to-position, arrow keys, mouse wheel.

    Either bind to a `searchlight.settings.Setting`-shaped object via `spec=`
    (uses its min/max/step/kind/format/nudge/fraction/from_fraction), or pass
    `min_value`/`max_value`/`step`/`format_fn` directly for a generic slider.
    `on_change(value)` fires on every value change (while dragging too).

    `show_value=False` suppresses the slider's own inline value label - for
    callers (like a settings row) that already show the formatted value in
    their own column and don't want it drawn twice.
    """

    def __init__(self, value: Any, *, min_value: float = 0.0, max_value: float = 1.0,
                 step: Optional[float] = None, integer: bool = False,
                 format_fn: Optional[Callable[[Any], str]] = None,
                 spec: Any = None, on_change: Optional[Callable[[Any], None]] = None,
                 show_value: bool = True, rect: Optional[pygame.Rect] = None) -> None:
        super().__init__(rect)
        self.spec = spec
        self.min_value = spec.min if spec is not None else min_value
        self.max_value = spec.max if spec is not None else max_value
        self.step = spec.step if spec is not None else step
        self.integer = (spec.kind == "int") if spec is not None else integer
        self._format_fn = format_fn
        self.on_change = on_change
        self.show_value = show_value
        self.value = value
        self._dragging = False

    # -- spec-aware helpers --------------------------------------------

    def _fraction(self, value: Any) -> float:
        if self.spec is not None:
            return self.spec.fraction(value)
        if self.max_value == self.min_value:
            return 0.0
        return clamp((float(value) - self.min_value) / (self.max_value - self.min_value), 0.0, 1.0)

    def _from_fraction(self, fraction: float) -> Any:
        if self.spec is not None:
            return self.spec.from_fraction(fraction)
        raw = self.min_value + clamp(fraction, 0.0, 1.0) * (self.max_value - self.min_value)
        return self._clamp(raw)

    def _clamp(self, value: float) -> Any:
        if self.spec is not None:
            return self.spec.clamp(value)
        value = clamp(value, self.min_value, self.max_value)
        if self.step:
            value = self.min_value + round((value - self.min_value) / self.step) * self.step
            value = clamp(value, self.min_value, self.max_value)
        return int(round(value)) if self.integer else round(value, 6)

    def _nudge(self, value: Any, steps: int) -> Any:
        if self.spec is not None:
            return self.spec.nudge(value, steps)
        return self._clamp(value + steps * (self.step or 1))

    def _format(self, value: Any) -> str:
        if self.spec is not None:
            return self.spec.format(value)
        if self._format_fn is not None:
            return self._format_fn(value)
        return f"{value:g}"

    def set_value(self, value: Any, notify: bool = True) -> None:
        value = self._clamp(value) if not isinstance(value, str) else value
        if value == self.value:
            return
        self.value = value
        if notify and self.on_change is not None:
            self.on_change(self.value)

    # -- track geometry ---------------------------------------------------

    def _track_rect(self) -> pygame.Rect:
        h = max(8, int(self.rect.height * 0.34))
        return pygame.Rect(self.rect.left, self.rect.centery - h // 2, self.rect.width, h)

    def _fraction_at(self, x: int) -> float:
        if self.rect.width == 0:
            return 0.0
        return clamp((x - self.rect.left) / self.rect.width, 0.0, 1.0)

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            if self._dragging:
                self.set_value(self._from_fraction(self._fraction_at(event.pos[0])))
                return True
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._dragging = True
                self.set_value(self._from_fraction(self._fraction_at(event.pos[0])))
                return True
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging = False
                return True
            return False
        if event.type == pygame.MOUSEWHEEL:
            pos = _mouse_pos(event)
            if self.rect.collidepoint(pos):
                self.set_value(self._nudge(self.value, 1 if event.y > 0 else -1))
                return True
            return False
        if event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_LEFT, pygame.K_DOWN):
                self.set_value(self._nudge(self.value, -1))
                return True
            if event.key in (pygame.K_RIGHT, pygame.K_UP):
                self.set_value(self._nudge(self.value, 1))
                return True
            if event.key == pygame.K_HOME:
                self.set_value(self.min_value)
                return True
            if event.key == pygame.K_END:
                self.set_value(self.max_value)
                return True
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        hover, focus, press = self._update_state_anims(now, pressed=self._dragging)
        track = self._track_rect()
        color = theme.GOLD if self.enabled else theme.INK_FAINT
        fraction = self._fraction(self.value)
        theme.meter(surface, track, fraction, color, track=theme.BG_SUNK)
        pygame.draw.rect(surface, theme.LINE, track, width=1, border_radius=track.height // 2)
        knob_r = max(11, int(self.rect.height * 0.42)) + int(press * 2)
        knob_x = int(self.rect.left + fraction * self.rect.width)
        knob_color = theme.INK if self.enabled else theme.INK_MUTED
        if self.enabled:
            theme.glow(surface, pygame.Rect(knob_x - knob_r, track.centery - knob_r, knob_r * 2, knob_r * 2),
                       color, spread=int(3 + 5 * hover), intensity=int(60 * hover))
        pygame.draw.circle(surface, knob_color, (knob_x, track.centery), knob_r)
        pygame.draw.circle(surface, color, (knob_x, track.centery), knob_r, width=2)
        if self.show_value:
            label = self._format(self.value)
            theme.text(surface, label, "mono", max(11, int(self.rect.height * 0.36)),
                       theme.INK if self.enabled else theme.INK_FAINT,
                       midright=(self.rect.right, track.top - 4))
        focus_ring(surface, self.rect.inflate(4, 4), focus, radius=track.height)


# --------------------------------------------------------------- text field


class TextField(Widget):
    """A single-line text box with a blinking caret.

    Typing uses `event.unicode` off `KEYDOWN` (so it works without enabling
    SDL text input), filtered by `allowed_chars` (a string/set of allowed
    characters; `None` allows any printable character) and `max_length`.
    Enter calls `on_commit(text)` and leaves the committed text in place;
    Escape reverts to the last committed value. Editing (before commit) fires
    `on_change(text)` on every keystroke, if given.
    """

    def __init__(self, value: str = "", *, max_length: Optional[int] = None,
                 allowed_chars: Optional[str] = None, uppercase: bool = False,
                 placeholder: str = "", on_commit: Optional[Callable[[str], None]] = None,
                 on_change: Optional[Callable[[str], None]] = None,
                 rect: Optional[pygame.Rect] = None) -> None:
        super().__init__(rect)
        self.max_length = max_length
        self.allowed_chars = set(allowed_chars) if allowed_chars is not None else None
        self.uppercase = uppercase
        self.placeholder = placeholder
        self.on_commit = on_commit
        self.on_change = on_change
        self._committed = value
        self.text = value
        self.caret = len(value)

    def set_value(self, value: str) -> None:
        value = value.upper() if self.uppercase else value
        self._committed = value
        self.text = value
        self.caret = len(value)

    def get_value(self) -> str:
        return self.text

    def _insert(self, ch: str) -> bool:
        if not ch or ch in ("\n", "\r", "\t"):
            return False
        if ord(ch) < 32:
            return False
        if self.uppercase:
            ch = ch.upper()
        if self.allowed_chars is not None and ch not in self.allowed_chars:
            return False
        if self.max_length is not None and len(self.text) >= self.max_length:
            return False
        self.text = self.text[:self.caret] + ch + self.text[self.caret:]
        self.caret += 1
        return True

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        if event.type != pygame.KEYDOWN or not self.focused:
            return False
        changed = False
        if event.key == pygame.K_BACKSPACE:
            if self.caret > 0:
                self.text = self.text[:self.caret - 1] + self.text[self.caret:]
                self.caret -= 1
                changed = True
        elif event.key == pygame.K_DELETE:
            if self.caret < len(self.text):
                self.text = self.text[:self.caret] + self.text[self.caret + 1:]
                changed = True
        elif event.key == pygame.K_LEFT:
            self.caret = max(0, self.caret - 1)
            return True
        elif event.key == pygame.K_RIGHT:
            self.caret = min(len(self.text), self.caret + 1)
            return True
        elif event.key == pygame.K_HOME:
            self.caret = 0
            return True
        elif event.key == pygame.K_END:
            self.caret = len(self.text)
            return True
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._committed = self.text
            if self.on_commit is not None:
                self.on_commit(self.text)
            return True
        elif event.key == pygame.K_ESCAPE:
            self.text = self._committed
            self.caret = len(self.text)
            return True
        else:
            changed = self._insert(getattr(event, "unicode", ""))
        if changed and self.on_change is not None:
            self.on_change(self.text)
        return changed or event.key in (pygame.K_BACKSPACE, pygame.K_DELETE)

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        _, focus, _ = self._update_state_anims(now)
        radius = max(4, self.rect.height // 5)
        bg = theme.BG_SUNK if self.enabled else theme.BG_RAISED
        border = theme.mix(theme.LINE, theme.GOLD, focus)
        theme.panel(surface, self.rect, bg, radius=radius, border=border, width=2)
        pad = max(8, self.rect.height // 4)
        size = max(11, int(self.rect.height * 0.58))
        shown = self.text if (self.text or self.focused) else self.placeholder
        color = theme.INK if (self.text or not self.enabled) else theme.INK_FAINT
        if not self.text and self.placeholder and not self.focused:
            color = theme.INK_FAINT
        clip = surface.get_clip()
        surface.set_clip(self.rect.inflate(-2, -2))
        theme.text(surface, shown, "mono", size, color if self.enabled else theme.INK_FAINT,
                   midleft=(self.rect.left + pad, self.rect.centery))
        if self.focused and self.enabled and theme.pulse(now, speed=3.2) > 0.5:
            caret_x = self.rect.left + pad + theme.font("mono", size).size(self.text[:self.caret])[0]
            pygame.draw.line(surface, theme.GOLD, (caret_x, self.rect.top + 6),
                             (caret_x, self.rect.bottom - 6), 2)
        surface.set_clip(clip)
        focus_ring(surface, self.rect.inflate(4, 4), focus, radius=radius + 2)


# ------------------------------------------------------------- number field


class NumberField(TextField):
    """A `TextField` that parses to a number.

    `invalid` is true whenever the current buffer doesn't parse or fails
    `validator`; the field draws a red border in that state but keeps the
    text so the player can keep fixing it. `on_commit(number)` (note: a
    number, not text) only fires when Enter is pressed *and* the buffer is
    valid; an invalid Enter just flags `invalid` and keeps editing.
    """

    def __init__(self, value: float, *, parser: Callable[[str], float] = float,
                 formatter: Callable[[float], str] = lambda v: f"{v:g}",
                 validator: Optional[Callable[[float], bool]] = None,
                 on_commit: Optional[Callable[[float], None]] = None,
                 max_length: Optional[int] = None,
                 rect: Optional[pygame.Rect] = None) -> None:
        self._parser = parser
        self._formatter = formatter
        self._validator = validator
        self._on_number_commit = on_commit
        self.invalid = False
        super().__init__(formatter(value), max_length=max_length,
                          allowed_chars="0123456789.-", on_change=self._check,
                          rect=rect)
        self._number = value
        self._check(self.text)

    def _check(self, text: str) -> None:
        try:
            number = self._parser(text)
            ok = self._validator(number) if self._validator else True
        except (ValueError, TypeError):
            ok = False
        self.invalid = not ok

    def get_number(self) -> Optional[float]:
        try:
            number = self._parser(self.text)
        except (ValueError, TypeError):
            return None
        if self._validator and not self._validator(number):
            return None
        return number

    def set_number(self, value: float) -> None:
        self._number = value
        self.set_value(self._formatter(value))
        self.invalid = False

    def handle(self, event: pygame.event.Event) -> bool:
        if (self._active(event) and event.type == pygame.KEYDOWN and self.focused
                and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)):
            number = self.get_number()
            if number is None:
                self.invalid = True
                return True
            self._number = number
            self._committed = self._formatter(number)
            self.text = self._committed
            self.caret = len(self.text)
            self.invalid = False
            if self._on_number_commit is not None:
                self._on_number_commit(number)
            return True
        return super().handle(event)

    def draw(self, surface: pygame.Surface, now: float) -> None:
        super().draw(surface, now)
        if self.invalid and self.visible:
            pygame.draw.rect(surface, theme.RED, self.rect, width=2,
                             border_radius=max(4, self.rect.height // 5))


# ------------------------------------------------------------------- tabs


class Tabs(Widget):
    """A tab bar. Click a tab, or move focus to it and press Left/Right;
    digit keys 1-9 jump straight to a tab regardless of focus (kiosk
    shortcut). `on_change(index)` fires on any change."""

    def __init__(self, labels: Sequence[str], active: int = 0,
                 on_change: Optional[Callable[[int], None]] = None, *,
                 rect: Optional[pygame.Rect] = None) -> None:
        super().__init__(rect)
        self.labels = list(labels)
        self.active_index = active
        self.on_change = on_change

    def set_active(self, index: int, notify: bool = True) -> None:
        index = clampi(index, 0, max(0, len(self.labels) - 1))
        if index == self.active_index:
            return
        self.active_index = index
        if notify and self.on_change is not None:
            self.on_change(index)

    def _segments(self) -> List[pygame.Rect]:
        n = max(1, len(self.labels))
        w = self.rect.width / n
        return [pygame.Rect(int(self.rect.left + i * w), self.rect.top, int(w), self.rect.height)
                for i in range(n)]

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, seg in enumerate(self._segments()):
                if seg.collidepoint(event.pos):
                    self.set_active(i)
                    return True
            return False
        if event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_9:
                index = event.key - pygame.K_1
                if index < len(self.labels):
                    self.set_active(index)
                    return True
            if self.focused:
                if event.key == pygame.K_LEFT:
                    self.set_active((self.active_index - 1) % len(self.labels))
                    return True
                if event.key == pygame.K_RIGHT:
                    self.set_active((self.active_index + 1) % len(self.labels))
                    return True
        return False

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        _, focus, _ = self._update_state_anims(now)
        pygame.draw.line(surface, theme.LINE, self.rect.bottomleft, self.rect.bottomright, 2)
        for i, (seg, label) in enumerate(zip(self._segments(), self.labels)):
            active = i == self.active_index
            color = theme.GOLD if active else theme.INK_MUTED
            theme.text(surface, label.upper(), "bold", max(12, int(self.rect.height * 0.4)),
                       color, center=seg.center, max_width=seg.width - 10)
            if active:
                underline = pygame.Rect(seg.left + 10, self.rect.bottom - 3, seg.width - 20, 3)
                pygame.draw.rect(surface, theme.GOLD, underline, border_radius=2)
        focus_ring(surface, self.rect.inflate(2, 2), focus, radius=8)


# -------------------------------------------------------------- scroll list


class ScrollList(Widget):
    """A scrollable, selectable list of arbitrary items.

    `render_row(surface, rect, item, index, selected, hovered, now)` draws
    one row; the list handles selection, scrolling and keeping the selection
    visible. A single click selects (`on_select`); Enter or a double click
    activates (`on_activate`) - e.g. "open this entry for editing".
    """

    def __init__(self, items: Sequence[Any],
                 render_row: Callable[[pygame.Surface, pygame.Rect, Any, int, bool, bool, float], None],
                 *, row_height: int = 40, on_select: Optional[Callable[[int], None]] = None,
                 on_activate: Optional[Callable[[int], None]] = None,
                 rect: Optional[pygame.Rect] = None, scrollbar: bool = True) -> None:
        super().__init__(rect)
        self.items = list(items)
        self.render_row = render_row
        self.row_height = row_height
        self.on_select = on_select
        self.on_activate = on_activate
        self.scrollbar = scrollbar
        self.selected_index: Optional[int] = 0 if self.items else None
        self._scroll = 0  # index of the first visible row
        self._scroll_anim = Anim()
        self._hover_index: Optional[int] = None
        self._last_click_ms = -10_000
        self._last_click_index: Optional[int] = None
        self._drag_bar = False

    # -- data --------------------------------------------------------------

    def set_items(self, items: Sequence[Any], keep_selection: bool = True) -> None:
        self.items = list(items)
        if not self.items:
            self.selected_index = None
        elif not keep_selection or self.selected_index is None:
            self.selected_index = 0
        else:
            self.selected_index = clampi(self.selected_index, 0, len(self.items) - 1)
        self._scroll = clampi(self._scroll, 0, self._max_scroll())
        if self.selected_index is not None:
            self.scroll_to(self.selected_index)

    def select(self, index: Optional[int], notify: bool = True, scroll_into_view: bool = True) -> None:
        if index is not None:
            if not self.items:
                index = None
            else:
                index = clampi(index, 0, len(self.items) - 1)
        if index == self.selected_index:
            return
        self.selected_index = index
        if scroll_into_view and index is not None:
            self.scroll_to(index)
        if notify and self.on_select is not None:
            self.on_select(index)

    # -- scrolling -----------------------------------------------------

    def _visible_rows(self) -> int:
        return max(1, self.rect.height // self.row_height)

    def _max_scroll(self) -> int:
        return max(0, len(self.items) - self._visible_rows())

    def scroll_to(self, index: int) -> None:
        visible = self._visible_rows()
        if index < self._scroll:
            self._scroll = index
        elif index >= self._scroll + visible:
            self._scroll = index - visible + 1
        self._scroll = clampi(self._scroll, 0, self._max_scroll())

    def _row_rects(self) -> List[Tuple[int, pygame.Rect]]:
        out = []
        for row_i in range(self._visible_rows() + 1):
            idx = self._scroll + row_i
            if idx >= len(self.items):
                break
            r = pygame.Rect(self.rect.left, self.rect.top + row_i * self.row_height,
                            self.rect.width, self.row_height)
            out.append((idx, r))
        return out

    # -- events --------------------------------------------------------

    def handle(self, event: pygame.event.Event) -> bool:
        if not self._active(event):
            return False
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
            self._hover_index = None
            if self._hover:
                for idx, r in self._row_rects():
                    if r.collidepoint(event.pos):
                        self._hover_index = idx
                        break
            if self._drag_bar:
                self._drag_scroll(event.pos[1])
                return True
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            bar = self._scrollbar_rect()
            if bar is not None and bar.collidepoint(event.pos):
                self._drag_bar = True
                self._drag_scroll(event.pos[1])
                return True
            if self.rect.collidepoint(event.pos):
                for idx, r in self._row_rects():
                    if r.collidepoint(event.pos):
                        self.select(idx)
                        now_ms = pygame.time.get_ticks()
                        double = (now_ms - self._last_click_ms < 400 and self._last_click_index == idx)
                        self._last_click_ms, self._last_click_index = now_ms, idx
                        if double and self.on_activate is not None:
                            self.on_activate(idx)
                        return True
                return True
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._drag_bar:
                self._drag_bar = False
                return True
            return False
        if event.type == pygame.MOUSEWHEEL:
            pos = _mouse_pos(event)
            if self.rect.collidepoint(pos):
                self._scroll = clampi(self._scroll - event.y, 0, self._max_scroll())
                return True
            return False
        if event.type == pygame.KEYDOWN and self.focused and self.items:
            cur = self.selected_index if self.selected_index is not None else 0
            if event.key == pygame.K_UP:
                self.select(max(0, cur - 1))
                return True
            if event.key == pygame.K_DOWN:
                self.select(min(len(self.items) - 1, cur + 1))
                return True
            if event.key == pygame.K_PAGEUP:
                self.select(max(0, cur - self._visible_rows()))
                return True
            if event.key == pygame.K_PAGEDOWN:
                self.select(min(len(self.items) - 1, cur + self._visible_rows()))
                return True
            if event.key == pygame.K_HOME:
                self.select(0)
                return True
            if event.key == pygame.K_END:
                self.select(len(self.items) - 1)
                return True
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.selected_index is not None and self.on_activate is not None:
                    self.on_activate(self.selected_index)
                return True
        return False

    def _scrollbar_rect(self) -> Optional[pygame.Rect]:
        if not self.scrollbar or len(self.items) <= self._visible_rows():
            return None
        track = pygame.Rect(self.rect.right - 8, self.rect.top, 6, self.rect.height)
        visible_frac = self._visible_rows() / len(self.items)
        thumb_h = max(24, int(track.height * visible_frac))
        travel = track.height - thumb_h
        pos = 0 if self._max_scroll() == 0 else self._scroll / self._max_scroll()
        thumb_y = track.top + int(travel * pos)
        return pygame.Rect(track.left, thumb_y, track.width, thumb_h)

    def _drag_scroll(self, mouse_y: int) -> None:
        if self._max_scroll() == 0:
            return
        frac = clamp((mouse_y - self.rect.top) / max(1, self.rect.height), 0.0, 1.0)
        self._scroll = clampi(round(frac * self._max_scroll()), 0, self._max_scroll())

    # -- drawing ---------------------------------------------------------

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible:
            return
        _, focus, _ = self._update_state_anims(now)
        self._scroll_anim.set(float(self._scroll * self.row_height), now)
        offset = self._scroll_anim.update(now, duration=0.15)
        clip = surface.get_clip()
        surface.set_clip(self.rect)
        top = self.rect.top - int(offset)
        for i, item in enumerate(self.items):
            r = pygame.Rect(self.rect.left, top + i * self.row_height, self.rect.width, self.row_height)
            if r.bottom < self.rect.top or r.top > self.rect.bottom:
                continue
            self.render_row(surface, r, item, i, i == self.selected_index, i == self._hover_index, now)
        surface.set_clip(clip)
        bar = self._scrollbar_rect()
        if bar is not None:
            pygame.draw.rect(surface, theme.LINE, pygame.Rect(bar.left, self.rect.top, bar.width, self.rect.height),
                             border_radius=3)
            pygame.draw.rect(surface, theme.INK_MUTED, bar, border_radius=3)
        focus_ring(surface, self.rect.inflate(2, 2), focus, radius=10)


# -------------------------------------------------------------- dialog/toast


class ConfirmDialog(Widget):
    """A modal yes/no dialog: title, message, two buttons.

    Call `show()` to open it and `close()` (or confirm/cancel) to dismiss.
    While `open`, the container should route every event to the dialog only
    (it consumes everything, `handle` always returns True while open) so
    nothing behind it can be interacted with. Enter confirms, Escape cancels.
    `danger=True` styles the confirm button red (for delete/clear).
    """

    focusable = False

    def __init__(self, title: str, message: str, *, confirm_label: str = "Confirm",
                 cancel_label: str = "Cancel", on_confirm: Optional[Callable[[], None]] = None,
                 on_cancel: Optional[Callable[[], None]] = None, danger: bool = False) -> None:
        super().__init__()
        self.title = title
        self.message = message
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.open = False
        self._danger = danger
        self._confirm = Button(confirm_label, self._do_confirm,
                               variant="danger" if danger else "primary")
        self._cancel = Button(cancel_label, self._do_cancel, variant="secondary")
        self._appear = Anim()

    def show(self) -> None:
        self.open = True

    def close(self) -> None:
        self.open = False

    def _do_confirm(self) -> None:
        self.close()
        if self.on_confirm is not None:
            self.on_confirm()

    def _do_cancel(self) -> None:
        self.close()
        if self.on_cancel is not None:
            self.on_cancel()

    def layout(self, rect: pygame.Rect) -> None:
        """`rect` is the full screen area to dim; the panel centres in it.

        The panel's height grows to fit the (word-wrapped) message rather
        than a fixed fraction, so a long confirmation ("this deletes all 14
        entries...") never gets clipped or run into the buttons.
        """
        super().layout(rect)
        self._pad = theme.scaled(rect.height, 0.022)
        self._title_size = theme.scaled(rect.height, 0.04)
        self._msg_size = theme.scaled(rect.height, 0.022)
        self._line_h = int(self._msg_size * 1.4)
        btn_h = theme.scaled(rect.height, 0.05)
        w = min(rect.width - 80, max(theme.scaled(rect.height, 0.5), 360))
        inner_w = w - self._pad * 2
        self._lines = wrap_text(self.message, "body", self._msg_size, inner_w)

        title_block = int(self._title_size * 1.3)
        gap_after_title = theme.scaled(rect.height, 0.018)
        gap_before_buttons = theme.scaled(rect.height, 0.025)
        content_h = (self._pad + title_block + gap_after_title
                     + len(self._lines) * self._line_h + gap_before_buttons + btn_h + self._pad)
        panel_h = max(theme.scaled(rect.height, 0.16), content_h)

        self.panel_rect = pygame.Rect(0, 0, w, panel_h)
        self.panel_rect.center = rect.center
        self._title_y = self.panel_rect.top + self._pad
        self._message_y = self._title_y + title_block + gap_after_title

        # Each button is as wide as its label needs (never narrower than the
        # default), so "YES, CLEAR ALL" is not squeezed to fit a fixed width.
        gap = theme.scaled(rect.height, 0.015)
        label_size = max(12, int(btn_h * 0.42))

        def button_width(label: str) -> int:
            wanted = theme.font("bold", label_size).size(label)[0] + btn_h
            return min(inner_w // 2 - gap, max(theme.scaled(rect.height, 0.14), wanted))

        by = self.panel_rect.bottom - btn_h - self._pad
        cancel_w, confirm_w = button_width(self._cancel.label), button_width(self._confirm.label)
        self._cancel.layout(pygame.Rect(self.panel_rect.right - self._pad - cancel_w, by,
                                        cancel_w, btn_h))
        self._confirm.layout(pygame.Rect(self._cancel.rect.left - gap - confirm_w, by,
                                         confirm_w, btn_h))

    def handle(self, event: pygame.event.Event) -> bool:
        if not self.open:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._do_cancel()
                return True
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._do_confirm()
                return True
        self._confirm.handle(event)
        self._cancel.handle(event)
        return True

    def draw(self, surface: pygame.Surface, now: float) -> None:
        self._appear.set(1.0 if self.open else 0.0, now)
        amount = self._appear.update(now, duration=0.15, ease=theme.ease_out_back)
        if amount <= 0.001:
            return
        theme.dim(surface, self.rect, int(160 * theme.clamp01(amount)))
        panel = self.panel_rect.inflate(0, 0)
        scale = theme.clamp01(0.85 + 0.15 * theme.clamp01(amount))
        drawn = pygame.Rect(0, 0, int(panel.width * scale), int(panel.height * scale))
        drawn.center = panel.center
        accent = theme.RED if self._danger else theme.GOLD
        # Halo first, as an outline, so the panel body stays a clean dark.
        theme.glow(surface, drawn, accent, spread=14, radius=18, intensity=60, width=2)
        theme.panel(surface, drawn, theme.BG_RAISED, radius=18, border=accent, width=2)
        theme.text(surface, self.title, "display", self._title_size, theme.GOLD,
                   topleft=(drawn.left + self._pad, self._title_y))
        for i, line in enumerate(self._lines):
            theme.text(surface, line, "body", self._msg_size, theme.INK_MUTED,
                       topleft=(drawn.left + self._pad, self._message_y + i * self._line_h))
        self._confirm.draw(surface, now)
        self._cancel.draw(surface, now)


class Toast:
    """A transient message that fades in, holds, then fades out.

    Not a `Widget` (nothing to focus or click). Call `show(message)` whenever
    there's something to announce, and `draw(surface, now)` every frame; it
    no-ops when there is nothing to show.
    """

    def __init__(self) -> None:
        self.message: Optional[str] = None
        self.kind = "info"
        self._shown_at = 0.0
        self.duration = 2.0
        self._fade = 0.35

    def show(self, message: str, *, kind: str = "info", duration: float = 2.0, now: float = 0.0) -> None:
        self.message = message
        self.kind = kind
        self.duration = duration
        self._shown_at = now

    def visible(self, now: float) -> bool:
        return self.message is not None and now < self._shown_at + self.duration + self._fade

    def draw(self, surface: pygame.Surface, now: float) -> None:
        if not self.visible(now):
            return
        age = now - self._shown_at
        if age < self._fade:
            alpha = theme.ease_out_cubic(age / self._fade)
        elif age < self.duration:
            alpha = 1.0
        else:
            alpha = 1.0 - theme.clamp01((age - self.duration) / self._fade)
        color = {"info": theme.CYAN, "success": theme.GREEN, "error": theme.RED}.get(self.kind, theme.CYAN)
        size = surface.get_size()
        pad_x, pad_y = 20, 12
        text_w = theme.font("bold", theme.scaled(size[1], 0.026)).size(self.message)[0]
        w, h = text_w + pad_x * 2, theme.scaled(size[1], 0.026) + pad_y * 2
        rect = pygame.Rect(0, 0, w, h)
        rect.midbottom = (size[0] // 2, size[1] - theme.scaled(size[1], 0.06))
        theme.panel(surface, rect, theme.BG_RAISED, radius=rect.height // 2, border=color, width=2,
                    alpha=int(235 * alpha))
        theme.text(surface, self.message, "bold", theme.scaled(size[1], 0.026), color,
                   center=rect.center, alpha=int(255 * alpha))


# -------------------------------------------------------------- focus manager


class FocusManager:
    """Tracks which widget currently has keyboard focus among a flat list.

    Call `handle(event)` for every event *before* offering it to the widgets
    themselves: it moves focus on Tab/Shift+Tab (and, if `arrow_nav=True`,
    Up/Down too), and on a left click it focuses whichever focusable widget
    the click landed in (without consuming the click, so the widget still
    gets to act on it). It keeps every widget's `.focused` flag in sync.
    """

    def __init__(self, widgets: Sequence[Widget] = (), *, arrow_nav: bool = False) -> None:
        self._widgets: List[Widget] = []
        self._index: Optional[int] = None
        self.arrow_nav = arrow_nav
        self.set_widgets(widgets)

    def set_widgets(self, widgets: Sequence[Widget]) -> None:
        self._widgets = list(widgets)
        focusable = self._focusable()
        if not focusable:
            self._index = None
        elif self._index is None or self._index >= len(focusable):
            self._index = 0
        self._sync()

    def _focusable(self) -> List[Widget]:
        return [w for w in self._widgets if w.focusable and w.visible and w.enabled]

    def _sync(self) -> None:
        focusable = self._focusable()
        current = focusable[self._index] if (self._index is not None and focusable) else None
        for w in self._widgets:
            w.focused = (w is current)

    def focused_widget(self) -> Optional[Widget]:
        focusable = self._focusable()
        return focusable[self._index] if (self._index is not None and focusable) else None

    def focus(self, widget: Optional[Widget]) -> None:
        focusable = self._focusable()
        self._index = focusable.index(widget) if widget in focusable else None
        self._sync()

    def focus_index(self, index: Optional[int]) -> None:
        focusable = self._focusable()
        self._index = clampi(index, 0, len(focusable) - 1) if (index is not None and focusable) else None
        self._sync()

    def _move(self, step: int) -> None:
        focusable = self._focusable()
        if not focusable:
            self._index = None
            return
        if self._index is None:
            self._index = 0 if step > 0 else len(focusable) - 1
        else:
            self._index = (self._index + step) % len(focusable)
        self._sync()

    def handle(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self._move(-1 if (event.mod & pygame.KMOD_SHIFT) else 1)
                return True
            if self.arrow_nav:
                if event.key == pygame.K_DOWN:
                    self._move(1)
                    return True
                if event.key == pygame.K_UP:
                    self._move(-1)
                    return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for w in self._focusable():
                if w.rect.collidepoint(event.pos):
                    self.focus(w)
                    break
        return False
