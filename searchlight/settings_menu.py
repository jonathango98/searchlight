"""The in-game settings overlay: tabs by group, rows generated from
`searchlight.settings.SETTINGS`, presets, and a save/cancel footer.

`SettingsMenu` owns a *working copy* of `Settings` (`with_changes` at every
edit, so clamping and cross-field consistency - e.g. `min_side` never above
`start_side` - always apply) and only ever hands the caller a finished,
already-valid `Settings` via `on_save`. Nothing is written to the store from
here; the caller (the game loop) persists it.

Layout: everything is derived from the surface's height via `theme.scaled`,
so the menu fits without clipping from 1280x720 up to 4K. The Detection tab
has nine rows, more than fit in a short window, so the row list scrolls like
`widgets.ScrollList` (same keep-selection-visible idea), except each row is a
real interactive control rather than a rendered line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import pygame

from .settings import (
    DEFAULTS,
    GROUPS,
    LIVE,
    NEXT_RUN,
    BOOL,
    CHOICE,
    Settings,
    SETTINGS_BY_KEY,
    apply_preset,
    matching_preset,
    settings_in,
)
from .ui import theme
from .ui import widgets as w

PRESET_NAMES: Tuple[str, ...] = ("Easy", "Normal", "Hard")
PRESET_TABS = {"Gameplay", "Square"}


def _pointer(event: pygame.event.Event) -> Tuple[int, int]:
    pos = getattr(event, "pos", None)
    return pos if pos is not None else pygame.mouse.get_pos()


@dataclass
class _Row:
    spec: Any
    control: w.Widget
    reset_btn: w.Button
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    value_rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    label_pos: Tuple[int, int] = (0, 0)
    dot_pos: Tuple[int, int] = (0, 0)


class SettingsMenu:
    """A full-screen modal overlay. Construct it, then feed it events/frames:

        menu = SettingsMenu(store.load_settings(), on_save=..., on_close=...)
        ...
        menu.handle(event)   # every event, while the menu is open
        menu.update(now)     # once a frame, before draw
        menu.draw(surface)   # once a frame

    `on_save(new_settings)` is called (then `on_close()`) when Save is
    pressed with unsaved changes. Cancel/Esc calls `on_close()` directly, or
    - if there are unsaved changes - after a "discard changes?" confirm.
    The caller is expected to stop calling into the menu once `on_close`
    fires (and to persist `new_settings`, e.g. via `store.save_settings`).
    """

    def __init__(self, settings: Settings, on_save: Callable[[Settings], None],
                 on_close: Callable[[], None]) -> None:
        self._saved = settings
        self._working = settings
        self.on_save = on_save
        self.on_close = on_close

        self._tabs = w.Tabs(list(GROUPS), active=0, on_change=lambda _i: self._on_tab_change())
        self._preset_buttons = [w.Button(name, on_click=self._preset_click(name))
                                for name in PRESET_NAMES]
        self._save_btn = w.Button("Save", on_click=self._do_save, variant="primary")
        self._cancel_btn = w.Button("Cancel", on_click=self._request_cancel, variant="secondary")
        self._reset_all_btn = w.Button("Reset all to defaults", on_click=self._confirm_reset_all,
                                       variant="danger")

        self._discard_dialog = w.ConfirmDialog(
            "Discard changes?", "Your changes have not been saved.",
            confirm_label="Discard", cancel_label="Keep editing",
            on_confirm=self.on_close)
        self._reset_dialog = w.ConfirmDialog(
            "Reset all settings?", "Every value on every tab goes back to its default. "
            "This does not affect the leaderboard.",
            confirm_label="Reset all", danger=True, on_confirm=self._do_reset_all)

        self._rows: List[_Row] = []
        self._scroll = 0
        self._row_h = 40
        self._viewport = pygame.Rect(0, 0, 0, 0)
        self._H = 720
        self._changed: set = set()
        self._current_preset: Optional[str] = None
        self._now = 0.0

        self._focus = w.FocusManager([], arrow_nav=True)
        self._build_rows_for_tab()

    # ---------------------------------------------------------- tab/rows

    def _active_group(self) -> str:
        return GROUPS[self._tabs.active_index]

    def _preset_click(self, name: str) -> Callable[[], None]:
        return lambda: self._apply_preset(name)

    def _on_tab_change(self) -> None:
        self._build_rows_for_tab()

    def _make_control(self, spec: Any, value: Any) -> w.Widget:
        on_change = lambda v, key=spec.key: self._on_row_change(key, v)
        if spec.kind == BOOL:
            return w.Toggle(bool(value), on_change=on_change)
        if spec.kind == CHOICE:
            return w.Choice(spec.choices, value, on_change=on_change)
        # The row already draws the formatted value in its own column, so the
        # slider doesn't need to draw its own (that was showing it twice).
        return w.Slider(value, spec=spec, on_change=on_change, show_value=False)

    def _build_rows_for_tab(self) -> None:
        specs = settings_in(self._active_group())
        rows = []
        for spec in specs:
            control = self._make_control(spec, self._working.get(spec.key))
            reset_btn = w.Button("RESET", variant="secondary")
            reset_btn.focusable = False
            rows.append(_Row(spec, control, reset_btn))
        self._rows = rows
        self._scroll = 0
        self._layout_rows()
        self._rebuild_focus_list()

    def _rebuild_focus_list(self) -> None:
        widgets: List[w.Widget] = []
        if self._active_group() in PRESET_TABS:
            widgets.extend(self._preset_buttons)
        widgets.extend(row.control for row in self._rows)
        widgets.append(self._reset_all_btn)
        widgets.append(self._cancel_btn)
        widgets.append(self._save_btn)
        self._focus.set_widgets(widgets)

    def _sync_controls(self) -> None:
        for row in self._rows:
            row.control.set_value(self._working.get(row.spec.key), notify=False)

    def _on_row_change(self, key: str, value: Any) -> None:
        self._working = self._working.with_changes(**{key: value})
        self._sync_controls()

    def _reset_row(self, key: str) -> None:
        default = SETTINGS_BY_KEY[key].default
        self._working = self._working.with_changes(**{key: default})
        self._sync_controls()

    def _row_at_default(self, row: _Row) -> bool:
        return self._working.get(row.spec.key) == row.spec.default

    def _reset_focused_row(self) -> bool:
        widget = self._focus.focused_widget()
        for row in self._rows:
            if row.control is widget:
                self._reset_row(row.spec.key)
                return True
        return False

    def _focused_spec(self) -> Optional[Any]:
        widget = self._focus.focused_widget()
        for row in self._rows:
            if row.control is widget:
                return row.spec
        return None

    def _apply_preset(self, name: str) -> None:
        self._working = apply_preset(self._working, name)
        self._sync_controls()

    # ----------------------------------------------------------------- actions

    def _do_save(self) -> None:
        self.on_save(self._working)
        self.on_close()

    def _request_cancel(self) -> None:
        if self._working.changed_keys(self._saved):
            self._discard_dialog.show()
        else:
            self.on_close()

    def _confirm_reset_all(self) -> None:
        self._reset_dialog.show()

    def _do_reset_all(self) -> None:
        self._working = Settings()
        self._sync_controls()

    # ------------------------------------------------------------------ scroll

    def _capacity(self) -> int:
        return max(1, self._viewport.height // self._row_h) if self._row_h else 1

    def _max_scroll(self) -> int:
        return max(0, len(self._rows) - self._capacity())

    def _ensure_focus_visible(self) -> None:
        if not self._rows or not self._row_h:
            return
        widget = self._focus.focused_widget()
        for i, row in enumerate(self._rows):
            if row.control is widget:
                capacity = self._capacity()
                if i < self._scroll:
                    self._scroll = i
                elif i >= self._scroll + capacity:
                    self._scroll = i - capacity + 1
                self._scroll = max(0, min(self._scroll, self._max_scroll()))
                self._layout_rows()
                return

    def _handle_wheel(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.MOUSEWHEEL:
            return False
        if not self._viewport.collidepoint(_pointer(event)):
            return False
        self._scroll = max(0, min(self._scroll - event.y, self._max_scroll()))
        self._layout_rows()
        return True

    # ------------------------------------------------------------------ events

    def handle(self, event: pygame.event.Event) -> bool:
        for dialog in (self._discard_dialog, self._reset_dialog):
            if dialog.open:
                dialog.handle(event)
                return True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._request_cancel()
            return True

        if self._focus.handle(event):
            self._ensure_focus_visible()
            return True

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            if self._reset_focused_row():
                return True

        if self._tabs.handle(event):
            return True

        if self._active_group() in PRESET_TABS:
            for btn in self._preset_buttons:
                if btn.handle(event):
                    return True

        if self._handle_wheel(event):
            return True

        is_mouse = event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                                  pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL)
        for row in self._rows:
            if is_mouse and not self._viewport.colliderect(row.rect):
                continue
            if not self._row_at_default(row) and row.reset_btn.handle(event):
                self._reset_row(row.spec.key)
                return True
            if row.control.handle(event):
                return True

        if self._reset_all_btn.handle(event):
            return True
        if self._cancel_btn.handle(event):
            return True
        if self._save_btn.handle(event):
            return True
        return True

    # ------------------------------------------------------------------ frame

    def update(self, now: float) -> None:
        self._now = now
        self._changed = self._working.changed_keys(self._saved)
        self._save_btn.enabled = bool(self._changed)
        self._current_preset = matching_preset(self._working)
        for btn, name in zip(self._preset_buttons, PRESET_NAMES):
            btn.variant = "primary" if name == self._current_preset else "secondary"

    # ------------------------------------------------------------------ layout

    def _layout(self, rect: pygame.Rect) -> None:
        H = rect.height
        self._H = H
        margin = theme.scaled(H, 0.035)
        panel = rect.inflate(-margin * 2, -margin * 2)
        self._panel = panel
        pad = theme.scaled(H, 0.022)
        inner = panel.inflate(-pad * 2, -pad * 2)

        title_h = theme.scaled(H, 0.05)
        self._title_rect = pygame.Rect(inner.left, inner.top, inner.width, title_h)

        tabs_h = theme.scaled(H, 0.05)
        tabs_rect = pygame.Rect(inner.left, self._title_rect.bottom + theme.scaled(H, 0.008),
                                inner.width, tabs_h)
        self._tabs.layout(tabs_rect)

        y = tabs_rect.bottom + theme.scaled(H, 0.015)

        if self._active_group() in PRESET_TABS:
            preset_h = theme.scaled(H, 0.045)
            btn_w = theme.scaled(H, 0.12)
            gap = theme.scaled(H, 0.012)
            label_w = theme.scaled(H, 0.09)
            for i, btn in enumerate(self._preset_buttons):
                btn.layout(pygame.Rect(inner.left + label_w + i * (btn_w + gap), y, btn_w, preset_h))
            self._preset_rect = pygame.Rect(inner.left, y, inner.width, preset_h)
            self._preset_label_pos = (inner.left, y + preset_h // 2)
            y += preset_h + theme.scaled(H, 0.02)
        else:
            self._preset_rect = None

        help_h = theme.scaled(H, 0.085)
        footer_h = theme.scaled(H, 0.06)
        footer_gap = theme.scaled(H, 0.015)
        viewport_h = max(theme.scaled(H, 0.12),
                         inner.bottom - footer_h - footer_gap - help_h - footer_gap - y)
        self._viewport = pygame.Rect(inner.left, y, inner.width, viewport_h)

        self._row_h = theme.scaled(H, 0.062)
        self._scroll = max(0, min(self._scroll, self._max_scroll()))
        self._layout_rows()

        self._help_rect = pygame.Rect(inner.left, self._viewport.bottom + footer_gap,
                                      inner.width, help_h)

        footer_rect = pygame.Rect(inner.left, inner.bottom - footer_h, inner.width, footer_h)
        self._footer_rect = footer_rect
        save_w = theme.scaled(H, 0.15)
        cancel_w = theme.scaled(H, 0.12)
        reset_w = theme.scaled(H, 0.24)
        gap = theme.scaled(H, 0.012)
        self._save_btn.layout(pygame.Rect(footer_rect.right - save_w, footer_rect.top, save_w, footer_h))
        self._cancel_btn.layout(pygame.Rect(self._save_btn.rect.left - gap - cancel_w, footer_rect.top,
                                            cancel_w, footer_h))
        self._reset_all_btn.layout(pygame.Rect(footer_rect.left, footer_rect.top, reset_w, footer_h))

        self._discard_dialog.layout(rect)
        self._reset_dialog.layout(rect)

    def _layout_rows(self) -> None:
        vp = self._viewport
        row_h = self._row_h
        if row_h <= 0:
            return
        gutter = theme.scaled(self._H, 0.014)
        reset_w = theme.scaled(self._H, 0.075)
        value_w = max(60, int(vp.width * 0.14))
        label_w = max(80, int(vp.width * 0.27))
        col_gap = theme.scaled(self._H, 0.012)
        control_left = vp.left + gutter + label_w + col_gap
        control_w = max(20, vp.width - gutter - label_w - value_w - reset_w - col_gap * 3)
        for i, row in enumerate(self._rows):
            top = vp.top + (i - self._scroll) * row_h
            row.rect = pygame.Rect(vp.left, top, vp.width, row_h)
            if isinstance(row.control, w.Toggle):
                # A switch reads as a toggle at its natural size; stretched
                # across the whole control column it just looks like a bar.
                th = max(14, int(row_h * 0.5))
                tw = int(th * 1.9)
                row.control.layout(pygame.Rect(control_left, top + (row_h - th) // 2, tw, th))
            else:
                control_rect = pygame.Rect(control_left, top + int(row_h * 0.19),
                                           control_w, max(10, int(row_h * 0.62)))
                row.control.layout(control_rect)
            row.reset_btn.layout(pygame.Rect(vp.right - reset_w, top + int(row_h * 0.18),
                                             reset_w, max(10, int(row_h * 0.64))))
            row.label_pos = (vp.left + gutter, top + row_h // 2)
            row.value_rect = pygame.Rect(control_left + control_w + col_gap, top, value_w, row_h)
            row.dot_pos = (vp.left + max(4, gutter // 2), top + row_h // 2)

    # ------------------------------------------------------------------- draw

    def draw(self, surface: pygame.Surface) -> None:
        rect = surface.get_rect()
        self._layout(rect)
        now = self._now

        theme.dim(surface, rect, 190)
        theme.panel(surface, self._panel, theme.mix(theme.BG_RAISED, theme.BG, 0.25), radius=22,
                   border=theme.LINE, width=2, alpha=245)
        theme.text(surface, "SETTINGS", "display", theme.scaled(self._H, 0.045), theme.GOLD,
                  topleft=self._title_rect.topleft)

        self._tabs.draw(surface, now)

        if self._preset_rect is not None:
            theme.text(surface, "PRESET", "mono", theme.scaled(self._H, 0.02), theme.INK_FAINT,
                      midleft=self._preset_label_pos)
            for btn in self._preset_buttons:
                btn.draw(surface, now)
            if self._current_preset is None:
                last = self._preset_buttons[-1].rect
                theme.text(surface, "CUSTOM", "mono", theme.scaled(self._H, 0.02), theme.INK_MUTED,
                          midleft=(last.right + theme.scaled(self._H, 0.015), last.centery))

        self._draw_rows(surface, now)
        self._draw_help(surface)

        self._reset_all_btn.draw(surface, now)
        self._cancel_btn.draw(surface, now)
        self._save_btn.draw(surface, now)

        self._discard_dialog.draw(surface, now)
        self._reset_dialog.draw(surface, now)

    def _draw_rows(self, surface: pygame.Surface, now: float) -> None:
        vp = self._viewport
        clip = surface.get_clip()
        surface.set_clip(vp)
        label_size = theme.scaled(self._H, 0.021)
        value_size = theme.scaled(self._H, 0.022)
        for row in self._rows:
            if row.rect.bottom < vp.top or row.rect.top > vp.bottom:
                continue
            if row.control.focused:
                theme.panel(surface, row.rect.inflate(-4, -2), theme.BG_HOVER, radius=8, alpha=130)
            if row.spec.key in self._changed:
                pygame.draw.circle(surface, theme.GOLD, row.dot_pos, max(3, self._row_h // 12))
            theme.text(surface, row.spec.label, "body", label_size,
                      theme.INK if row.control.enabled else theme.INK_FAINT,
                      midleft=row.label_pos, max_width=row.value_rect.left - row.label_pos[0] - 10)
            row.control.draw(surface, now)
            value_text = row.spec.format(self._working.get(row.spec.key))
            theme.text(surface, value_text, "mono", value_size, theme.GOLD, midright=(
                row.value_rect.right - theme.scaled(self._H, 0.01), row.value_rect.centery))
            if not self._row_at_default(row):
                row.reset_btn.draw(surface, now)
        surface.set_clip(clip)

        if len(self._rows) > self._capacity():
            track = pygame.Rect(vp.right - 6, vp.top, 4, vp.height)
            pygame.draw.rect(surface, theme.LINE, track, border_radius=2)
            visible_frac = self._capacity() / len(self._rows)
            thumb_h = max(18, int(track.height * visible_frac))
            travel = track.height - thumb_h
            pos = 0 if self._max_scroll() == 0 else self._scroll / self._max_scroll()
            thumb = pygame.Rect(track.left, track.top + int(travel * pos), track.width, thumb_h)
            pygame.draw.rect(surface, theme.INK_MUTED, thumb, border_radius=2)

    def _draw_help(self, surface: pygame.Surface) -> None:
        rect = self._help_rect
        pygame.draw.line(surface, theme.LINE, rect.topleft, rect.topright, 1)
        spec = self._focused_spec()
        if spec is None:
            theme.text(surface, "Use UP/DOWN to browse, LEFT/RIGHT to change a value.",
                      "body", theme.scaled(self._H, 0.02), theme.INK_FAINT,
                      topleft=(rect.left, rect.top + theme.scaled(self._H, 0.012)))
            return
        applies_now = spec.applies == LIVE
        tag = "APPLIES NOW" if applies_now else "APPLIES NEXT RUN"
        tag_color = theme.CYAN if applies_now else theme.INK_MUTED
        theme.text(surface, spec.label.upper(), "bold", theme.scaled(self._H, 0.022), theme.GOLD,
                  topleft=(rect.left, rect.top + theme.scaled(self._H, 0.01)))
        tag_size = theme.scaled(self._H, 0.016)
        tag_w = theme.font("mono", tag_size).size(tag)[0] + theme.scaled(self._H, 0.02)
        tag_rect = pygame.Rect(0, 0, tag_w, theme.scaled(self._H, 0.028))
        tag_rect.topright = (rect.right, rect.top + theme.scaled(self._H, 0.006))
        theme.panel(surface, tag_rect, theme.BG_SUNK, radius=tag_rect.height // 2,
                   border=tag_color, width=1)
        theme.text(surface, tag, "mono", tag_size, tag_color, center=tag_rect.center)
        lines = w.wrap_text(spec.help, "body", theme.scaled(self._H, 0.019), rect.width)
        line_h = int(theme.scaled(self._H, 0.019) * 1.35)
        top = rect.top + theme.scaled(self._H, 0.04)
        for i, line in enumerate(lines[:2]):
            theme.text(surface, line, "body", theme.scaled(self._H, 0.019), theme.INK_MUTED,
                      topleft=(rect.left, top + i * line_h))
