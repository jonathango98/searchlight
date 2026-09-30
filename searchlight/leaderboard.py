"""The audience window: Searchlight's standings, plus whoever is in the light now.

Two modes, one process:

* **display** - what the crowd sees. The top rows (from ``settings.board_rows``),
  a NOW PLAYING panel while someone's mid-run, and a flash on whatever row the
  game just announced.
* **edit** - press **E** to switch. The operator can add, rename, re-score,
  delete or clear entries with keyboard shortcuts or the mouse, entirely from
  this window - no separate admin tool.

Both modes poll ``store.Store`` (revisions + live state) rather than owning any
state of their own, since the game process writes the same database.
"""

from __future__ import annotations

# pygame first - see the SDL2 note in searchlight/game.py.
import pygame

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import store as store_mod
from .store import (Entry, InvalidName, Live, MAX_NAME, Store, format_score,
                     normalise_name, parse_score)
from .ui import theme
from .ui.widgets import (Anim, Button, ConfirmDialog, FocusManager, NumberField,
                          ScrollList, TextField, Toast, split_cols, stack_rows)
from .ui.window import Window, is_quit_key

# ----------------------------------------------------------------- constants

POLL_INTERVAL = 0.1        # seconds between store polls (revisions are cheap, but not free)
HIGHLIGHT_SECONDS = 10.0   # how long a freshly announced row keeps glowing
LIVE_STALE_SECONDS = 3.0   # older than this and the game is treated as idle
RESULT_TAG_SECONDS = 6.0   # how long "JUST FINISHED" shows after a result
FADE_IN_SECONDS = 0.6

ACTIVE_PHASES = (store_mod.PHASE_FIND_PLAYER, store_mod.PHASE_COUNTDOWN,
                  store_mod.PHASE_PLAYING)


def _score_formatter(score: int) -> str:
    return f"{score / 10:.1f}"


def _score_validator(score: float) -> bool:
    return 0 <= score <= store_mod.MAX_SCORE


# --------------------------------------------------------------------- app


class LeaderboardApp:
    """All the leaderboard window's state and logic, independent of pygame's
    event loop - so it can be driven directly by tests with synthetic events.
    """

    def __init__(self, store: Store, size: tuple = (1280, 720)) -> None:
        self.store = store
        self.mode = "display"
        self.quit = False
        self._size = size

        # ---- polled data
        self._board_rev: Optional[int] = None
        self._settings_rev: Optional[int] = None
        self._next_poll = 0.0
        self._now_cache = 0.0
        self._board_rows = 10
        self._top: List[Entry] = []
        self._all: List[Entry] = []
        self._total = 0
        self._live = Live()
        self._live_stale = True

        # ---- highlight / result animation
        self._highlight_id: Optional[int] = None
        self._highlight_seq = 0
        self._highlight_started = -1e9
        self._result_shown_at = -1e9
        self._result_score = 0
        self._result_name = ""
        self._last_phase = store_mod.PHASE_IDLE

        # ---- display-mode motion
        self._started_at: Optional[float] = None
        self._row_anims: Dict[int, Anim] = {}
        self._panel_anim = Anim(0.0)

        # ---- edit mode state
        self._editing: Optional[str] = None      # None | "name" | "time" | "add"
        self._editing_id: Optional[int] = None
        self._pending_rename: Optional[tuple] = None  # (entry_id, clean_name, other_entry)

        self.toast = Toast()

        self._edit_list = ScrollList([], self._render_edit_row, row_height=60,
                                      on_select=None, on_activate=self._activate_row)

        self._rename_field = TextField(uppercase=True, max_length=MAX_NAME,
                                        on_commit=self._commit_rename)
        self._time_field = NumberField(0.0, parser=self._parse_seconds,
                                        formatter=_score_formatter,
                                        validator=_score_validator,
                                        on_commit=self._commit_time)

        self._add_name = TextField(uppercase=True, max_length=MAX_NAME,
                                    placeholder="NAME")
        self._add_time = NumberField(0.0, parser=self._parse_seconds,
                                      formatter=_score_formatter,
                                      validator=_score_validator,
                                      on_commit=lambda _n: self._commit_add())

        self._btn_rename = Button("EDIT NAME (R)", self._start_rename)
        self._btn_time = Button("EDIT TIME (T)", self._start_time)
        self._btn_add = Button("ADD (N)", self._start_add, variant="primary")
        self._btn_delete = Button("DELETE (DEL)", self._ask_delete, variant="danger")
        self._btn_clear = Button("CLEAR ALL", self._ask_clear, variant="danger")
        self._btn_exit = Button("EXIT (E)", self._exit_edit)
        self._btn_add_confirm = Button("ADD", self._commit_add, variant="primary")
        self._btn_add_cancel = Button("CANCEL", self._cancel_add)

        self._confirm_delete = ConfirmDialog(
            "DELETE ENTRY", "", confirm_label="DELETE", cancel_label="CANCEL",
            on_confirm=self._do_delete, danger=True)
        self._confirm_clear1 = ConfirmDialog(
            "CLEAR ALL ENTRIES", "", confirm_label="CLEAR ALL", cancel_label="CANCEL",
            on_confirm=self._ask_clear_again, danger=True)
        self._confirm_clear2 = ConfirmDialog(
            "ARE YOU SURE?", "This cannot be undone.", confirm_label="YES, CLEAR ALL",
            cancel_label="CANCEL", on_confirm=self._do_clear, danger=True)
        self._confirm_merge = ConfirmDialog(
            "MERGE ENTRIES?", "", confirm_label="MERGE", cancel_label="CANCEL",
            on_confirm=self._confirm_merge_rename, on_cancel=self._cancel_merge)

        # The list stays out of the focus rotation so Tab only cycles the
        # buttons, while the arrow keys *always* move the selection (wired by
        # hand in `_handle_edit`) - the operator never has to focus the list
        # first. Row selection is shown with its own cyan border.
        self._edit_list.focusable = False
        self._focus = FocusManager(
            [self._btn_rename, self._btn_time, self._btn_add,
             self._btn_delete, self._btn_clear, self._btn_exit],
            arrow_nav=False)

        self._layout()

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _parse_seconds(text: str) -> int:
        return parse_score(text)

    def _selected_entry(self) -> Optional[Entry]:
        idx = self._edit_list.selected_index
        if idx is None or not (0 <= idx < len(self._all)):
            return None
        return self._all[idx]

    def _index_of_id(self, entry_id: int) -> Optional[int]:
        for i, e in enumerate(self._all):
            if e.id == entry_id:
                return i
        return None

    # ---------------------------------------------------------------- poll

    def poll(self, now: float) -> None:
        """Re-read the store roughly every `POLL_INTERVAL` seconds."""
        self._now_cache = now
        if now < self._next_poll:
            return
        self._next_poll = now + POLL_INTERVAL
        if self._started_at is None:
            self._started_at = now

        board_rev, settings_rev = self.store.revisions()
        need_refresh = board_rev != self._board_rev or self._board_rev is None
        if settings_rev != self._settings_rev:
            self._settings_rev = settings_rev
            self._board_rows = self.store.load_settings().board_rows
            need_refresh = True  # board_rows changed - re-slice top() even if board_rev didn't

        self._board_rev = board_rev
        if need_refresh:
            self._refresh_entries(now)

        live = self.store.get_live()
        if live.highlight_seq != self._highlight_seq:
            self._highlight_seq = live.highlight_seq
            self._highlight_id = live.highlight_id
            self._highlight_started = now
        if live.phase == store_mod.PHASE_RESULT and self._last_phase != store_mod.PHASE_RESULT:
            self._result_shown_at = now
            self._result_score = live.score
            self._result_name = live.player_name or ""
        self._last_phase = live.phase
        self._live = live
        self._live_stale = (time.time() - live.updated_at) > LIVE_STALE_SECONDS

    def _standings_band(self, h: int) -> tuple:
        """The vertical space standings rows live in - constant regardless of
        whether the NOW PLAYING panel is currently shown, so the board never
        jumps as the panel slides in and out of its own reserved area."""
        top = int(h * 0.20)
        panel_h = theme.scaled(h, 0.16)
        bottom = h - panel_h - theme.scaled(h, 0.05)
        return top, bottom

    def _display_row_height(self, h: int, top: int, bottom: int, count: int) -> int:
        base = theme.scaled(h, 0.062)
        if count <= 0:
            return base
        return max(20, min(base, (bottom - top) // count))

    def _refresh_entries(self, now: float) -> None:
        self._top = self.store.top(self._board_rows)
        self._total = self.store.total_players()
        top, bottom = self._standings_band(self._size[1])
        row_h = self._display_row_height(self._size[1], top, bottom, len(self._top))
        for i, entry in enumerate(self._top):
            target = i * row_h
            anim = self._row_anims.get(entry.id)
            if anim is None:
                anim = Anim(float(target))
                self._row_anims[entry.id] = anim
            anim.set(float(target), now)
        live_ids = {e.id for e in self._top}
        for stale_id in [k for k in self._row_anims if k not in live_ids]:
            del self._row_anims[stale_id]

        if self.mode == "edit":
            self._refresh_all(keep_selection=True)

    def _refresh_all(self, keep_selection: bool = True) -> None:
        selected = self._selected_entry()
        selected_id = selected.id if selected else None
        self._all = self.store.list_all()
        self._edit_list.set_items(self._all, keep_selection=keep_selection)
        if selected_id is not None:
            idx = self._index_of_id(selected_id)
            if idx is not None:
                self._edit_list.select(idx, notify=False)

    # -------------------------------------------------------------- layout

    def _layout(self) -> None:
        w, h = self._size
        full = pygame.Rect(0, 0, w, h)
        if self.mode == "edit":
            self._layout_edit(full)
        for dlg in (self._confirm_delete, self._confirm_clear1, self._confirm_clear2,
                    self._confirm_merge):
            dlg.layout(full)

    def _layout_edit(self, full: pygame.Rect) -> None:
        w, h = full.width, full.height
        pad = theme.scaled(h, 0.02)
        banner_h = theme.scaled(h, 0.11)
        btn_row_h = theme.scaled(h, 0.06)
        self._banner_rect = pygame.Rect(0, 0, w, banner_h)

        btn_row = pygame.Rect(pad, banner_h + pad, w - pad * 2, btn_row_h)
        buttons = [self._btn_rename, self._btn_time, self._btn_add, self._btn_delete,
                   self._btn_clear, self._btn_exit]
        gap = theme.scaled(h, 0.012)
        bw = (btn_row.width - gap * (len(buttons) - 1)) // len(buttons)
        for rect, btn in zip(split_cols(btn_row, [bw] * len(buttons), gap=gap), buttons):
            btn.layout(rect)

        top = btn_row.bottom + pad
        add_h = theme.scaled(h, 0.09) if self._editing == "add" else 0
        if self._editing == "add":
            self._add_rect = pygame.Rect(pad, top, w - pad * 2, add_h)
            inner = self._add_rect.inflate(-pad * 2, -pad * 2)
            field_h = inner.height
            cols = split_cols(inner, [int(inner.width * 0.32), int(inner.width * 0.22),
                                       int(inner.width * 0.18), int(inner.width * 0.18)],
                               gap=pad)
            self._add_name.layout(cols[0].inflate(0, -int(field_h * 0.15)))
            self._add_time.layout(cols[1].inflate(0, -int(field_h * 0.15)))
            self._btn_add_confirm.layout(cols[2].inflate(0, -int(field_h * 0.15)))
            self._btn_add_cancel.layout(cols[3].inflate(0, -int(field_h * 0.15)))
            top += add_h + pad

        list_rect = pygame.Rect(pad, top, w - pad * 2, max(0, h - top - pad))
        self._edit_list.row_height = theme.scaled(h, 0.062)
        self._edit_list.layout(list_rect)

    # ------------------------------------------------------------- events

    def handle(self, event: pygame.event.Event) -> bool:
        """Returns False when the app has decided it's time to quit."""
        self._layout()
        if is_quit_key(event):
            self.quit = True
            return False

        if self.mode == "display":
            self._handle_display(event)
        else:
            self._handle_edit(event)
        return not self.quit

    def _handle_display(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e:
                self._enter_edit()
            elif event.key == pygame.K_ESCAPE:
                self.quit = True

    def _handle_edit(self, event: pygame.event.Event) -> None:
        # Modal dialogs eat everything while open.
        for dlg in (self._confirm_merge, self._confirm_clear2, self._confirm_clear1,
                    self._confirm_delete):
            if dlg.open:
                dlg.handle(event)
                return

        if self._editing == "name":
            self._handle_inline(event, self._rename_field)
            return
        if self._editing == "time":
            self._handle_inline(event, self._time_field)
            return
        if self._editing == "add":
            self._handle_add(event)
            return

        self._focus.handle(event)
        for widget in (self._btn_rename, self._btn_time, self._btn_add,
                       self._btn_delete, self._btn_clear, self._btn_exit):
            if widget.handle(event):
                return
        # Not in the focus rotation (see the constructor), but arrow keys
        # always drive it: flip `.focused` on just for this call so
        # `ScrollList.handle`'s keyboard branch fires.
        self._edit_list.focused = True
        consumed = self._edit_list.handle(event)
        self._edit_list.focused = False
        if consumed:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_e or event.key == pygame.K_ESCAPE:
                self._exit_edit()
            elif event.key == pygame.K_r:
                self._start_rename()
            elif event.key == pygame.K_t:
                self._start_time()
            elif event.key in (pygame.K_n, pygame.K_a):
                self._start_add()
            elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if event.mod & pygame.KMOD_SHIFT:
                    self._ask_clear()
                else:
                    self._ask_delete()

    def _handle_inline(self, event: pygame.event.Event, field) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._editing = None
            self._editing_id = None
            return
        field.focused = True
        field.handle(event)

    def _handle_add(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._cancel_add()
            return
        if self._btn_add_confirm.handle(event):
            return
        if self._btn_add_cancel.handle(event):
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._add_time.rect.collidepoint(event.pos):
                self._add_name.focused, self._add_time.focused = False, True
            elif self._add_name.rect.collidepoint(event.pos):
                self._add_time.focused, self._add_name.focused = False, True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self._add_name.focused, self._add_time.focused = (
                self._add_time.focused, self._add_name.focused)
            return
        # Whichever field currently has focus gets the keystroke.
        if self._add_time.focused:
            self._add_time.handle(event)
        else:
            self._add_name.focused = True
            self._add_name.handle(event)

    # -------------------------------------------------------------- actions

    def _enter_edit(self) -> None:
        self.mode = "edit"
        self._refresh_all(keep_selection=False)
        self._layout()

    def _exit_edit(self) -> None:
        self._editing = None
        self._editing_id = None
        self.mode = "display"
        self._layout()

    def _activate_row(self, index: int) -> None:
        self._start_rename()

    def _start_rename(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._editing = "name"
        self._editing_id = entry.id
        self._rename_field.set_value(entry.name)
        self._rename_field.focused = True

    def _start_time(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._editing = "time"
        self._editing_id = entry.id
        self._time_field.set_number(entry.score)
        self._time_field.focused = True

    def _start_add(self) -> None:
        self._editing = "add"
        self._add_name.set_value("")
        # Empty rather than "0.0" - an operator typing a time straight away
        # shouldn't have to clear a pre-filled zero first.
        self._add_time.set_value("")
        self._add_time.invalid = True
        self._add_name.focused = True
        self._add_time.focused = False
        self._layout()

    def _cancel_add(self) -> None:
        self._editing = None
        self._layout()

    def _commit_rename(self, text: str) -> None:
        entry_id = self._editing_id
        if entry_id is None:
            return
        clean = normalise_name(text)
        if not clean:
            self.toast.show("Name can't be empty", kind="error", now=self._now())
            return
        other = None
        for e in self._all:
            if e.id != entry_id and normalise_name(e.name) == clean:
                other = e
                break
        if other is not None:
            self._pending_rename = (entry_id, clean)
            self._confirm_merge.message = (
                f"{clean} already exists with {format_score(other.score)} - "
                f"merge and keep the higher score?")
            self._confirm_merge.layout(pygame.Rect(0, 0, *self._size))
            self._confirm_merge.show()
            return
        try:
            self.store.rename(entry_id, clean)
        except (InvalidName, KeyError) as exc:
            self.toast.show(f"Couldn't rename: {exc}", kind="error", now=self._now())
        else:
            self.toast.show(f"Renamed to {clean}", kind="success", now=self._now())
            self._editing = None
            self._editing_id = entry_id
            self._refresh_all_after(entry_id)

    def _confirm_merge_rename(self) -> None:
        if self._pending_rename is None:
            return
        entry_id, clean = self._pending_rename
        self._pending_rename = None
        try:
            survivor = self.store.rename(entry_id, clean)
        except (InvalidName, KeyError) as exc:
            self.toast.show(f"Couldn't merge: {exc}", kind="error", now=self._now())
        else:
            self.toast.show(f"Merged into {clean}", kind="success", now=self._now())
            self._editing = None
            self._refresh_all_after(survivor.id)

    def _cancel_merge(self) -> None:
        self._pending_rename = None

    def _commit_time(self, number: float) -> None:
        entry_id = self._editing_id
        if entry_id is None:
            return
        try:
            self.store.set_score(entry_id, int(number))
        except KeyError as exc:
            self.toast.show(f"Couldn't update: {exc}", kind="error", now=self._now())
        else:
            self.toast.show(f"Set to {format_score(int(number))}", kind="success",
                             now=self._now())
            self._editing = None
            self._refresh_all_after(entry_id)

    def _commit_add(self) -> None:
        name = self._add_name.get_value()
        clean = normalise_name(name)
        if not clean:
            self.toast.show("Name can't be empty", kind="error", now=self._now())
            return
        number = self._add_time.get_number()
        if number is None:
            self._add_time.invalid = True
            self.toast.show("Enter a valid time", kind="error", now=self._now())
            return
        existed = any(normalise_name(e.name) == clean for e in self._all)
        try:
            entry = self.store.add_entry(clean, int(number))
        except InvalidName as exc:
            self.toast.show(f"Couldn't add: {exc}", kind="error", now=self._now())
            return
        if existed:
            self.toast.show(f"{clean} already existed - kept the higher score",
                             kind="info", now=self._now())
        else:
            self.toast.show(f"Added {clean}", kind="success", now=self._now())
        self._editing = None
        self._refresh_all_after(entry.id)
        self._layout()

    def _ask_delete(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._editing_id = entry.id
        self._confirm_delete.message = (
            f"Delete {entry.name}'s score of {format_score(entry.score)}?")
        self._confirm_delete.layout(pygame.Rect(0, 0, *self._size))
        self._confirm_delete.show()

    def _do_delete(self) -> None:
        entry_id = self._editing_id
        if entry_id is None:
            return
        entry = self.store.get(entry_id)
        name = entry.name if entry else "entry"
        self.store.delete(entry_id)
        self.toast.show(f"Deleted {name}", kind="success", now=self._now())
        self._editing_id = None
        self._refresh_all(keep_selection=True)

    def _ask_clear(self) -> None:
        if not self._all:
            return
        self._confirm_clear1.message = (
            f"Clear all {len(self._all)} entries? This cannot be undone.")
        self._confirm_clear1.layout(pygame.Rect(0, 0, *self._size))
        self._confirm_clear1.show()

    def _ask_clear_again(self) -> None:
        self._confirm_clear2.layout(pygame.Rect(0, 0, *self._size))
        self._confirm_clear2.show()

    def _do_clear(self) -> None:
        self.store.clear()
        self.toast.show("Cleared the leaderboard", kind="success", now=self._now())
        self._refresh_all(keep_selection=False)

    def _refresh_all_after(self, keep_id: int) -> None:
        self._all = self.store.list_all()
        self._edit_list.set_items(self._all, keep_selection=True)
        idx = self._index_of_id(keep_id)
        if idx is not None:
            self._edit_list.select(idx, notify=False)

    def _now(self) -> float:
        """The most recent `now` seen from `poll`/`draw` - not a fresh
        `time.monotonic()` call, so a caller driving the app on a synthetic
        clock (tests, the render script) gets toast timings on that same
        clock instead of the real one."""
        return self._now_cache

    # --------------------------------------------------------------- draw

    def draw(self, surface: pygame.Surface, now: float) -> None:
        self._now_cache = now
        self._size = surface.get_size()
        self._layout()
        theme.backdrop(surface)
        if self.mode == "display":
            self._draw_display(surface, now)
        else:
            self._draw_edit(surface, now)
        self.toast.draw(surface, now)

    # ---------------------------------------------------------- display draw

    def _draw_display(self, surface: pygame.Surface, now: float) -> None:
        w, h = surface.get_size()
        reveal = theme.ease_out_cubic(
            (now - self._started_at) / FADE_IN_SECONDS) if self._started_at else 1.0

        self._draw_header(surface, now, reveal)

        live_active = self._live.phase in ACTIVE_PHASES and not self._live_stale
        result_active = (self._live.phase == store_mod.PHASE_RESULT and not self._live_stale
                          and now - self._result_shown_at < RESULT_TAG_SECONDS)

        top, bottom = self._standings_band(h)
        self._draw_rows(surface, top, bottom, now, reveal)

        self._panel_anim.set(1.0 if (live_active or result_active) else 0.0, now)
        amount = self._panel_anim.update(now, duration=0.25)
        if amount > 0.001:
            if live_active:
                self._draw_now_playing(surface, now, amount)
            elif result_active:
                self._draw_result_tag(surface, now, amount)

        hint_size = theme.scaled(h, 0.018)
        theme.text(surface, "E · EDIT", "body", hint_size, theme.INK_FAINT,
                   alpha=140, topright=(w - theme.scaled(h, 0.02), theme.scaled(h, 0.012)))

    def _draw_header(self, surface: pygame.Surface, now: float, reveal: float) -> None:
        w, h = surface.get_size()
        theme.text(surface, "SEARCHLIGHT", "display", theme.scaled(h, 0.085), theme.GOLD,
                   center=(w // 2, int(h * 0.085)), shadow=True, alpha=int(255 * reveal))
        subtitle = (f"{self._total} PLAYER{'S' if self._total != 1 else ''} SO FAR · "
                    f"LONGEST IN THE LIGHT" if self._total else "NOBODY HAS PLAYED YET")
        theme.text(surface, subtitle, "body", theme.scaled(h, 0.026), theme.INK_MUTED,
                   center=(w // 2, int(h * 0.15)), alpha=int(230 * reveal))

    def _draw_rows(self, surface: pygame.Surface, top: int, bottom: int, now: float,
                    reveal: float) -> None:
        w, h = surface.get_size()
        if not self._top:
            theme.text(surface, "BE THE FIRST", "display", theme.scaled(h, 0.07),
                       theme.INK_FAINT, center=(w // 2, (top + bottom) // 2),
                       alpha=int(255 * reveal))
            theme.text(surface, "STEP UP TO THE CAMERA", "body", theme.scaled(h, 0.024),
                       theme.INK_FAINT, center=(w // 2, (top + bottom) // 2 + theme.scaled(h, 0.06)),
                       alpha=int(200 * reveal))
            return

        row_h = self._display_row_height(h, top, bottom, len(self._top))
        left = int(w * 0.10)
        right = int(w * 0.90)
        name_size = theme.scaled(h, 0.040)

        for index, entry in enumerate(self._top):
            anim = self._row_anims.get(entry.id)
            y = top + (anim.update(now) if anim else index * row_h)
            rect = pygame.Rect(left - 20, int(y) + 3, (right - left) + 40, row_h - 8)

            highlighted = (entry.id == self._highlight_id
                           and now - self._highlight_started < HIGHLIGHT_SECONDS)
            podium = index < len(theme.PODIUM)
            colour = theme.PODIUM[index] if podium else theme.INK

            if highlighted:
                pulse = theme.pulse(now, speed=5.0)
                fill = theme.mix(theme.BG_RAISED, theme.MAGENTA, 0.20 + 0.35 * pulse)
                theme.panel(surface, rect, fill, radius=14)
                theme.glow(surface, rect, theme.MAGENTA, spread=int(6 + 6 * pulse), radius=14,
                           intensity=int(60 + 40 * pulse))
            elif podium:
                theme.panel(surface, rect, theme.mix(theme.BG_RAISED, colour, 0.12), radius=14,
                            border=theme.mix(theme.BG_RAISED, colour, 0.5), width=1)
            elif index % 2 == 0:
                theme.panel(surface, rect, theme.BG_RAISED, radius=14)

            row_size = int(name_size * 1.15) if podium else name_size
            centre_y = rect.centery
            theme.text(surface, f"{index + 1}", "display", row_size, colour,
                       midleft=(left, centre_y), alpha=int(255 * reveal))
            theme.text(surface, entry.name, "display", row_size,
                       theme.INK if not podium else colour,
                       midleft=(left + int(w * 0.09), centre_y),
                       max_width=int(w * 0.55), alpha=int(255 * reveal))
            theme.text(surface, format_score(entry.score), "display", row_size, colour,
                       midright=(right, centre_y), alpha=int(255 * reveal))

    def _draw_now_playing(self, surface: pygame.Surface, now: float, amount: float) -> None:
        w, h = surface.get_size()
        panel_h = theme.scaled(h, 0.16)
        target = pygame.Rect(int(w * 0.06), h - panel_h - theme.scaled(h, 0.03),
                              int(w * 0.88), panel_h)
        rect = target.move(0, int((1 - theme.ease_out_cubic(amount)) * panel_h * 0.6))
        theme.panel(surface, rect, theme.BG_RAISED, radius=20, border=theme.MAGENTA, width=3,
                    alpha=int(255 * amount))
        theme.glow(surface, rect, theme.MAGENTA, spread=10, radius=20, intensity=int(50 * amount))

        getting_ready = self._live.phase in (store_mod.PHASE_FIND_PLAYER,
                                              store_mod.PHASE_COUNTDOWN)
        label = "GETTING READY" if getting_ready else "IN THE LIGHT"
        theme.text(surface, label, "body", theme.scaled(h, 0.022), theme.MAGENTA,
                   topleft=(rect.left + 26, rect.top + 14), alpha=int(255 * amount))
        theme.text(surface, (self._live.player_name or "").upper(), "display",
                   theme.scaled(h, 0.05), theme.INK,
                   midleft=(rect.left + 26, rect.centery + int(h * 0.02)),
                   max_width=int(rect.width * 0.35), alpha=int(255 * amount))

        theme.text(surface, format_score(self._live.score), "display",
                   theme.scaled(h, 0.075), theme.GOLD,
                   midright=(rect.right - 30, rect.centery), alpha=int(255 * amount))

        lives, max_lives = self._live.lives, max(self._live.max_lives, 1)
        size = theme.scaled(h, 0.042)
        start_x = int(w * 0.50)
        for i in range(max_lives):
            theme.draw_heart(surface, (start_x + i * int(size * 1.3), rect.centery), size,
                              theme.MAGENTA if i < lives else theme.INK_FAINT)

    def _draw_result_tag(self, surface: pygame.Surface, now: float, amount: float) -> None:
        w, h = surface.get_size()
        panel_h = theme.scaled(h, 0.10)
        target = pygame.Rect(int(w * 0.30), h - panel_h - theme.scaled(h, 0.03),
                              int(w * 0.40), panel_h)
        rect = target.move(0, int((1 - theme.ease_out_cubic(amount)) * panel_h * 0.6))
        theme.panel(surface, rect, theme.BG_RAISED, radius=18, border=theme.GOLD, width=2,
                    alpha=int(230 * amount))
        theme.text(surface, "JUST FINISHED", "body", theme.scaled(h, 0.02), theme.GOLD,
                   midtop=(rect.centerx, rect.top + 10), alpha=int(255 * amount))
        label = f"{self._result_name.upper()} · {format_score(self._result_score)}"
        theme.text(surface, label, "display", theme.scaled(h, 0.04), theme.INK,
                   center=(rect.centerx, rect.centery + theme.scaled(h, 0.015)),
                   max_width=rect.width - 40, alpha=int(255 * amount))

    # ------------------------------------------------------------- edit draw

    def _draw_edit(self, surface: pygame.Surface, now: float) -> None:
        w, h = surface.get_size()
        theme.dim(surface, pygame.Rect(0, 0, w, h), 40, color=theme.MAGENTA)

        theme.panel(surface, self._banner_rect, theme.mix(theme.BG_RAISED, theme.MAGENTA, 0.18),
                    radius=0)
        pygame.draw.line(surface, theme.MAGENTA, self._banner_rect.bottomleft,
                         self._banner_rect.bottomright, 3)
        theme.text(surface, "EDIT LEADERBOARD", "display", theme.scaled(h, 0.05),
                   theme.MAGENTA, midleft=(theme.scaled(h, 0.02), self._banner_rect.centery - 6))
        theme.text(surface, f"{len(self._all)} ENTR{'Y' if len(self._all) == 1 else 'IES'}",
                   "body", theme.scaled(h, 0.02), theme.CYAN,
                   midleft=(theme.scaled(h, 0.02), self._banner_rect.centery + theme.scaled(h, 0.03)))
        theme.text(surface, "E / ESC · EXIT", "body", theme.scaled(h, 0.018),
                   theme.INK_MUTED, topright=(w - theme.scaled(h, 0.02), theme.scaled(h, 0.015)))

        for btn in (self._btn_rename, self._btn_time, self._btn_add, self._btn_delete,
                    self._btn_clear, self._btn_exit):
            btn.draw(surface, now)

        if self._editing == "add":
            theme.panel(surface, self._add_rect, theme.BG_RAISED, radius=14, border=theme.CYAN,
                       width=2)
            self._add_name.draw(surface, now)
            self._add_time.draw(surface, now)
            self._btn_add_confirm.draw(surface, now)
            self._btn_add_cancel.draw(surface, now)

        if not self._all:
            theme.text(surface, "NO ENTRIES YET", "display", theme.scaled(h, 0.045),
                       theme.INK_FAINT, center=self._edit_list.rect.center)
        else:
            theme.panel(surface, self._edit_list.rect.inflate(16, 16), theme.BG_SUNK, radius=16)
            self._edit_list.draw(surface, now)

        for dlg in (self._confirm_delete, self._confirm_clear1, self._confirm_clear2,
                    self._confirm_merge):
            dlg.draw(surface, now)

    def _render_edit_row(self, surface: pygame.Surface, rect: pygame.Rect, item: Entry,
                          index: int, selected: bool, hovered: bool, now: float) -> None:
        w = rect.width
        inner = rect.inflate(-8, -6)
        if selected:
            theme.panel(surface, inner, theme.mix(theme.BG_RAISED, theme.CYAN, 0.18), radius=10,
                       border=theme.CYAN, width=2)
        elif hovered:
            theme.panel(surface, inner, theme.BG_HOVER, radius=10)
        elif index % 2 == 0:
            theme.panel(surface, inner, theme.BG_RAISED, radius=10)

        size = max(14, int(rect.height * 0.36))
        rank_x = inner.left + 20
        name_x = inner.left + int(w * 0.12)
        score_x = inner.right - 20

        theme.text(surface, f"{index + 1}", "bold", size, theme.INK_MUTED,
                   midleft=(rank_x, inner.centery))

        editing_this = self._editing_id == item.id
        if editing_this and self._editing == "name":
            field_rect = pygame.Rect(name_x, inner.top + 6, int(w * 0.45), inner.height - 12)
            self._rename_field.layout(field_rect)
            self._rename_field.draw(surface, now)
        else:
            theme.text(surface, item.name, "bold", size, theme.INK,
                       midleft=(name_x, inner.centery), max_width=int(w * 0.45))

        if editing_this and self._editing == "time":
            field_rect = pygame.Rect(score_x - int(w * 0.22), inner.top + 6, int(w * 0.22),
                                      inner.height - 12)
            self._time_field.layout(field_rect)
            self._time_field.draw(surface, now)
        else:
            theme.text(surface, format_score(item.score), "mono", size, theme.GOLD,
                       midright=(score_x, inner.centery))


# ----------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Searchlight - leaderboard window")
    parser.add_argument("--display", type=int, default=1, help="monitor index")
    parser.add_argument("--db", default=str(store_mod.DEFAULT_DB))
    parser.add_argument("--fullscreen", action="store_true",
                        help="take over the display (default is a window)")
    parser.add_argument("--pos", default=None,
                        help="window position as X,Y (windowed mode only)")
    parser.add_argument("--size", default="1280x720")
    args = parser.parse_args(argv)

    if args.pos and not args.fullscreen:
        os.environ["SDL_VIDEO_WINDOW_POS"] = args.pos

    pygame.init()
    pygame.display.set_caption("Searchlight Leaderboard")
    width, height = (int(v) for v in args.size.lower().split("x"))
    window = Window((width, height), args.fullscreen, args.display)
    store = Store(Path(args.db))
    app = LeaderboardApp(store, (width, height))
    clock = pygame.time.Clock()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if window.handle(event):
                    continue
                if event.type == pygame.QUIT:
                    running = False
                elif not app.handle(event):
                    running = False
            now = time.monotonic()
            app.poll(now)
            app.draw(window.surface, now)
            pygame.display.flip()
            clock.tick(60)
    finally:
        store.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
