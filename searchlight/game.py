"""The player window: attract, name entry, finding the player, the run, the result.

Runs as its own process so it and the leaderboard can each claim a display; the
two share only the SQLite file in `store.py`.

Screen flow:

    ATTRACT -> NAME_ENTRY -> FIND_PLAYER -> COUNTDOWN -> PLAYING -> RESULT -> ATTRACT
       ^                         ^              |
       |                         +--------------+  (player lost during the countdown)
       +-- settings overlay (Tab or the gear, attract screen only)

FIND_PLAYER exists because a countdown with nobody in front of the camera is how a
run gets wasted: the square starts moving, the player walks up two seconds late
and has already lost a life. So the countdown only starts once a head has sat in
the start square for `face_lock_seconds`, and it falls back here if the head
leaves during the countdown.

With "Record runs" on, the raw camera feed is saved from the first countdown to
the result screen, one file per run. A player who never gets past the countdown
leaves no file behind.

All state changes happen in `update()`, timed by an injectable clock, so the
whole flow can be tested - and rendered to PNGs by `scripts/render_screens.py` -
without a camera, a window or real time passing.
"""

from __future__ import annotations

# pygame first: it and opencv both bundle SDL2, and whichever registers its
# Objective-C classes first wins. Keep this above anything that imports cv2.
import pygame

import argparse
import math
import os
import random
import re
import sqlite3
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, Tuple

import numpy as np
import pygame.gfxdraw

from . import fx
from . import store as store_mod
from .rules import Run, RunResult
from .settings import SETTINGS_BY_KEY, Settings
from .square import Square, SquareConfig
from .store import (MAX_NAME, Entry, Store, SubmitResult, format_score, normalise_name,
                    to_score)
from .tracking.base import CameraOptions, HeadDetection
from .ui import theme
from .ui.window import Window, is_quit_key

Point = Tuple[float, float]

# Screens share their names with the live-state phases the leaderboard reads.
ATTRACT = store_mod.PHASE_ATTRACT
NAME_ENTRY = store_mod.PHASE_NAME_ENTRY
FIND_PLAYER = store_mod.PHASE_FIND_PLAYER
COUNTDOWN = store_mod.PHASE_COUNTDOWN
PLAYING = store_mod.PHASE_PLAYING
RESULT = store_mod.PHASE_RESULT

DEFAULT_RECORDINGS = Path(__file__).resolve().parent.parent / "recordings"
RESULT_SECONDS = 8.0            # the result screen returns to attract by itself
RESULT_KEY_DELAY = 0.8          # ...but ignores keys this long, so a key held
                                # from play cannot skip it unseen
LOSS_FLASH_SECONDS = 0.6
GO_SECONDS = 0.8                # "GO!" after the countdown
FIND_TIMEOUT_SECONDS = 30.0     # nobody in view this long -> back to attract
FIND_HINT_AFTER = 6.0           # when to suggest the far-player settings
LOCK_DROPOUT_SECONDS = 0.2      # a tracker blink this short keeps the lock
NAME_IDLE_SECONDS = 60.0        # someone walked off mid-name
TOAST_SECONDS = 2.6

LIVE_INTERVAL = 0.1             # live-state writes: at most 10 Hz...
LIVE_HEARTBEAT = 2.0            # ...and only on change, plus this heartbeat
POLL_INTERVAL = 0.5             # settings / leaderboard revision polling

VEIL_ALPHA = 200                # how dark the room is outside the light
ATTRACT_VEIL_ALPHA = 212
TINT_DIM = (72, 76, 112)        # camera multiply-tint behind text-heavy screens
TINT_RESULT = (46, 48, 76)      # darker still, so the result and confetti pop

ALLOWED_NAME_CHARS = set(string.ascii_uppercase + string.digits + " -_")

# Keys that never start a game from the attract screen: a Cmd held on the way to
# Cmd+Q, or the fullscreen toggle, should not drop someone into name entry.
_MODIFIER_KEYS = {pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_LCTRL, pygame.K_RCTRL,
                  pygame.K_LALT, pygame.K_RALT, pygame.K_LMETA, pygame.K_RMETA,
                  pygame.K_LGUI, pygame.K_RGUI, pygame.K_CAPSLOCK, pygame.K_NUMLOCK,
                  pygame.K_MODE, pygame.K_F11}

# Result outcomes, best first.
NEW_TOP, PERSONAL_BEST, FIRST_RUN, NOT_BEST, NOT_SAVED = (
    "new_top", "personal_best", "first_run", "not_best", "not_saved")

MODE_LABELS = dict(SETTINGS_BY_KEY["mode"].choices)


class CameraLike(Protocol):
    """What the game needs from `camera.CameraThread` - a stub satisfies it in tests."""

    error: Optional[str]

    def latest(self) -> HeadDetection: ...

    def configure(self, options: CameraOptions) -> None: ...

    def reset_tracking(self) -> None: ...

    def start_recording(self, path: Path) -> None: ...

    def stop_recording(self, keep: bool = True) -> Optional[Path]: ...

    def close(self) -> None: ...


def _default_menu_factory(settings: Settings, on_save: Callable[[Settings], None],
                          on_close: Callable[[], None]) -> Any:
    # Imported lazily: the menu is optional furniture, and a kiosk whose menu
    # module is missing or broken should still run games.
    from .settings_menu import SettingsMenu
    return SettingsMenu(settings, on_save=on_save, on_close=on_close)


class Game:
    def __init__(self, store: Store, camera: CameraLike, dev: bool = False,
                 seed: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic,
                 menu_factory: Optional[Callable[..., Any]] = None,
                 recordings: Optional[Path] = None) -> None:
        self.store = store
        self.camera = camera
        self.dev = dev
        self.recordings = recordings                   # folder for run videos
        self.recording = False
        self.clock = clock
        self.menu_factory = menu_factory or _default_menu_factory
        self._rng = random.Random(seed)

        now = clock()
        self.running = True
        self.screen = ATTRACT
        self.phase_started = now
        self._last_update = now

        self.settings: Settings = store.load_settings()
        self._settings_rev = store.revisions()[1]
        self._board_rev: Optional[int] = None
        self._last_poll = -math.inf
        self.leaders: List[Entry] = []

        self.detection: HeadDetection = camera.latest()
        self.head: Optional[Point] = None
        self._marker: Optional[Point] = None           # smoothed head, for drawing
        self._marker_seen = -math.inf

        # --dev: the mouse is the head, and leaving the window means "nobody".
        self.dev_head: Optional[Point] = None
        self._mouse_inside = True

        self.name = ""
        self.returning: Optional[Entry] = None         # typed name's existing row
        self._name_touched = now
        self._name_error_at = -math.inf

        self.run = self._new_run()
        self.last_result: Optional[RunResult] = None

        # FIND_PLAYER
        self.lock = 0.0                                # seconds held in the square
        self._lock_shown = 0.0                         # eased, for the ring
        self._last_inside = -math.inf
        self._last_seen = now
        self.player_best: Optional[Entry] = None

        # COUNTDOWN
        self._countdown_total = float(self.settings.countdown_seconds)
        self._away_since: Optional[float] = None

        # PLAYING
        self.loss_flash_at = -math.inf
        self._lost_heart: Optional[Tuple[int, float]] = None   # (index, when)
        self._resumed_at = -math.inf                   # end of the last respawn

        # RESULT
        self.final_score = 0
        self.submit: Optional[SubmitResult] = None
        self.total_players = 0
        self.outcome = NOT_BEST

        self.menu: Any = None
        self._toast: Optional[Tuple[str, float]] = None

        self._live_state: Optional[tuple] = None
        self._live_at = -math.inf

        # Attract-mode light: a slow, fixed-size square that roams on its own.
        self._demo = self._demo_square(self._aspect())

        # Drawing state.
        self.confetti = fx.Confetti(seed)
        self.transition = fx.Transition()
        self._pending_transition: Optional[Tuple[float, float]] = (0.0, 0.6)
        self.veil = fx.Veil(theme.BG_SUNK, VEIL_ALPHA)
        self.attract_veil = fx.Veil((8, 9, 22), ATTRACT_VEIL_ALPHA)
        self._view = pygame.Rect(0, 0, 1280, 720)
        self._screen_size = (1280, 720)
        self._cam_key: tuple = ()
        self._cam_surface: Optional[pygame.Surface] = None
        self._placeholder: Optional[pygame.Surface] = None
        self._flash_layer: Optional[pygame.Surface] = None
        self.gear_rect = pygame.Rect(0, 0, 0, 0)
        self._gear_hover = False

    # ------------------------------------------------------------ settings

    def _aspect(self, detection: Optional[HeadDetection] = None) -> float:
        width, height = self._frame_size(detection or self.detection)
        return width / height

    def _frame_size(self, detection: HeadDetection) -> Tuple[int, int]:
        """The size of the picture being shown: the camera frame if there is one,
        otherwise what the camera is set to deliver."""
        if detection.frame is not None and detection.frame.ndim == 3:
            height, width = detection.frame.shape[:2]
            if width > 0 and height > 0:
                return width, height
        width, height = (int(v) for v in self.settings.resolution.split("x"))
        return width, height

    def _new_run(self) -> Run:
        """A fresh square and rules from the current settings - the only place
        square/rules settings take effect, so they never change mid-run."""
        square = Square(self.settings.square_config(self._aspect()),
                        seed=self._rng.randrange(1 << 30))
        return Run(square, self.settings.run_config())

    @staticmethod
    def _demo_square(aspect: float) -> Square:
        return Square(SquareConfig(aspect=aspect, start_side=0.46, min_side=0.46,
                                   shrink_per_second=0, start_speed=0.07, max_speed=0.07,
                                   speed_ramp=0, turn_rate=0.8, margin=0.10), seed=7)

    def _apply_settings(self, settings: Settings) -> None:
        old, self.settings = self.settings, settings
        if settings.camera_options() != old.camera_options():
            # Detection settings are live: the camera reconfigures itself.
            self.camera.configure(settings.camera_options())
        if self.screen == FIND_PLAYER:
            # No run has started yet, so the parked square may follow the change.
            self.run = self._new_run()

    def _poll(self, now: float, force: bool = False) -> None:
        """Pick up saved settings and leaderboard changes, a couple of times a second."""
        if not force and now - self._last_poll < POLL_INTERVAL:
            return
        self._last_poll = now
        try:
            board_rev, settings_rev = self.store.revisions()
            if settings_rev != self._settings_rev:
                self._settings_rev = settings_rev
                self._apply_settings(self.store.load_settings())
            if board_rev != self._board_rev:
                self._board_rev = board_rev
                self.leaders = self.store.top(3)
                if self.screen == NAME_ENTRY:
                    self._lookup_name()
        except sqlite3.Error:
            pass            # the leaderboard may hold a write lock; try next poll

    # ------------------------------------------------------------ settings menu

    def open_settings(self) -> bool:
        if self.screen != ATTRACT or self.menu is not None:
            return False
        try:
            self.menu = self.menu_factory(self.settings, on_save=self._on_settings_saved,
                                          on_close=self._close_settings)
        except Exception as exc:  # noqa: BLE001 - a missing menu must not stop play
            print(f"Settings menu unavailable: {exc!r}")
            self.toast("SETTINGS MENU UNAVAILABLE")
            self.menu = None
            return False
        return True

    def _on_settings_saved(self, settings: Settings) -> None:
        self.store.save_settings(settings)
        self._poll(self.clock(), force=True)
        self.toast("SETTINGS SAVED")

    def _close_settings(self) -> None:
        self.menu = None

    def toast(self, message: str) -> None:
        self._toast = (message, self.clock())

    # ------------------------------------------------------------- lifecycle

    def go(self, screen: str, transition: Optional[Tuple[float, float]] = (0.0, 0.35)
           ) -> None:
        """Switch screens. `transition` is (slide, seconds) for the cross-fade, or None."""
        now = self.clock()
        self.screen = screen
        self.phase_started = now

        if screen == ATTRACT:
            self._stop_recording(keep=False)      # nobody got as far as playing
            self.name = ""
            self.returning = None
            self.confetti.clear()
        elif screen == NAME_ENTRY:
            self.name = ""
            self.returning = None
            self._name_touched = now
        elif screen == FIND_PLAYER:
            self.camera.reset_tracking()          # a new player: forget the old face
            self.run = self._new_run()
            self.last_result = None
            self.lock = 0.0
            self._last_inside = -math.inf
            self._last_seen = now
            try:
                self.player_best = self.store.find(self.name)
            except sqlite3.Error:
                self.player_best = None
        elif screen == COUNTDOWN:
            self._countdown_total = float(self.settings.countdown_seconds)
            self._away_since = None
            self._start_recording()
        elif screen == PLAYING:
            self.run.start(now)
            self.last_result = None
            self._lost_heart = None
            self._resumed_at = -math.inf
            self.loss_flash_at = -math.inf
        elif screen == RESULT:
            self._stop_recording(keep=True)

        if transition is not None:
            self._pending_transition = transition
        self._publish(now, force=True)

    def _start_recording(self) -> None:
        # Once per player: a fall-back to FIND_PLAYER keeps the same file going.
        if self.recording or not self.settings.record_runs or self.recordings is None:
            return
        self.camera.start_recording(recording_path(self.recordings, self.name, datetime.now()))
        self.recording = True

    def _stop_recording(self, keep: bool) -> None:
        if self.recording:
            self.camera.stop_recording(keep)
            self.recording = False

    def close(self) -> None:
        self._stop_recording(keep=self.screen == PLAYING)
        try:
            self.store.set_live(store_mod.PHASE_IDLE)
        except sqlite3.Error:
            pass

    # ---------------------------------------------------------------- events

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT or is_quit_key(event):
            self.running = False
            return

        if event.type == pygame.WINDOWLEAVE:
            self._mouse_inside = False
        elif event.type == pygame.WINDOWENTER:
            self._mouse_inside = True
        elif event.type == pygame.MOUSEMOTION:
            self._mouse_inside = True
            if self.dev:
                self.dev_head = self._to_normalised(event.pos)
            self._gear_hover = self.gear_rect.collidepoint(event.pos)

        if self.menu is not None:
            # The overlay owns input while open; nothing leaks to the screen below,
            # so a stray key can never start a game behind the menu.
            self.menu.handle(event)
            return

        if (event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 1) == 1
                and self.screen == ATTRACT and self.gear_rect.collidepoint(event.pos)):
            self.open_settings()
            return

        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        now = self.clock()

        if self.screen == ATTRACT:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_TAB:
                self.open_settings()
            elif key not in _MODIFIER_KEYS:
                self.go(NAME_ENTRY, (0.04, 0.35))

        elif self.screen == NAME_ENTRY:
            self._name_touched = now
            if key == pygame.K_ESCAPE:
                self.go(ATTRACT, (-0.04, 0.3))
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if normalise_name(self.name):
                    self.name = normalise_name(self.name)
                    self.go(FIND_PLAYER, (0.0, 0.4))
                else:
                    self._name_error_at = now
            elif key == pygame.K_BACKSPACE:
                self.name = self.name[:-1]
                self._lookup_name()
            else:
                char = (event.unicode or "").upper()
                if (len(char) == 1 and char in ALLOWED_NAME_CHARS
                        and len(self.name) < MAX_NAME
                        and not (char == " " and (not self.name or self.name.endswith(" ")))):
                    self.name += char
                    self._lookup_name()

        elif self.screen in (FIND_PLAYER, COUNTDOWN):
            if key == pygame.K_ESCAPE:
                self.go(ATTRACT)

        elif self.screen == PLAYING:
            if key == pygame.K_ESCAPE:
                self._finish_run(now)

        elif self.screen == RESULT:
            if now - self.phase_started >= RESULT_KEY_DELAY and key not in _MODIFIER_KEYS:
                self.go(ATTRACT, (0.0, 0.5))

    def _lookup_name(self) -> None:
        clean = normalise_name(self.name)
        try:
            self.returning = self.store.find(clean) if clean else None
        except sqlite3.Error:
            self.returning = None

    def _to_normalised(self, pos: Tuple[int, int]) -> Optional[Point]:
        view = self._view
        if view.width <= 0 or view.height <= 0:
            return None
        if not pygame.Rect((0, 0), self._screen_size).collidepoint(pos):
            return None
        x = (pos[0] - view.left) / view.width
        y = (pos[1] - view.top) / view.height
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return (x, y)

    # ---------------------------------------------------------------- update

    def _read_head(self, detection: HeadDetection) -> Optional[Point]:
        if self.dev:
            return self.dev_head if self._mouse_inside else None
        return detection.head

    def inside(self) -> bool:
        return self.head is not None and self.run.square.contains(*self.head)

    def update(self) -> HeadDetection:
        now = self.clock()
        dt = min(0.1, max(0.0, now - self._last_update))
        self._last_update = now

        detection = self.camera.latest()
        self.detection = detection
        self._poll(now)
        self.head = self._read_head(detection)
        self._update_marker(now, dt)

        if self.screen == ATTRACT:
            aspect = self._aspect()
            if abs(aspect - self._demo.config.aspect) > 0.01:
                self._demo = self._demo_square(aspect)
            self._demo.update(dt, 0.0)
        elif self.screen == NAME_ENTRY:
            if now - self._name_touched > NAME_IDLE_SECONDS:
                self.go(ATTRACT)
        elif self.screen == FIND_PLAYER:
            self._update_find_player(now, dt)
        elif self.screen == COUNTDOWN:
            self._update_countdown(now)
        elif self.screen == PLAYING:
            self._update_playing(now)
        elif self.screen == RESULT:
            if now - self.phase_started >= RESULT_SECONDS:
                self.go(ATTRACT, (0.0, 0.5))

        if self.menu is not None:
            self.menu.update(now)
        self.confetti.update(now)
        self._lock_shown = fx.approach(
            self._lock_shown, self.lock / max(0.01, self.settings.face_lock_seconds), dt,
            18.0 if self.lock > 0 else 7.0)
        self._publish(now)
        return detection

    def _update_marker(self, now: float, dt: float) -> None:
        """Glide the drawn head marker toward the head instead of jumping."""
        if self.head is None:
            return
        if self._marker is None or now - self._marker_seen > 0.3:
            self._marker = self.head
        else:
            self._marker = (fx.approach(self._marker[0], self.head[0], dt, 28.0),
                            fx.approach(self._marker[1], self.head[1], dt, 28.0))
        self._marker_seen = now

    def _update_find_player(self, now: float, dt: float) -> None:
        if abs(self._aspect() - self.run.square.config.aspect) > 0.01:
            self.run = self._new_run()          # camera resolution changed under us
        if self.head is not None:
            self._last_seen = now
        if self.inside():
            self.lock += dt
            self._last_inside = now
        elif now - self._last_inside > LOCK_DROPOUT_SECONDS:
            self.lock = 0.0
        if self.lock >= self.settings.face_lock_seconds:
            self.go(COUNTDOWN, (0.0, 0.25))
        elif now - self._last_seen > FIND_TIMEOUT_SECONDS:
            self.go(ATTRACT, (0.0, 0.5))

    def _update_countdown(self, now: float) -> None:
        if self.inside():
            self._away_since = None
        else:
            if self._away_since is None:
                self._away_since = now
            if now - self._away_since > self.run.config.grace_seconds:
                self.toast("LOST YOU - STEP BACK INTO THE SQUARE")
                self.go(FIND_PLAYER, (0.0, 0.25))
                return
        if now - self.phase_started >= self._countdown_total:
            self.go(PLAYING, None)

    def _update_playing(self, now: float) -> None:
        # Stepped every render frame, not every camera frame: the square glides at
        # the display's rate, and it is judged against the newest head we have -
        # exactly what the player sees on screen.
        result = self.run.update(now, self.head)
        self.last_result = result
        if result.life_lost:
            self.loss_flash_at = now
            self._lost_heart = (result.lives, now)
        if result.resumed:
            self._resumed_at = now
        if result.over:
            self._finish_run(now)

    def _finish_run(self, now: float) -> None:
        self.final_score = to_score(self.run.survived(now))
        try:
            self.submit = self.store.submit_score(self.name, self.final_score)
            self.store.announce(self.submit.entry.id)
            self.total_players = self.store.total_players()
            self.leaders = self.store.top(3)
        except sqlite3.Error as exc:
            print(f"Could not save the score: {exc!r}")
            self.submit = None
        self.outcome = self._outcome(self.submit)
        self.go(RESULT, (0.0, 0.5))
        count = {NEW_TOP: 230, PERSONAL_BEST: 160, FIRST_RUN: 120}.get(self.outcome, 0)
        if count:
            colours = ((theme.GOLD, theme.GOLD, theme.MAGENTA, theme.INK, (255, 140, 60))
                       if self.outcome == NEW_TOP else fx.CONFETTI_COLOURS)
            self.confetti.burst(self._screen_size[0] / self._screen_size[1], now + 0.25,
                                count, colours)

    @staticmethod
    def _outcome(submit: Optional[SubmitResult]) -> str:
        if submit is None:
            return NOT_SAVED
        if submit.is_personal_best and submit.rank == 1:
            return NEW_TOP
        if submit.is_new_player:
            return FIRST_RUN
        if submit.is_personal_best:
            return PERSONAL_BEST
        return NOT_BEST

    # ------------------------------------------------------------ live state

    def _live_values(self, now: float) -> tuple:
        name = normalise_name(self.name) or None if self.screen != ATTRACT else None
        lives = self.run.lives if self.screen in (PLAYING, RESULT) else self.run.config.lives
        max_lives = self.run.config.lives
        if self.screen == PLAYING:
            score = to_score(self.run.survived(now))
        elif self.screen == RESULT:
            score = self.final_score
        else:
            score = 0
        if self.screen in (ATTRACT, NAME_ENTRY):
            lives = max_lives = 0
        return (self.screen, name, score, lives, max_lives)

    def _publish(self, now: float, force: bool = False) -> None:
        """Mirror the game into the DB for the leaderboard's NOW PLAYING panel.

        Throttled: a write is a commit, and committing 60 times a second only
        makes the leaderboard's reads wait. Phase changes always go straight out.
        """
        values = self._live_values(now)
        if not force:
            if now - self._live_at < LIVE_INTERVAL:
                return
            if values == self._live_state and now - self._live_at < LIVE_HEARTBEAT:
                return
        try:
            self.store.set_live(*values)
        except sqlite3.Error:
            return
        self._live_state = values
        self._live_at = now

    # ================================================================== draw

    def draw(self, surface: pygame.Surface, detection: Optional[HeadDetection] = None
             ) -> None:
        detection = detection or self.detection
        now = self.clock()
        if self._pending_transition is not None:
            slide, seconds = self._pending_transition
            self._pending_transition = None
            self.transition.begin(surface.copy(), now, slide, seconds)

        self._screen_size = surface.get_size()
        self._layout(surface, detection)
        tint = {NAME_ENTRY: TINT_DIM, RESULT: TINT_RESULT}.get(self.screen)
        self._draw_camera(surface, detection, tint)

        {ATTRACT: self._draw_attract, NAME_ENTRY: self._draw_name_entry,
         FIND_PLAYER: self._draw_find_player, COUNTDOWN: self._draw_countdown,
         PLAYING: self._draw_playing, RESULT: self._draw_result}[self.screen](surface, now)

        if self.confetti.alive:
            self.confetti.draw(surface, now)
        if now - self.loss_flash_at < LOSS_FLASH_SECONDS:
            self._draw_loss_flash(surface, now)
        self._draw_toast(surface, now)
        self.transition.draw(surface, now)
        if self.menu is not None:
            self.menu.draw(surface)

    # --------------------------------------------------------- camera layer

    def _layout(self, surface: pygame.Surface, detection: HeadDetection) -> None:
        """Where the picture lands. Cover the window when the shapes nearly match
        (a sliver cropped is fine - the square keeps a margin), otherwise letterbox
        so the square can never wander somewhere the window cuts off."""
        fw, fh = self._frame_size(detection)
        sw, sh = surface.get_size()
        cover = max(sw / fw, sh / fh)
        contain = min(sw / fw, sh / fh)
        scale = cover if cover / contain <= 1.08 else contain
        view = pygame.Rect(0, 0, max(1, round(fw * scale)), max(1, round(fh * scale)))
        view.center = (sw // 2, sh // 2)
        self._view = view

    def _draw_camera(self, surface: pygame.Surface, detection: HeadDetection,
                     tint: Optional[theme.Color]) -> None:
        view = self._view
        if view.size != surface.get_size():
            theme.backdrop(surface)
        frame = detection.frame
        if frame is None:
            surface.blit(self._placeholder_for(view.size), view.topleft)
            return
        key = (id(frame), detection.timestamp, view.size, tint)
        if key != self._cam_key or self._cam_surface is None:
            h, w = frame.shape[:2]
            raw = pygame.image.frombuffer(np.ascontiguousarray(frame, dtype=np.uint8),
                                          (w, h), "RGB")
            # Into the display's pixel format first: 32-bit scales faster than the
            # camera's 24-bit and then blits ~5x faster every render frame.
            # Scaled once per camera frame, never per render frame.
            if pygame.display.get_surface() is not None:
                raw = raw.convert()
            image = (pygame.transform.smoothscale(raw, view.size)
                     if (w, h) != view.size else raw.copy())
            if tint is not None:
                image.fill(tint, special_flags=pygame.BLEND_RGB_MULT)
            self._cam_surface = image
            self._cam_key = key
        surface.blit(self._cam_surface, view.topleft)

    def _placeholder_for(self, size: Tuple[int, int]) -> pygame.Surface:
        """A dark grid standing in for the camera until frames arrive."""
        if self._placeholder is None or self._placeholder.get_size() != size:
            layer = pygame.Surface(size)
            theme.backdrop(layer)
            step = max(24, size[1] // 12)
            for x in range(step // 2, size[0], step):
                pygame.draw.line(layer, (22, 23, 40), (x, 0), (x, size[1]))
            for y in range(step // 2, size[1], step):
                pygame.draw.line(layer, (22, 23, 40), (0, y), (size[0], y))
            self._placeholder = layer
        return self._placeholder

    def _to_pixels(self, x: float, y: float) -> Tuple[int, int]:
        view = self._view
        return (int(view.left + x * view.width), int(view.top + y * view.height))

    def _square_rect(self, square: Square) -> pygame.Rect:
        left, top, right, bottom = square.bounds()
        x0, y0 = self._to_pixels(left, top)
        x1, y1 = self._to_pixels(right, bottom)
        # Snapped to 4px steps, so the cached soft-edge patches are reused while it
        # shrinks: a new size costs a few ms to build, about once a second.
        rect = pygame.Rect(0, 0, max(8, (x1 - x0) // 4 * 4), max(8, (y1 - y0) // 4 * 4))
        rect.center = self._to_pixels(square.x, square.y)
        return rect

    # ------------------------------------------------------- scene pieces

    def _draw_light(self, surface: pygame.Surface, square: Square, colour: theme.Color,
                    veil: fx.Veil, glow: float = 1.0, line: bool = True) -> pygame.Rect:
        """The spotlight: soft-edged hole in the veil, a glow, and a crisp edge."""
        height = surface.get_height()
        rect = self._square_rect(square)
        radius = max(6, rect.width // 11 // 2 * 2)
        veil.draw(surface, rect, radius, theme.scaled(height, 0.05))
        fx.draw_glow(surface, rect, colour, radius, spread=theme.scaled(height, 0.02),
                     line=2, intensity=glow)
        if line:
            fx.draw_border(surface, rect, colour, radius, max(2, round(height * 0.0045)))
        return rect

    def _draw_tracking_box(self, surface: pygame.Surface, detection: HeadDetection,
                           colour: theme.Color = theme.CYAN) -> None:
        if self.dev or not self.settings.show_tracking or detection.box is None:
            return
        left, top, width, height = detection.box
        x0, y0 = self._to_pixels(left, top)
        x1, y1 = self._to_pixels(left + width, top + height)
        rect = pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        _brackets(surface, rect, colour, max(8, rect.width // 5),
                  max(2, surface.get_height() // 360))

    def _draw_head_marker(self, surface: pygame.Surface, inside: bool) -> None:
        if self.head is None or self._marker is None:
            return
        height = surface.get_height()
        centre = self._to_pixels(*self._marker)
        radius = theme.scaled(height, 0.026)
        width = max(3, round(height * 0.0065))
        colour = theme.GREEN if inside else theme.RED
        _aa_ring(surface, centre, radius + 2, width + 4, theme.BG_SUNK)
        _aa_ring(surface, centre, radius, width, colour)
        pygame.draw.circle(surface, theme.BG_SUNK, centre, max(3, width))
        pygame.draw.circle(surface, colour, centre, max(2, width - 1))

    def _draw_pointer(self, surface: pygame.Surface, target: pygame.Rect, now: float,
                      colour: theme.Color = theme.GOLD) -> None:
        """Arrowheads from the head toward the square, when the head is outside it."""
        if self.head is None or self._marker is None:
            return
        height = surface.get_height()
        hx, hy = self._to_pixels(*self._marker)
        dx, dy = target.centerx - hx, target.centery - hy
        distance = math.hypot(dx, dy)
        radius = theme.scaled(height, 0.026)
        if distance < radius * 3:
            return
        angle = math.atan2(dy, dx)
        size = theme.scaled(height, 0.024)
        room = distance - target.width / 2 - radius * 1.5
        for i in range(2):
            travel = radius * 2.0 + (i * size * 2.0) + (now * 2.2 % 1.0) * size * 0.6
            if travel > max(radius * 2, room):
                break
            centre = (hx + math.cos(angle) * travel, hy + math.sin(angle) * travel)
            _arrowhead(surface, centre, angle, size, colour)

    # ------------------------------------------------------------ attract

    def _draw_attract(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        self._draw_light(surface, self._demo, theme.mix(theme.GOLD, theme.BG, 0.35),
                         self.attract_veil, glow=0.4)
        cx = width // 2

        theme.text(surface, "SEARCHLIGHT", "display", theme.scaled(height, 0.20),
                   theme.GOLD, center=(cx, int(height * 0.285)), shadow=True,
                   max_width=int(width * 0.86))
        theme.text(surface, "KEEP YOUR HEAD IN THE LIGHT.  HOW LONG CAN YOU LAST?", "body",
                   theme.scaled(height, 0.032), theme.INK, center=(cx, int(height * 0.425)),
                   shadow=True, max_width=int(width * 0.86))

        # The call to action: a gold pill that breathes.
        glow = theme.pulse(now, 3.0)
        label_size = theme.scaled(height, 0.058)
        label = "PRESS ENTER TO PLAY"
        label_w = theme.font("display", label_size).size(label)[0]
        pill = pygame.Rect(0, 0, min(int(width * 0.8), label_w + label_size * 2),
                           int(label_size * 1.55))
        pill.center = (cx, int(height * 0.565))
        fx.draw_glow(surface, pill, theme.GOLD, pill.height // 2,
                     spread=theme.scaled(height, 0.02), intensity=0.25 + 0.45 * glow)
        theme.panel(surface, pill, theme.mix(theme.GOLD, (255, 236, 150), glow * 0.5),
                    radius=pill.height // 2)
        theme.text(surface, label, "display", label_size, theme.BG_SUNK,
                   center=(cx, pill.centery + max(1, label_size // 20)),
                   max_width=pill.width - label_size)

        self._draw_podium(surface, int(height * 0.725))
        self._draw_status_chip(surface)
        self._draw_settings_chip(surface)

    def _draw_podium(self, surface: pygame.Surface, top: int) -> None:
        width, height = surface.get_size()
        cx = width // 2
        _chip(surface, "LONGEST IN THE LIGHT", (cx, top + theme.scaled(height, 0.012)),
              theme.scaled(height, 0.019), theme.INK_MUTED, fill=theme.BG_SUNK)
        card_top = top + theme.scaled(height, 0.042)
        card_h = theme.scaled(height, 0.12)
        if not self.leaders:
            box = pygame.Rect(0, card_top, int(width * 0.46), card_h)
            box.centerx = cx
            theme.panel(surface, box, theme.BG_SUNK, radius=card_h // 4, alpha=190,
                        border=theme.LINE)
            theme.text(surface, "NOBODY YET - BE THE FIRST ON THE BOARD", "display",
                       theme.scaled(height, 0.045), theme.INK, center=box.center,
                       max_width=box.width - card_h // 2)
            return
        gap = theme.scaled(height, 0.02)
        card_w = min(int(width * 0.27), int(height * 0.44))
        count = len(self.leaders)
        total = count * card_w + (count - 1) * gap
        left = cx - total // 2
        for i, entry in enumerate(self.leaders):
            box = pygame.Rect(left + i * (card_w + gap), card_top, card_w, card_h)
            colour = theme.PODIUM[i]
            theme.panel(surface, box, theme.BG_SUNK, radius=card_h // 4, alpha=200,
                        border=theme.mix(colour, theme.BG, 0.45))
            medal_r = int(card_h * 0.26)
            medal = (box.left + int(card_h * 0.45), box.centery)
            pygame.draw.circle(surface, colour, medal, medal_r)
            theme.text(surface, str(i + 1), "display", int(medal_r * 1.5), theme.BG_SUNK,
                       center=(medal[0], medal[1] + max(1, medal_r // 12)))
            text_left = medal[0] + medal_r + theme.scaled(height, 0.018)
            room = box.right - text_left - theme.scaled(height, 0.02)
            theme.text(surface, entry.name, "bold", theme.scaled(height, 0.032), theme.INK,
                       midleft=(text_left, box.centery - card_h // 5), max_width=room)
            theme.text(surface, format_score(entry.score), "display",
                       theme.scaled(height, 0.05), colour,
                       midleft=(text_left, box.centery + card_h // 6), max_width=room)

    def _draw_status_chip(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        detection = self.detection
        error = getattr(self.camera, "error", None)
        if error:
            dot, message = theme.RED, f"CAMERA: {error}".upper()
        elif not detection.ok:
            dot, message = theme.GOLD, "STARTING CAMERA..."
        else:
            dot = theme.GREEN
            parts = ["CAMERA LIVE", MODE_LABELS.get(detection.mode, detection.mode).upper()]
            if detection.fps > 0:
                parts.append(f"{detection.fps:.0f} FPS")
            if detection.head is not None:
                parts.append("PLAYER IN VIEW")
            elif detection.searching:
                parts.append("SCANNING")
            message = "  ·  ".join(parts)
        if self.dev:
            message += "  ·  DEV: MOUSE = HEAD"
        margin = theme.scaled(height, 0.03)
        chip_h = theme.scaled(height, 0.046)
        size = theme.scaled(height, 0.02)
        max_text = int(width * 0.55)
        text_w = min(max_text, theme.font("bold", size).size(message)[0])
        rect = pygame.Rect(margin, margin, text_w + int(chip_h * 1.35), chip_h)
        theme.panel(surface, rect, theme.BG_SUNK, radius=chip_h // 2, alpha=200,
                    border=theme.LINE)
        dot_c = (rect.left + chip_h // 2, rect.centery)
        pygame.draw.circle(surface, theme.mix(dot, theme.BG, 0.6), dot_c, chip_h // 4)
        pygame.draw.circle(surface, dot, dot_c, max(3, chip_h // 7))
        theme.text(surface, message, "bold", size, theme.INK_MUTED,
                   midleft=(rect.left + int(chip_h * 0.95), rect.centery + 1),
                   max_width=max_text)

    def _draw_settings_chip(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        margin = theme.scaled(height, 0.03)
        chip_h = theme.scaled(height, 0.046)
        size = theme.scaled(height, 0.02)
        label_w = theme.font("bold", size).size("SETTINGS")[0]
        cap_w = _keycap_width("TAB", chip_h)
        rect = pygame.Rect(0, margin, chip_h + label_w + cap_w + int(chip_h * 0.9), chip_h)
        rect.right = width - margin
        self.gear_rect = rect
        hover = self._gear_hover
        theme.panel(surface, rect, theme.BG_RAISED if hover else theme.BG_SUNK,
                    radius=chip_h // 2, alpha=220 if hover else 200,
                    border=theme.GOLD if hover else theme.LINE)
        gear_c = (rect.left + chip_h // 2 + 2, rect.centery)
        _gear(surface, gear_c, int(chip_h * 0.3), theme.GOLD if hover else theme.INK_MUTED,
              theme.BG_SUNK)
        theme.text(surface, "SETTINGS", "bold", size, theme.INK if hover else theme.INK_MUTED,
                   midleft=(rect.left + chip_h + 2, rect.centery + 1))
        _keycap(surface, "TAB", (rect.right - int(chip_h * 0.3) - cap_w, rect.centery),
                chip_h)

    # --------------------------------------------------------- name entry

    def _draw_name_entry(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        cx = width // 2
        theme.text(surface, "WHO'S PLAYING?", "display", theme.scaled(height, 0.12),
                   theme.GOLD, center=(cx, int(height * 0.20)), shadow=True,
                   max_width=int(width * 0.9))
        theme.text(surface, f"TYPE YOUR NAME  ·  UP TO {MAX_NAME} CHARACTERS", "bold",
                   theme.scaled(height, 0.026), theme.INK_MUTED,
                   center=(cx, int(height * 0.30)), shadow=True, max_width=int(width * 0.9))

        # Arcade-style slots: one per character, so the length limit is visible.
        slot_w = min(int(height * 0.075), int(width * 0.9 / MAX_NAME))
        gap = max(3, slot_w // 8)
        slots_w = MAX_NAME * slot_w + (MAX_NAME - 1) * gap
        shake = 0.0
        since_error = now - self._name_error_at
        if since_error < 0.4:
            shake = math.sin(since_error * 60) * (1 - since_error / 0.4) * height * 0.012
        pad = theme.scaled(height, 0.03)
        panel = pygame.Rect(0, 0, slots_w + pad * 2, int(height * 0.19))
        panel.center = (cx + int(shake), int(height * 0.475))
        theme.panel(surface, panel, theme.BG_SUNK, radius=theme.scaled(height, 0.03),
                    alpha=215, border=theme.mix(theme.GOLD, theme.BG, 0.5), width=2)
        glyph = theme.scaled(height, 0.11)
        blink = int(now * 2.5) % 2 == 0
        left = panel.left + pad
        underline_h = max(3, height // 180)
        for i in range(MAX_NAME):
            slot = pygame.Rect(left + i * (slot_w + gap), panel.top, slot_w, panel.height)
            base_y = panel.bottom - int(panel.height * 0.2)
            current = i == len(self.name)
            filled = i < len(self.name)
            if current:
                hi = pygame.Rect(slot.left, panel.top + pad // 2, slot_w,
                                 base_y - panel.top - pad // 2 + underline_h * 2)
                theme.panel(surface, hi, theme.BG_RAISED, radius=max(4, slot_w // 6),
                            alpha=220)
            bar_colour = (theme.GOLD if filled else
                          (theme.GOLD if current and blink else theme.INK_FAINT))
            pygame.draw.rect(surface, bar_colour,
                             (slot.left + 2, base_y, slot_w - 4, underline_h),
                             border_radius=underline_h)
            if filled and self.name[i] != " ":
                theme.text(surface, self.name[i], "display", glyph, theme.INK,
                           midbottom=(slot.centerx, base_y - underline_h))

        # Returning player? Say so while they type.
        info_y = int(height * 0.645)
        if self.returning is not None:
            message = f"WELCOME BACK — BEST {format_score(self.returning.score)}"
            _chip(surface, message, (cx, info_y), theme.scaled(height, 0.03), theme.GREEN,
                  fill=theme.mix(theme.GREEN, theme.BG_SUNK, 0.82))
        elif normalise_name(self.name):
            _chip(surface, "NEW PLAYER", (cx, info_y), theme.scaled(height, 0.026),
                  theme.CYAN, fill=theme.BG_SUNK)
        elif since_error < 1.5:
            _chip(surface, "TYPE A NAME FIRST", (cx, info_y), theme.scaled(height, 0.026),
                  theme.RED, fill=theme.BG_SUNK)

        _hints(surface, [("ENTER", "CONTINUE"), ("ESC", "BACK"), ("DEL", "DELETE")],
               (cx, int(height * 0.87)), theme.scaled(height, 0.046))

    # -------------------------------------------------------- find player

    def _find_state(self) -> str:
        if self.head is None:
            return "nobody"
        return "locking" if self.inside() else "outside"

    def _draw_find_player(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        cx = width // 2
        state = self._find_state()
        fill = theme.clamp01(self._lock_shown)

        if state == "locking":
            colour = theme.mix(theme.CYAN, theme.GREEN, fill)
        else:
            colour = theme.mix(theme.GOLD, theme.BG, 0.25 * theme.pulse(now, 4.0))
        rect = self._draw_light(surface, self.run.square, colour, self.veil,
                                glow=0.55 + 0.45 * fill)
        # Lock-on brackets close in on the square as the lock fills.
        gap = int((1 - theme.ease_out_cubic(fill)) * theme.scaled(height, 0.045)) + \
            theme.scaled(height, 0.012)
        _brackets(surface, rect.inflate(gap * 2, gap * 2),
                  theme.mix(theme.INK_MUTED, theme.GREEN, fill),
                  max(12, rect.width // 6), max(3, round(height * 0.006)))

        self._draw_tracking_box(surface, self.detection)
        if state == "outside":
            self._draw_pointer(surface, rect, now)
        self._draw_head_marker(surface, state == "locking")

        theme.text(surface, "STEP INTO THE LIGHT", "display", theme.scaled(height, 0.085),
                   theme.GOLD, center=(cx, int(height * 0.085)), shadow=True,
                   max_width=int(width * 0.9))
        theme.text(surface, f"GET READY, {self.name}", "bold", theme.scaled(height, 0.026),
                   theme.INK, center=(cx, int(height * 0.155)), shadow=True,
                   max_width=int(width * 0.8))

        nobody_for = now - self._last_seen
        if state == "nobody":
            title, detail = "NO ONE DETECTED", "FACE THE CAMERA  ·  STEP CLOSER"
            tone = theme.GOLD
            left = FIND_TIMEOUT_SECONDS - nobody_for
            if left < 10:
                detail = f"BACK TO THE START IN {max(0, math.ceil(left))}s"
            elif nobody_for > FIND_HINT_AFTER:
                detail = "TOO FAR? TRY UPPER BODY OR ZOOM IN SETTINGS"
        elif state == "outside":
            title, detail, tone = "MOVE INTO THE SQUARE", "PUT YOUR HEAD IN THE LIGHT", \
                theme.GOLD
        else:
            title, detail, tone = "HOLD STILL", "LOCKING ON...", theme.GREEN
        self._status_panel(surface, title, detail, tone, fill,
                           theme.GREEN if state == "locking" else theme.GOLD, state)
        _hints(surface, [("ESC", "BACK")],
               (theme.scaled(height, 0.03), height - theme.scaled(height, 0.045)),
               theme.scaled(height, 0.04), align="left")

    def _status_panel(self, surface: pygame.Surface, title: str, detail: str,
                      tone: theme.Color, fill: float, ring_colour: theme.Color,
                      state: str) -> None:
        """Bottom-centre guidance: a lock ring with a person icon, and two lines."""
        width, height = surface.get_size()
        panel_h = theme.scaled(height, 0.13)
        title_size = theme.scaled(height, 0.056)
        detail_size = theme.scaled(height, 0.024)
        text_w = max(theme.font("display", title_size).size(title)[0],
                     theme.font("bold", detail_size).size(detail)[0])
        panel_w = min(int(width * 0.92), text_w + int(panel_h * 1.55))
        panel = pygame.Rect(0, 0, panel_w, panel_h)
        panel.midbottom = (width // 2, height - theme.scaled(height, 0.035))
        theme.panel(surface, panel, theme.BG_SUNK, radius=panel_h // 2, alpha=215,
                    border=theme.mix(tone, theme.BG, 0.55))
        ring_r = int(panel_h * 0.36)
        centre = (panel.left + panel_h // 2 + 4, panel.centery)
        theme.ring(surface, centre, ring_r, max(4, ring_r // 5), fill, ring_colour,
                   track=theme.LINE)
        _person(surface, centre, int(ring_r * 0.62),
                ring_colour if state != "nobody" else theme.INK_FAINT)
        text_left = panel.left + panel_h + 4
        room = panel.right - text_left - panel_h // 3
        theme.text(surface, title, "display", title_size, tone,
                   midleft=(text_left, panel.centery - panel_h // 7), max_width=room)
        theme.text(surface, detail, "bold", detail_size, theme.INK_MUTED,
                   midleft=(text_left, panel.centery + panel_h // 4), max_width=room)

    # ---------------------------------------------------------- countdown

    def _draw_countdown(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        cx = width // 2
        elapsed = now - self.phase_started
        remaining = max(0.0, self._countdown_total - elapsed)
        inside = self.inside()
        beat = elapsed % 1.0
        colour = theme.GREEN if inside else theme.RED
        rect = self._draw_light(surface, self.run.square, colour, self.veil,
                                glow=1.0 - 0.6 * beat)
        _brackets(surface, rect.inflate(theme.scaled(height, 0.024),
                                        theme.scaled(height, 0.024)),
                  colour, max(12, rect.width // 6), max(3, round(height * 0.006)))
        self._draw_tracking_box(surface, self.detection)
        self._draw_head_marker(surface, inside)

        theme.text(surface, "GET READY", "display", theme.scaled(height, 0.085), theme.INK,
                   center=(cx, int(height * 0.085)), shadow=True)
        theme.text(surface, "STAY IN THE SQUARE" if inside else "BACK IN THE SQUARE!",
                   "bold", theme.scaled(height, 0.026),
                   theme.INK if inside else theme.RED, center=(cx, int(height * 0.155)),
                   shadow=True)

        number = max(1, math.ceil(remaining))
        scale = 1.0 + 0.6 * (1 - theme.ease_out_back(beat / 0.45))
        alpha = int(255 * (1 - theme.clamp01((beat - 0.7) / 0.3) * 0.75))
        _pop_text(surface, str(number), "display", theme.scaled(height, 0.22), theme.GOLD,
                  (cx, int(height * 0.855)), scale, alpha)

    # ------------------------------------------------------------ playing

    def _square_colour(self, now: float) -> theme.Color:
        result = self.last_result
        if result is not None and result.waiting:
            return theme.GOLD if int(now * 8) % 2 == 0 else theme.INK_FAINT
        if result is not None and result.respawning:
            return theme.GREEN if self.inside() else theme.RED
        if self.inside():
            return theme.GREEN
        danger = result.danger if result is not None else 0.0
        return theme.mix(theme.GOLD, theme.RED, danger)

    def _draw_playing(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        cx = width // 2
        result = self.last_result
        inside = self.inside()
        danger = result.danger if result is not None else 0.0
        respawning = bool(result and result.respawning)

        colour = self._square_colour(now)
        rect = self._draw_light(surface, self.run.square, colour, self.veil,
                                glow=0.7 + 0.3 * danger)
        if respawning:
            # The same frame as the start countdown: this square is waiting for you.
            _brackets(surface, rect.inflate(theme.scaled(height, 0.024),
                                            theme.scaled(height, 0.024)),
                      colour, max(12, rect.width // 6), max(3, round(height * 0.006)))
        fx.draw_vignette(surface, theme.RED, danger * 0.85)
        self._draw_tracking_box(surface, self.detection, theme.mix(theme.CYAN, theme.BG, 0.4))
        if not inside:
            self._draw_pointer(surface, rect, now, theme.mix(theme.GOLD, theme.RED, danger))
        self._draw_head_marker(surface, inside)

        # ---- HUD
        survived = self.run.survived(now)
        best = self.player_best.score if self.player_best else None
        beating = best is not None and to_score(survived) > best
        self._draw_timer(surface, survived, theme.GREEN if beating else theme.GOLD)
        self._draw_hearts(surface, now)
        self._draw_player_tag(surface, best, beating)

        since_go = now - max(self.phase_started, self._resumed_at)
        if since_go < GO_SECONDS:
            t = since_go / GO_SECONDS
            _pop_text(surface, "GO!", "display", theme.scaled(height, 0.24), theme.GREEN,
                      (cx, self._alert_y(height)), 1.0 + 0.5 * (1 - theme.ease_out_back(
                          min(1.0, t / 0.4))), int(255 * (1 - theme.clamp01((t - 0.6) / 0.4))))
        elif respawning and now - self.loss_flash_at >= LOSS_FLASH_SECONDS:
            assert result is not None
            if result.waiting:
                self._bottom_alert(surface, "PAUSED - GET BACK IN THE SQUARE", theme.GOLD)
            else:
                self._draw_resume_countdown(surface, result.resume_in, inside)
        elif danger > 0:
            self._bottom_alert(surface, "GET BACK IN!", theme.RED, danger)

    def _draw_resume_countdown(self, surface: pygame.Surface, remaining: float,
                               inside: bool) -> None:
        """The short count back into play once the player is in the square again."""
        width, height = surface.get_size()
        cx, y = width // 2, self._alert_y(height)
        beat = (-remaining) % 1.0
        scale = 1.0 + 0.6 * (1 - theme.ease_out_back(beat / 0.45))
        alpha = int(255 * (1 - theme.clamp01((beat - 0.7) / 0.3) * 0.75))
        _pop_text(surface, str(max(1, math.ceil(remaining))), "display",
                  theme.scaled(height, 0.16), theme.GOLD, (cx, y), scale, alpha)
        theme.text(surface, "STAY IN THE SQUARE" if inside else "BACK IN THE SQUARE!",
                   "bold", theme.scaled(height, 0.026), theme.INK if inside else theme.RED,
                   center=(cx, y + theme.scaled(height, 0.1)), shadow=True)

    def _alert_y(self, height: int) -> int:
        """Where big messages go: whichever of the top or bottom band the square is
        not in, so a warning never covers the thing the player has to watch."""
        if self.run.square.y > 0.5:
            return int(height * 0.235)
        return int(height * 0.845)

    def _bottom_alert(self, surface: pygame.Surface, message: str, colour: theme.Color,
                      fill: Optional[float] = None) -> None:
        width, height = surface.get_size()
        cx = width // 2
        y = self._alert_y(height)
        theme.text(surface, message, "display", theme.scaled(height, 0.075), colour,
                   center=(cx, y), shadow=True, max_width=int(width * 0.9))
        if fill is None:
            return
        bar = pygame.Rect(0, 0, int(width * 0.28), max(8, theme.scaled(height, 0.016)))
        bar.center = (cx, y + theme.scaled(height, 0.068))
        theme.meter(surface, bar.inflate(4, 4), 0, theme.BG_SUNK, track=theme.BG_SUNK)
        theme.meter(surface, bar, fill, colour, track=theme.LINE)

    @staticmethod
    def _hud_mid(height: int) -> int:
        """The shared centre line of the timer, hearts and player tag."""
        return int(height * 0.085)

    def _draw_timer(self, surface: pygame.Surface, seconds: float,
                    colour: theme.Color) -> None:
        """The clock, with the decimal point pinned so the digits never jitter."""
        width, height = surface.get_size()
        size = theme.scaled(height, 0.12)
        glyphs = theme.font("display", size)
        whole, tenth = divmod(int(seconds * 10), 10)
        left_text, right_text = f"{whole}", f".{tenth}"
        small = theme.scaled(height, 0.05)
        # Laid out for the digits in play (at least two), so nothing moves until 100s.
        int_w = glyphs.size("0" * max(2, len(left_text)))[0]
        tail_w = glyphs.size(".0")[0]
        suffix_w = theme.font("display", small).size("s")[0] + max(3, size // 30)
        total = int_w + tail_w + suffix_w
        pill = pygame.Rect(0, 0, total + size // 2, int(size * 1.02))
        pill.center = (width // 2, self._hud_mid(height))
        theme.panel(surface, pill, theme.BG_SUNK, radius=pill.height // 2, alpha=190,
                    border=theme.LINE)
        pivot = pill.centerx - total // 2 + int_w
        middle = pill.centery + max(1, size // 24)
        theme.text(surface, left_text, "display", size, colour,
                   midright=(pivot, middle), shadow=True)
        tail = theme.text(surface, right_text, "display", size, colour,
                          midleft=(pivot, middle), shadow=True)
        theme.text(surface, "s", "display", small, theme.mix(colour, theme.INK_MUTED, 0.5),
                   midleft=(tail.right + max(3, size // 30), tail.centery + size // 5))

    def _draw_hearts(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        total = self.run.config.lives
        lives = self.run.lives
        size = theme.scaled(height, 0.042)
        step = int(size * 1.3)
        margin = theme.scaled(height, 0.03)
        chip_h = theme.scaled(height, 0.08)
        rect = pygame.Rect(margin, 0, step * total + chip_h // 2, chip_h)
        rect.centery = self._hud_mid(height)
        theme.panel(surface, rect, theme.BG_SUNK, radius=chip_h // 2, alpha=190,
                    border=theme.LINE)
        for i in range(total):
            centre = (rect.left + chip_h // 4 + step * i + step // 2,
                      rect.centery + size // 10)
            if i < lives:
                theme.draw_heart(surface, centre, size, theme.MAGENTA)
            else:
                theme.draw_heart(surface, centre, size, (52, 52, 74))
        if self._lost_heart is not None:
            index, when = self._lost_heart
            t = (now - when) / 0.7
            if 0 <= t < 1 and index < total:
                centre = (rect.left + chip_h // 4 + step * index + step // 2,
                          rect.centery + size // 10)
                grown = int(size * (1 + 1.2 * theme.ease_out_cubic(t)))
                layer = pygame.Surface((grown * 2, grown * 2), pygame.SRCALPHA)
                theme.draw_heart(layer, (grown, grown), grown, theme.MAGENTA)
                layer.set_alpha(int(255 * (1 - t)))
                surface.blit(layer, (centre[0] - grown, centre[1] - grown))

    def _draw_player_tag(self, surface: pygame.Surface, best: Optional[int],
                         beating: bool) -> None:
        width, height = surface.get_size()
        margin = theme.scaled(height, 0.03)
        name_size = theme.scaled(height, 0.036)
        line_size = theme.scaled(height, 0.019)
        if beating:
            line, tone = "NEW PERSONAL BEST!", theme.GREEN
        elif best is not None:
            line, tone = f"YOUR BEST {format_score(best)}", theme.INK_MUTED
        else:
            line, tone = "FIRST RUN", theme.CYAN
        text_w = max(theme.font("display", name_size).size(self.name)[0],
                     theme.font("bold", line_size).size(line)[0])
        chip_h = theme.scaled(height, 0.08)
        rect = pygame.Rect(0, 0, min(int(width * 0.3), text_w + chip_h), chip_h)
        rect.right = width - margin
        rect.centery = self._hud_mid(height)
        theme.panel(surface, rect, theme.BG_SUNK, radius=chip_h // 2, alpha=190,
                    border=theme.mix(tone, theme.BG, 0.5) if beating else theme.LINE)
        room = rect.width - chip_h // 2
        theme.text(surface, self.name, "display", name_size, theme.INK,
                   midright=(rect.right - chip_h // 3, rect.top + int(chip_h * 0.38)),
                   max_width=room)
        theme.text(surface, line, "bold", line_size, tone,
                   midright=(rect.right - chip_h // 3, rect.top + int(chip_h * 0.74)),
                   max_width=room)

    def _draw_loss_flash(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        t = (now - self.loss_flash_at) / LOSS_FLASH_SECONDS
        if self._flash_layer is None or self._flash_layer.get_size() != (width, height):
            self._flash_layer = pygame.Surface((width, height))
            self._flash_layer.fill(theme.RED)
        self._flash_layer.set_alpha(int(110 * (1 - t)))
        surface.blit(self._flash_layer, (0, 0))
        pygame.draw.rect(surface, theme.RED, surface.get_rect(),
                         width=max(6, int(height * 0.012)))
        out = self.run.lives <= 0
        _pop_text(surface, "OUT!" if out else "-1 LIFE", "display",
                  theme.scaled(height, 0.13), theme.RED if not out else theme.INK,
                  (width // 2, self._alert_y(height)),
                  1.0 + 0.35 * (1 - theme.ease_out_back(min(1.0, t / 0.35))),
                  int(255 * (1 - theme.clamp01((t - 0.6) / 0.4))))

    # ------------------------------------------------------------- result

    def _draw_result(self, surface: pygame.Surface, now: float) -> None:
        width, height = surface.get_size()
        cx = width // 2
        elapsed = now - self.phase_started

        headlines = {
            NEW_TOP: ("NEW HIGH SCORE!", theme.mix(theme.GOLD, theme.MAGENTA,
                                                   theme.pulse(now, 6.0))),
            PERSONAL_BEST: ("NEW PERSONAL BEST!", theme.GREEN),
            FIRST_RUN: ("YOU'RE ON THE BOARD!", theme.CYAN),
            NOT_BEST: ("OUT OF LIVES", theme.INK),
            NOT_SAVED: ("OUT OF LIVES", theme.INK),
        }
        headline, tone = headlines[self.outcome]

        # Staggered entrances: name, headline, time, then the chips.
        def appear(delay: float) -> Tuple[float, int]:
            t = theme.ease_out_cubic((elapsed - delay) / 0.45)
            return (1 - t) * height * 0.03, int(255 * t)

        dy, alpha = appear(0.0)
        theme.text(surface, self.name, "bold", theme.scaled(height, 0.042), theme.INK,
                   center=(cx, int(height * 0.12 + dy)), shadow=True, alpha=alpha)
        dy, alpha = appear(0.1)
        theme.text(surface, headline, "display", theme.scaled(height, 0.10), tone,
                   center=(cx, int(height * 0.225 + dy)), shadow=True, alpha=alpha,
                   max_width=int(width * 0.9))
        pop = 1.0 + 0.25 * (1 - theme.ease_out_back(theme.clamp01((elapsed - 0.15) / 0.5)))
        _pop_text(surface, f"{self.final_score / 10:.1f}", "display",
                  theme.scaled(height, 0.30), theme.GOLD, (cx, int(height * 0.465)), pop,
                  int(255 * theme.clamp01((elapsed - 0.15) / 0.25)), suffix="s")
        dy, alpha = appear(0.3)
        theme.text(surface, "IN THE LIGHT", "bold", theme.scaled(height, 0.03),
                   theme.INK_MUTED, center=(cx, int(height * 0.635 + dy)), shadow=True,
                   alpha=alpha)

        dy, alpha = appear(0.45)
        if alpha > 0:
            chips: List[Tuple[str, theme.Color]] = []
            submit = self.submit
            if submit is None:
                chips.append(("SCORE NOT SAVED", theme.RED))
            else:
                chips.append((f"RANK #{submit.rank} OF {self.total_players}",
                              theme.PODIUM[submit.rank - 1] if submit.rank <= 3
                              else theme.INK))
                if self.outcome == NOT_BEST and submit.previous_best is not None:
                    chips.append((f"YOUR BEST {format_score(submit.previous_best)}",
                                  theme.INK_MUTED))
                elif (self.outcome in (PERSONAL_BEST, NEW_TOP)
                      and submit.previous_best is not None):
                    gain = self.final_score - submit.previous_best
                    chips.append((f"+{format_score(gain)} ON YOUR BEST", theme.GREEN))
            _chip_row(surface, chips, (cx, int(height * 0.745 + dy)),
                      theme.scaled(height, 0.036), alpha)

        # Auto-return: a draining bar, and the prompt once keys are accepted.
        if elapsed >= RESULT_KEY_DELAY:
            theme.text(surface, "PRESS ANY KEY", "bold", theme.scaled(height, 0.024),
                       theme.mix(theme.INK_FAINT, theme.INK, theme.pulse(now, 3.0)),
                       center=(cx, int(height * 0.875)), shadow=True)
        bar = pygame.Rect(0, 0, int(width * 0.2), max(4, height // 150))
        bar.center = (cx, int(height * 0.915))
        theme.meter(surface, bar, 1 - fx.progress(self.phase_started, RESULT_SECONDS, now),
                    theme.INK_MUTED, track=theme.LINE)

    # -------------------------------------------------------------- toast

    def _draw_toast(self, surface: pygame.Surface, now: float) -> None:
        if self._toast is None:
            return
        message, at = self._toast
        age = now - at
        if age > TOAST_SECONDS:
            self._toast = None
            return
        width, height = surface.get_size()
        alpha = int(255 * min(theme.clamp01(age / 0.15),
                              theme.clamp01((TOAST_SECONDS - age) / 0.3)))
        _chip(surface, message, (width // 2, int(height * 0.70)),
              theme.scaled(height, 0.03), theme.GOLD, fill=theme.BG_SUNK, alpha=alpha)


# ===================================================================== widgets


def _aa_ring(surface: pygame.Surface, centre: Tuple[int, int], radius: int, width: int,
             colour: theme.Color) -> None:
    """A thick ring with anti-aliased edges (draw.circle alone is jagged)."""
    x, y = int(centre[0]), int(centre[1])
    pygame.draw.circle(surface, colour, (x, y), radius, width)
    try:
        pygame.gfxdraw.aacircle(surface, x, y, radius, colour)
        if radius - width + 1 > 0:
            pygame.gfxdraw.aacircle(surface, x, y, radius - width + 1, colour)
    except (OverflowError, pygame.error):
        pass


def _brackets(surface: pygame.Surface, rect: pygame.Rect, colour: theme.Color,
              length: int, width: int) -> None:
    """Four corner brackets - a targeting reticle rather than a closed box."""
    length = min(length, rect.width // 2, rect.height // 2)
    for (x, y), (sx, sy) in (((rect.left, rect.top), (1, 1)),
                             ((rect.right - 1, rect.top), (-1, 1)),
                             ((rect.left, rect.bottom - 1), (1, -1)),
                             ((rect.right - 1, rect.bottom - 1), (-1, -1))):
        pygame.draw.line(surface, colour, (x, y), (x + sx * length, y), width)
        pygame.draw.line(surface, colour, (x, y), (x, y + sy * length), width)
        pygame.draw.circle(surface, colour, (x, y), max(1, width // 2))


def _arrowhead(surface: pygame.Surface, centre: Tuple[float, float], angle: float,
               size: int, colour: theme.Color) -> None:
    c, s = math.cos(angle), math.sin(angle)
    shape = ((size, 0), (-size * 0.6, -size * 0.8), (-size * 0.2, 0), (-size * 0.6, size * 0.8))
    points = [(centre[0] + x * c - y * s, centre[1] + x * s + y * c) for x, y in shape]
    pygame.draw.polygon(surface, theme.BG_SUNK, points, max(3, size // 4))
    pygame.draw.polygon(surface, colour, points)


def _gear(surface: pygame.Surface, centre: Tuple[int, int], radius: int,
          colour: theme.Color, hole: theme.Color) -> None:
    """A cog: a disc, eight teeth and a hole. No icon font is trusted on a kiosk."""
    cx, cy = centre
    teeth = 8
    points = []
    for i in range(teeth * 4):
        angle = 2 * math.pi * i / (teeth * 4)
        r = radius if (i % 4) in (0, 1) else radius * 0.74
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    pygame.draw.polygon(surface, colour, points)
    pygame.draw.circle(surface, colour, centre, int(radius * 0.78))
    pygame.draw.circle(surface, hole, centre, max(2, int(radius * 0.34)))


def _person(surface: pygame.Surface, centre: Tuple[int, int], size: int,
            colour: theme.Color) -> None:
    """Head and shoulders - the thing the camera is looking for."""
    cx, cy = centre
    head_r = max(3, int(size * 0.36))
    pygame.draw.circle(surface, colour, (cx, cy - int(size * 0.28)), head_r)
    shoulders = pygame.Rect(0, 0, int(size * 1.5), int(size * 1.1))
    shoulders.midtop = (cx, cy + int(size * 0.2))
    clip = surface.get_clip()
    surface.set_clip(pygame.Rect(shoulders.left, shoulders.top, shoulders.width,
                                 int(size * 0.62)).clip(clip) if clip else None)
    pygame.draw.ellipse(surface, colour, shoulders)
    surface.set_clip(clip)


def _keycap_width(label: str, height: int) -> int:
    size = max(10, int(height * 0.42))
    return max(int(height * 0.9), theme.font("mono", size).size(label)[0] + int(height * 0.5))


def _keycap(surface: pygame.Surface, label: str, midleft: Tuple[int, int],
            height: int) -> pygame.Rect:
    """A little key, so hints read as "press this" rather than as prose."""
    cap_h = int(height * 0.72)
    rect = pygame.Rect(0, 0, _keycap_width(label, height), cap_h)
    rect.midleft = midleft
    lip = max(2, cap_h // 10)
    theme.panel(surface, rect.move(0, lip), theme.BG_SUNK, radius=cap_h // 4)
    theme.panel(surface, rect, theme.BG_HOVER, radius=cap_h // 4, border=theme.LINE,
                width=1)
    size = max(10, int(height * 0.42))
    font_kind = "mono" if label.isascii() else "bold"
    theme.text(surface, label, font_kind, size, theme.INK, center=rect.center)
    return rect


def _hints(surface: pygame.Surface, items: List[Tuple[str, str]], anchor: Tuple[int, int],
           height: int, align: str = "center") -> None:
    """A row of [KEY] ACTION pairs, centred on `anchor` (or starting at it)."""
    left_aligned = align == "left"
    size = max(10, int(height * 0.4))
    glyphs = theme.font("bold", size)
    gap = int(height * 0.25)
    spacer = int(height * 0.8)
    widths = [_keycap_width(k, height) + gap + glyphs.size(a)[0] for k, a in items]
    total = sum(widths) + spacer * (len(items) - 1)
    x = anchor[0] if left_aligned else anchor[0] - total // 2
    backing = pygame.Rect(x - int(height * 0.35), 0, total + int(height * 0.7), height)
    backing.centery = anchor[1]
    theme.panel(surface, backing, theme.BG_SUNK, radius=height // 2, alpha=170)
    for (key, action), w in zip(items, widths):
        cap = _keycap(surface, key, (x, anchor[1]), height)
        theme.text(surface, action, "bold", size, theme.INK_MUTED,
                   midleft=(cap.right + gap, anchor[1] + 1))
        x += w + spacer


def _chip(surface: pygame.Surface, message: str, center: Tuple[int, int], size: int,
          colour: theme.Color, fill: theme.Color = theme.BG_SUNK, alpha: int = 255) -> None:
    """A pill of text with a coloured edge - for statuses and notices."""
    glyphs = theme.font("bold", size)
    w = glyphs.size(message)[0]
    h = int(size * 1.9)
    rect = pygame.Rect(0, 0, w + h, h)
    rect.center = center
    if alpha < 255:
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        theme.panel(layer, layer.get_rect(), fill, radius=h // 2, alpha=225,
                    border=colour)
        theme.text(layer, message, "bold", size, colour,
                   center=(rect.width // 2, rect.height // 2 + 1))
        layer.set_alpha(alpha)
        surface.blit(layer, rect.topleft)
        return
    theme.panel(surface, rect, fill, radius=h // 2, alpha=225, border=colour)
    theme.text(surface, message, "bold", size, colour, center=(center[0], center[1] + 1))


def _chip_row(surface: pygame.Surface, chips: List[Tuple[str, theme.Color]],
              center: Tuple[int, int], size: int, alpha: int = 255) -> None:
    glyphs = theme.font("display", size)
    h = int(size * 1.6)
    widths = [glyphs.size(text)[0] + h for text, _ in chips]
    gap = size // 2
    total = sum(widths) + gap * (len(chips) - 1)
    x = center[0] - total // 2
    layer = pygame.Surface((total, h), pygame.SRCALPHA)
    lx = 0
    for (text, colour), w in zip(chips, widths):
        rect = pygame.Rect(lx, 0, w, h)
        theme.panel(layer, rect, theme.BG_SUNK, radius=h // 2, alpha=225,
                    border=theme.mix(colour, theme.BG, 0.3))
        theme.text(layer, text, "display", size, colour,
                   center=(rect.centerx, rect.centery + max(1, size // 16)))
        lx += w + gap
    if alpha < 255:
        layer.set_alpha(alpha)
    surface.blit(layer, (x, center[1] - h // 2))


def _pop_text(surface: pygame.Surface, value: str, kind: str, size: int,
              colour: theme.Color, center: Tuple[int, int], scale: float = 1.0,
              alpha: int = 255, suffix: str = "") -> None:
    """Big type that can scale and fade - countdown digits, GO!, the final time.

    `suffix` is set small and baseline-aligned after the value ("42.3" + "s"), and
    the pair is centred on the value alone so the number sits dead centre.
    """
    if alpha <= 0:
        return
    glyphs = theme.font(kind, size)
    face = glyphs.render(value, True, colour)
    shadow = glyphs.render(value, True, theme.BG_SUNK)
    offset = max(3, size // 22)
    tail = tail_shadow = None
    if suffix:
        small = theme.font(kind, max(10, int(size * 0.4)))
        tail = small.render(suffix, True, theme.mix(colour, theme.INK_MUTED, 0.45))
        tail_shadow = small.render(suffix, True, theme.BG_SUNK)
    extra = tail.get_width() + size // 20 if tail is not None else 0
    body_w = face.get_width() + offset
    layer = pygame.Surface((body_w + 2 * extra, face.get_height() + offset),
                           pygame.SRCALPHA)
    layer.blit(shadow, (extra + offset, offset))
    layer.blit(face, (extra, 0))
    if tail is not None and tail_shadow is not None:
        # Baselines line up because both faces share ascent ratios.
        base = glyphs.get_ascent()
        top = base - theme.font(kind, max(10, int(size * 0.4))).get_ascent()
        x = extra + face.get_width() + size // 20
        layer.blit(tail_shadow, (x + max(2, offset // 2), top + max(2, offset // 2)))
        layer.blit(tail, (x, top))
    if abs(scale - 1.0) > 0.01:
        layer = pygame.transform.smoothscale(
            layer, (max(1, int(layer.get_width() * scale)),
                    max(1, int(layer.get_height() * scale))))
    if alpha < 255:
        layer.set_alpha(alpha)
    surface.blit(layer, layer.get_rect(center=center))


def recording_path(folder: Path, name: str, when: datetime) -> Path:
    """`folder/2026-09-28_14-32-05_ALEX.mp4`: sorts by time, says who played."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "PLAYER"
    return folder / f"{when:%Y-%m-%d_%H-%M-%S}_{safe}.mp4"


# ======================================================================== main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Searchlight - player window")
    parser.add_argument("--display", type=int, default=0, help="monitor index")
    parser.add_argument("--db", default=str(store_mod.DEFAULT_DB))
    parser.add_argument("--fullscreen", action="store_true",
                        help="take over the display (default is a window)")
    parser.add_argument("--pos", default=None,
                        help="window position as X,Y (windowed mode only)")
    parser.add_argument("--size", default="1280x720", help="window size")
    parser.add_argument("--dev", action="store_true",
                        help="the mouse stands in for your head")
    parser.add_argument("--recordings", default=str(DEFAULT_RECORDINGS),
                        help="folder for run videos (when Record runs is on)")
    args = parser.parse_args(argv)

    if args.pos and not args.fullscreen:
        os.environ["SDL_VIDEO_WINDOW_POS"] = args.pos

    pygame.init()
    pygame.display.set_caption("Searchlight")
    width, height = (int(v) for v in args.size.lower().split("x"))
    window = Window((width, height), args.fullscreen, args.display)

    store = Store(Path(args.db))
    # Imported here so tests and the render script never load OpenCV/MediaPipe.
    from .camera import CameraThread
    camera = CameraThread(store.load_settings().camera_options()).start()
    game = Game(store, camera, dev=args.dev, recordings=Path(args.recordings))
    game.go(ATTRACT, (0.0, 0.6))

    clock = pygame.time.Clock()
    try:
        while game.running:
            for event in pygame.event.get():
                if not window.handle(event):
                    game.handle(event)
            detection = game.update()
            game.draw(window.surface, detection)
            pygame.display.flip()
            clock.tick(60)
    finally:
        game.close()                # first: it decides whether a run's video is kept
        camera.close()
        store.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
