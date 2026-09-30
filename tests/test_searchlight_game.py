"""Screen flow for the Searchlight player window, headless.

A stub camera stands in for `CameraThread` and a fake clock for time.monotonic,
so every timing rule (lock-on, countdown, grace, timeouts) is stepped through
simulated seconds instead of slept through.
"""

from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from searchlight import game as g
from searchlight.game import (ATTRACT, COUNTDOWN, FIND_PLAYER, NAME_ENTRY, PLAYING, RESULT,
                              Game)
from searchlight.settings import Settings
from searchlight.store import Store
from searchlight.tracking.base import MODE_UPPER_BODY, HeadDetection

pygame.init()

FOLLOW = "follow"        # step() option: keep the head on the square's centre
OUTSIDE = (0.03, 0.05)   # far from wherever the square is parked


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class StubCamera:
    def __init__(self):
        self.detection = HeadDetection(ok=True)
        self.error = None
        self.configured = []
        self.resets = 0
        self.recording = None               # path being recorded
        self.kept, self.discarded = [], []

    def latest(self):
        return self.detection

    def configure(self, options):
        self.configured.append(options)

    def reset_tracking(self):
        self.resets += 1

    def start_recording(self, path):
        assert self.recording is None, "a recording is already running"
        self.recording = path

    def stop_recording(self, keep=True):
        path, self.recording = self.recording, None
        (self.kept if keep else self.discarded).append(path)
        return path if keep else None

    def close(self):
        pass


class FakeMenu:
    """Stands in for settings_menu.SettingsMenu: records what it is given."""

    instances = []

    def __init__(self, settings, on_save, on_close):
        self.settings, self.on_save, self.on_close = settings, on_save, on_close
        self.events = []
        self.drawn = 0
        FakeMenu.instances.append(self)

    def handle(self, event):
        self.events.append(event)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.on_close()
        return True

    def update(self, now):
        pass

    def draw(self, surface):
        self.drawn += 1


TEST_SETTINGS = Settings().with_changes(lives=3, grace_seconds=0.1, respawn_seconds=0.5,
                                        face_lock_seconds=1.0, countdown_seconds=3)


def make_rig(tmp_path, dev=False, menu_factory=FakeMenu, settings=TEST_SETTINGS):
    store = Store(tmp_path / "g.db")
    store.save_settings(settings)
    clock = FakeClock()
    camera = StubCamera()
    game = Game(store, camera, dev=dev, seed=0, clock=clock, menu_factory=menu_factory,
                recordings=tmp_path / "recordings")
    return SimpleNamespace(store=store, clock=clock, camera=camera, game=game)


@pytest.fixture
def rig(tmp_path):
    r = make_rig(tmp_path)
    yield r
    r.store.close()


def step(rig, seconds, head=None, fps=30):
    """Advance simulated time, reporting `head` (or following the square)."""
    for _ in range(int(round(seconds * fps))):
        rig.clock.t += 1.0 / fps
        if head == FOLLOW:
            square = rig.game.run.square
            position = (square.x, square.y)
        else:
            position = head
        rig.camera.detection = HeadDetection(head=position, ok=True, timestamp=rig.clock.t)
        rig.game.update()


def press(game, key, unicode="", mod=0):
    game.handle(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode, mod=mod))


def type_name(game, name):
    for char in name:
        key = pygame.K_SPACE if char == " " else pygame.key.key_code(char.lower())
        press(game, key, char)


def to_find_player(rig, name="JO"):
    press(rig.game, pygame.K_RETURN)
    type_name(rig.game, name)
    press(rig.game, pygame.K_RETURN)
    assert rig.game.screen == FIND_PLAYER


def to_playing(rig, name="JO"):
    to_find_player(rig, name)
    step(rig, 1.2, FOLLOW)
    assert rig.game.screen == COUNTDOWN
    step(rig, 3.1, FOLLOW)
    assert rig.game.screen == PLAYING


# ------------------------------------------------------------------ flow


def test_name_entry_leads_to_find_player(rig):
    game = rig.game
    press(game, pygame.K_SPACE, " ")
    assert game.screen == NAME_ENTRY
    type_name(game, "jo")
    press(game, pygame.K_RETURN)
    assert game.screen == FIND_PLAYER
    assert game.name == "JO"
    assert rig.camera.resets == 1          # a new player: tracking starts fresh


def test_blank_name_does_not_continue(rig):
    press(rig.game, pygame.K_RETURN)
    type_name(rig.game, " ")
    press(rig.game, pygame.K_RETURN)
    assert rig.game.screen == NAME_ENTRY
    press(rig.game, pygame.K_ESCAPE)
    assert rig.game.screen == ATTRACT


def test_name_entry_recognises_a_returning_player(rig):
    rig.store.submit_score("JO", 234)
    press(rig.game, pygame.K_RETURN)
    type_name(rig.game, "jo")
    assert rig.game.returning is not None and rig.game.returning.score == 234
    press(rig.game, pygame.K_BACKSPACE)
    assert rig.game.returning is None


def test_modifier_keys_do_not_start_a_game(rig):
    press(rig.game, pygame.K_LMETA)
    press(rig.game, pygame.K_LSHIFT)
    assert rig.game.screen == ATTRACT


def test_escape_on_attract_and_cmd_q_anywhere_quit(tmp_path):
    for setup in (lambda r: None, lambda r: to_find_player(r)):
        r = make_rig(tmp_path / str(id(setup)))
        setup(r)
        press(r.game, pygame.K_q, "q", mod=pygame.KMOD_META)
        assert not r.game.running
        r.store.close()
    r = make_rig(tmp_path / "esc")
    press(r.game, pygame.K_ESCAPE)
    assert not r.game.running
    r.store.close()


# ---------------------------------------------------------- find player


def test_no_countdown_without_a_detected_head(rig):
    to_find_player(rig)
    step(rig, 25.0, None)
    assert rig.game.screen == FIND_PLAYER
    # Someone in view but never in the square: waits as long as they stay.
    step(rig, 60.0, OUTSIDE)
    assert rig.game.screen == FIND_PLAYER


def test_lock_needs_the_head_in_the_square_for_the_lock_time(rig):
    to_find_player(rig)
    step(rig, 0.8, FOLLOW)
    assert rig.game.screen == FIND_PLAYER
    step(rig, 0.3, FOLLOW)
    assert rig.game.screen == COUNTDOWN


def test_a_brief_dropout_keeps_the_lock(rig):
    to_find_player(rig)
    step(rig, 0.6, FOLLOW)
    step(rig, 0.1, None)                  # the tracker blinks
    step(rig, 0.5, FOLLOW)
    assert rig.game.screen == COUNTDOWN


def test_a_long_dropout_resets_the_lock(rig):
    to_find_player(rig)
    step(rig, 0.6, FOLLOW)
    step(rig, 0.4, OUTSIDE)
    step(rig, 0.6, FOLLOW)
    assert rig.game.screen == FIND_PLAYER
    step(rig, 0.5, FOLLOW)
    assert rig.game.screen == COUNTDOWN


def test_losing_the_head_during_the_countdown_returns_to_find_player(rig):
    to_find_player(rig)
    step(rig, 1.2, FOLLOW)
    assert rig.game.screen == COUNTDOWN
    step(rig, 0.5, None)
    assert rig.game.screen == FIND_PLAYER
    assert rig.game._toast is not None     # and says why


def test_a_blink_during_the_countdown_is_forgiven(rig):
    to_find_player(rig)
    step(rig, 1.2, FOLLOW)
    step(rig, 0.05, None)                  # shorter than the grace time
    step(rig, 3.0, FOLLOW)
    assert rig.game.screen == PLAYING


def test_find_player_times_out_to_attract(rig):
    to_find_player(rig)
    step(rig, 29.0, None)
    assert rig.game.screen == FIND_PLAYER
    step(rig, 1.5, None)
    assert rig.game.screen == ATTRACT


def test_escape_leaves_find_player(rig):
    to_find_player(rig)
    press(rig.game, pygame.K_ESCAPE)
    assert rig.game.screen == ATTRACT


# ------------------------------------------------------------- recording


@pytest.fixture
def recording_rig(tmp_path):
    r = make_rig(tmp_path, settings=TEST_SETTINGS.with_changes(record_runs=True))
    yield r
    r.store.close()


def test_nothing_is_recorded_by_default(rig):
    to_playing(rig)
    play_until_out(rig, 1.0)
    assert rig.camera.kept == [] and rig.camera.recording is None


def test_a_run_is_recorded_from_the_countdown_to_the_result(recording_rig):
    r = recording_rig
    to_find_player(r, "JO SMITH")
    assert r.camera.recording is None          # nobody locked on yet
    step(r, 1.2, FOLLOW)
    assert r.game.screen == COUNTDOWN
    path = r.camera.recording
    assert path.parent == r.game.recordings and path.name.endswith("_JO-SMITH.mp4")
    step(r, 3.1, FOLLOW)
    play_until_out(r, 1.0)
    assert r.camera.kept == [path] and r.camera.recording is None


def test_a_countdown_fallback_keeps_the_same_recording(recording_rig):
    r = recording_rig
    to_find_player(r)
    step(r, 1.2, FOLLOW)
    path = r.camera.recording
    step(r, 0.5, None)
    assert r.game.screen == FIND_PLAYER
    step(r, 1.2, FOLLOW)
    assert r.game.screen == COUNTDOWN
    assert r.camera.recording == path and r.camera.kept == r.camera.discarded == []


def test_a_player_who_never_plays_leaves_no_recording(recording_rig):
    r = recording_rig
    to_find_player(r)
    step(r, 1.2, FOLLOW)
    press(r.game, pygame.K_ESCAPE)
    assert r.game.screen == ATTRACT
    assert r.camera.kept == [] and len(r.camera.discarded) == 1


def test_quitting_mid_run_keeps_the_recording(recording_rig):
    r = recording_rig
    to_playing(r)
    step(r, 1.0, FOLLOW)
    r.game.close()
    assert len(r.camera.kept) == 1


# ------------------------------------------------------------- the run


def play_until_out(rig, inside_seconds):
    step(rig, inside_seconds, FOLLOW)
    for _ in range(30 * 20):
        if rig.game.screen != PLAYING:
            break
        step(rig, 1 / 30, OUTSIDE)
        if rig.game.run.respawning:
            step(rig, rig.game.run.config.respawn_seconds + 0.1, FOLLOW)
    assert rig.game.screen == RESULT


def test_losing_every_life_submits_and_keeps_the_best(rig):
    to_playing(rig)
    play_until_out(rig, 5.0)
    first = rig.game.final_score
    assert first >= 50
    entry = rig.store.find("JO")
    assert entry.score == first
    assert rig.store.get_live().highlight_id == entry.id     # the board is told
    assert rig.game.outcome == g.NEW_TOP

    step(rig, 9.0, None)                                     # result times out
    assert rig.game.screen == ATTRACT

    to_playing(rig, "jo")
    play_until_out(rig, 0.5)
    assert rig.game.final_score < first
    assert rig.store.find("JO").score == first               # not overwritten
    assert rig.game.submit.previous_best == first
    assert rig.game.outcome == g.NOT_BEST
    assert rig.store.total_players() == 1


def test_a_lost_life_pauses_until_the_player_is_back_then_counts_down(rig):
    to_playing(rig)
    step(rig, 1.0, FOLLOW)
    step(rig, 0.3, OUTSIDE)
    assert rig.game.run.lives == 2 and rig.game.last_result.waiting
    paused = rig.game.run.survived(rig.clock.t)
    step(rig, 10.0, OUTSIDE)                                 # no clock, no more lives
    assert rig.game.screen == PLAYING
    assert rig.game.run.lives == 2
    assert rig.game.run.survived(rig.clock.t) == paused
    surface = pygame.Surface((640, 360))
    rig.game.draw(surface)
    step(rig, 0.2, FOLLOW)
    assert rig.game.last_result.resume_in > 0
    rig.game.draw(surface)
    step(rig, TEST_SETTINGS.respawn_seconds, FOLLOW)
    assert not rig.game.run.respawning
    assert rig.game.run.survived(rig.clock.t) > paused
    rig.game.draw(surface)


def test_result_ignores_keys_at_first(rig):
    to_playing(rig)
    play_until_out(rig, 1.0)
    press(rig.game, pygame.K_SPACE, " ")
    assert rig.game.screen == RESULT
    step(rig, 1.0, None)
    press(rig.game, pygame.K_SPACE, " ")
    assert rig.game.screen == ATTRACT


def test_settings_apply_to_the_next_run_not_mid_run(rig):
    to_playing(rig)
    side = rig.game.run.square.side
    rig.store.save_settings(TEST_SETTINGS.with_changes(lives=5, start_side=0.30,
                                                       min_side=0.30))
    step(rig, 1.0, FOLLOW)                                   # long enough to poll
    assert rig.game.settings.lives == 5                      # picked up...
    assert rig.game.run.config.lives == 3                    # ...but not mid-run
    assert rig.game.run.square.config.start_side != 0.30
    assert rig.game.run.square.side <= side

    play_until_out(rig, 0.5)
    step(rig, 9.0, None)
    to_find_player(rig)
    assert rig.game.run.config.lives == 5
    assert rig.game.run.square.side == pytest.approx(0.30)


def test_detection_settings_apply_immediately(rig):
    rig.store.save_settings(TEST_SETTINGS.with_changes(mode=MODE_UPPER_BODY, zoom=2.0))
    step(rig, 1.0, None)
    assert rig.camera.configured
    assert rig.camera.configured[-1].mode == MODE_UPPER_BODY
    assert rig.camera.configured[-1].zoom == 2.0


def test_square_matches_the_frame_aspect(rig):
    rig.camera.detection = HeadDetection(frame=np.zeros((480, 640, 3), np.uint8), ok=True)
    rig.game.update()
    to_find_player(rig)
    assert rig.game.run.square.config.aspect == pytest.approx(640 / 480)


def test_live_state_is_throttled(rig):
    calls = []
    original = rig.store.set_live

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    rig.store.set_live = counting
    to_playing(rig)
    calls.clear()
    step(rig, 1.0, FOLLOW, fps=60)
    assert 5 <= len(calls) <= 12
    live = rig.store.get_live()
    assert live.phase == PLAYING and live.lives == 3 and live.max_lives == 3
    assert live.player_name == "JO" and live.score > 0


def test_phase_changes_are_published_at_once(rig):
    to_find_player(rig)
    assert rig.store.get_live().phase == FIND_PLAYER
    rig.game.close()
    assert rig.store.get_live().phase == "idle"


# ------------------------------------------------------------- settings


def test_tab_opens_settings_only_on_attract(rig):
    FakeMenu.instances.clear()
    game = rig.game
    press(game, pygame.K_TAB)
    assert game.menu is not None
    menu = game.menu
    press(game, pygame.K_a, "a")                 # the menu has the keyboard
    assert game.screen == ATTRACT and menu.events
    press(game, pygame.K_ESCAPE)                 # closes the menu, not the game
    assert game.menu is None and game.running

    press(game, pygame.K_RETURN)
    assert game.screen == NAME_ENTRY
    press(game, pygame.K_TAB)
    assert game.menu is None and game.name == ""
    type_name(game, "JO")
    press(game, pygame.K_RETURN)
    press(game, pygame.K_TAB)
    assert game.menu is None


def test_saving_from_the_menu_applies_the_settings(rig):
    press(rig.game, pygame.K_TAB)
    rig.game.menu.on_save(TEST_SETTINGS.with_changes(countdown_seconds=5))
    assert rig.store.load_settings().countdown_seconds == 5
    assert rig.game.settings.countdown_seconds == 5


def test_gear_click_opens_settings(rig):
    rig.game.draw(pygame.Surface((640, 360)))
    assert rig.game.gear_rect.width > 0
    rig.game.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rig.game.gear_rect.center,
                                       button=1))
    assert rig.game.menu is not None


def test_a_missing_settings_menu_is_not_fatal(tmp_path):
    def broken(*args, **kwargs):
        raise ImportError("no settings_menu yet")

    r = make_rig(tmp_path, menu_factory=broken)
    press(r.game, pygame.K_TAB)
    assert r.game.menu is None and r.game.running
    assert r.game._toast is not None
    r.game.draw(pygame.Surface((640, 360)))
    r.store.close()


# ---------------------------------------------------------------- drawing


def test_every_screen_draws(rig):
    game = rig.game
    frame = np.full((360, 640, 3), 90, np.uint8)
    for size in ((640, 360), (800, 800), (1280, 720)):
        surface = pygame.Surface(size)
        for with_frame in (False, True):
            detection = HeadDetection(frame=frame if with_frame else None, head=(0.5, 0.5),
                                      box=(0.45, 0.4, 0.1, 0.2), ok=True, timestamp=1.0)
            rig.camera.detection = detection
            game.name = "JO"
            for screen in (ATTRACT, NAME_ENTRY, FIND_PLAYER, COUNTDOWN, PLAYING):
                game.go(screen)
                step(rig, 0.4, (0.5, 0.5))
                game.draw(surface)
            # A loss flash, danger and respawn.
            step(rig, 0.3, OUTSIDE)
            game.draw(surface)
            step(rig, 0.05, OUTSIDE)
            game.draw(surface)
            game._finish_run(rig.clock.t)
            for _ in range(4):
                step(rig, 0.3, None)
                game.draw(surface)
    press(game, pygame.K_TAB)
    game.go(ATTRACT)
    press(game, pygame.K_TAB)
    game.draw(pygame.Surface((640, 360)))
    assert game.menu.drawn == 1


def test_camera_error_is_shown_not_raised(rig):
    rig.camera.error = "Camera 0 could not be opened"
    rig.camera.detection = HeadDetection(ok=False)
    rig.game.update()
    rig.game.draw(pygame.Surface((640, 360)))


# ---------------------------------------------------------------- dev mode


def test_dev_mouse_stands_in_for_the_head(tmp_path):
    r = make_rig(tmp_path, dev=True)
    game = r.game
    r.camera.detection = HeadDetection(head=(0.9, 0.9), ok=True)   # ignored in dev
    game.draw(pygame.Surface((640, 360)))                          # establishes the view
    game.update()
    assert game.head is None
    game.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(320, 180), rel=(0, 0),
                                   buttons=(0, 0, 0)))
    game.update()
    assert game.head == pytest.approx((0.5, 0.5), abs=0.01)
    game.handle(pygame.event.Event(pygame.WINDOWLEAVE))
    game.update()
    assert game.head is None
    r.store.close()


def test_dev_mouse_can_lock_on(tmp_path):
    r = make_rig(tmp_path, dev=True)
    game = r.game
    game.draw(pygame.Surface((640, 360)))
    to_find_player(r)
    game.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(320, 180), rel=(0, 0),
                                   buttons=(0, 0, 0)))
    for _ in range(40):
        r.clock.t += 1 / 30
        game.update()
    assert game.screen == COUNTDOWN
    r.store.close()
