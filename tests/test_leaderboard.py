"""Tests for searchlight.leaderboard - the audience/edit window.

conftest.py sets SDL_VIDEODRIVER=dummy before pygame is imported anywhere, so
`pygame.init()` here never touches a real display. Every test drives the app
on a synthetic clock (a plain float, advanced by hand) rather than real time,
so timing-sensitive behaviour (fades, highlights, toasts, polling) is
deterministic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygame
import pytest

from searchlight import store as store_mod
from searchlight.leaderboard import POLL_INTERVAL, LeaderboardApp
from searchlight.store import Store

pygame.init()

SIZE = (1280, 720)


# --------------------------------------------------------------- fixtures


@pytest.fixture
def store(tmp_path: Path):
    s = Store(tmp_path / "board.db")
    yield s
    s.close()


@pytest.fixture
def surface():
    return pygame.Surface(SIZE)


def make_app(store: Store, size=SIZE) -> LeaderboardApp:
    return LeaderboardApp(store, size)


def settle(app: LeaderboardApp, surface: pygame.Surface, now: float, dt: float = 1.0,
           times: int = 2) -> float:
    """Advance the clock and poll+draw a couple of times so eased animations
    (fade-in, panel slide) finish, and so `poll` (throttled to POLL_INTERVAL)
    actually re-reads the store rather than being skipped."""
    for _ in range(times):
        now += dt
        app.poll(now)
        app.draw(surface, now)
    return now


def key(k, mod=0, unicode="") -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=unicode)


def mouse_down(pos, button=1) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def type_into(app: LeaderboardApp, field_name: str, text: str) -> None:
    """Drive raw KEYDOWN events at a text/number field owned by `app`."""
    field = getattr(app, field_name)
    for ch in text:
        if ch.isdigit():
            k = pygame.key.key_code(ch)
        elif ch == ".":
            k = pygame.K_PERIOD
        else:
            k = pygame.key.key_code(ch.lower())
        field.handle(key(k, unicode=ch))


def clear_field(app: LeaderboardApp, field_name: str, presses: int = 16) -> None:
    field = getattr(app, field_name)
    for _ in range(presses):
        field.handle(key(pygame.K_BACKSPACE))


def seed(store: Store, *pairs) -> None:
    for name, score in pairs:
        store.add_entry(name, score)


# ------------------------------------------------------------------ modes


class TestModeToggle:
    def test_e_enters_edit_mode(self, store, surface):
        app = make_app(store)
        now = settle(app, surface, 0.0)
        assert app.mode == "display"
        app.handle(key(pygame.K_e))
        assert app.mode == "edit"

    def test_e_again_returns_to_display(self, store, surface):
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        app.handle(key(pygame.K_e))
        assert app.mode == "display"

    def test_esc_in_display_mode_signals_quit(self, store, surface):
        app = make_app(store)
        settle(app, surface, 0.0)
        still_running = app.handle(key(pygame.K_ESCAPE))
        assert app.quit is True
        assert still_running is False

    def test_esc_in_edit_mode_exits_edit_not_quit(self, store, surface):
        app = make_app(store)
        settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        assert app.mode == "edit"
        still_running = app.handle(key(pygame.K_ESCAPE))
        assert app.mode == "display"
        assert app.quit is False
        assert still_running is True

    def test_every_mode_draws_without_error(self, store, surface):
        seed(store, ("MAYA", 300), ("DEV", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)  # display, populated
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)
        app.draw(surface, now)  # edit, list
        app.handle(key(pygame.K_r))
        app.draw(surface, now)  # edit, inline rename
        app.handle(key(pygame.K_ESCAPE))
        app.handle(key(pygame.K_t))
        app.draw(surface, now)  # edit, inline time
        app.handle(key(pygame.K_ESCAPE))
        app.handle(key(pygame.K_n))
        app.draw(surface, now)  # edit, add form
        app.handle(key(pygame.K_ESCAPE))
        app.handle(key(pygame.K_DELETE))
        app.draw(surface, now)  # edit, delete confirm
        app.handle(key(pygame.K_ESCAPE))
        app.handle(key(pygame.K_DELETE, mod=pygame.KMOD_SHIFT))
        app.draw(surface, now)  # edit, clear confirm 1
        app.handle(key(pygame.K_RETURN))
        app.draw(surface, now)  # edit, clear confirm 2


# ------------------------------------------------------------------ rename


class TestRename:
    def test_rename_commits(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(key(pygame.K_r))
        assert app._editing == "name"
        clear_field(app, "_rename_field")
        type_into(app, "_rename_field", "SAM")
        app.handle(key(pygame.K_RETURN))

        assert app._editing is None
        names = [e.name for e in store.list_all()]
        assert names == ["SAM"]

    def test_rename_onto_existing_asks_to_confirm(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        alex = next(e for e in app._all if e.name == "ALEX")
        app._edit_list.select(app._index_of_id(alex.id))
        app.handle(key(pygame.K_r))
        clear_field(app, "_rename_field")
        type_into(app, "_rename_field", "SAM")
        app.handle(key(pygame.K_RETURN))

        assert app._confirm_merge.open is True
        # the merge hasn't happened yet - still two rows
        assert len(store.list_all()) == 2

    def test_rename_merge_confirm_keeps_the_higher_score(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        alex = next(e for e in app._all if e.name == "ALEX")
        app._edit_list.select(app._index_of_id(alex.id))
        app.handle(key(pygame.K_r))
        clear_field(app, "_rename_field")
        type_into(app, "_rename_field", "SAM")
        app.handle(key(pygame.K_RETURN))
        app.handle(key(pygame.K_RETURN))  # confirm the merge dialog

        assert app._confirm_merge.open is False
        remaining = store.list_all()
        assert len(remaining) == 1
        assert remaining[0].name == "SAM"
        assert remaining[0].score == 500  # kept the higher of the two

    def test_rename_merge_cancel_leaves_both_entries(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        alex = next(e for e in app._all if e.name == "ALEX")
        app._edit_list.select(app._index_of_id(alex.id))
        app.handle(key(pygame.K_r))
        clear_field(app, "_rename_field")
        type_into(app, "_rename_field", "SAM")
        app.handle(key(pygame.K_RETURN))
        app.handle(key(pygame.K_ESCAPE))  # cancel the merge dialog

        assert app._confirm_merge.open is False
        assert len(store.list_all()) == 2

    def test_empty_name_shows_error_toast_and_does_not_commit(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(key(pygame.K_r))
        clear_field(app, "_rename_field")
        app.handle(key(pygame.K_RETURN))

        assert store.list_all()[0].name == "ALEX"
        assert app.toast.kind == "error"


# --------------------------------------------------------------- time/add


class TestTimeAndAdd:
    def test_edit_time_valid_commits(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(key(pygame.K_t))
        clear_field(app, "_time_field")
        type_into(app, "_time_field", "42.5")
        app.handle(key(pygame.K_RETURN))

        assert app._editing is None
        assert store.list_all()[0].score == 425

    def test_edit_time_invalid_does_not_commit(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(key(pygame.K_t))
        clear_field(app, "_time_field")
        type_into(app, "_time_field", "-")  # not a valid number on its own
        app.handle(key(pygame.K_RETURN))

        assert app._time_field.invalid is True
        assert app._editing == "time"  # stayed in edit, no commit
        assert store.list_all()[0].score == 200

    def test_add_new_name(self, store, surface):
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_n))
        assert app._editing == "add"
        type_into(app, "_add_name", "SKY")
        app.handle(key(pygame.K_TAB))
        type_into(app, "_add_time", "18.6")
        app.handle(key(pygame.K_RETURN))

        assert app._editing is None
        entries = store.list_all()
        assert len(entries) == 1
        assert entries[0].name == "SKY"
        assert entries[0].score == 186
        assert app.toast.kind == "success"

    def test_add_existing_name_keeps_higher_score(self, store, surface):
        seed(store, ("SKY", 300))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_n))
        type_into(app, "_add_name", "SKY")
        app.handle(key(pygame.K_TAB))
        type_into(app, "_add_time", "18.6")  # lower than the existing 30.0s
        app.handle(key(pygame.K_RETURN))

        entries = store.list_all()
        assert len(entries) == 1
        assert entries[0].score == 300  # the higher score survives
        assert "kept the higher score" in (app.toast.message or "")

    def test_add_with_invalid_time_does_not_add(self, store, surface):
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_n))
        type_into(app, "_add_name", "SKY")
        app.handle(key(pygame.K_TAB))
        type_into(app, "_add_time", "abc")
        app.handle(key(pygame.K_RETURN))

        assert store.list_all() == []
        assert app._editing == "add"


# ------------------------------------------------------------ delete/clear


class TestDeleteAndClear:
    def test_delete_with_confirm_removes_entry(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        target_name = app._all[0].name
        app.handle(key(pygame.K_DELETE))
        assert app._confirm_delete.open is True
        app.handle(key(pygame.K_RETURN))

        assert app._confirm_delete.open is False
        remaining = [e.name for e in store.list_all()]
        assert target_name not in remaining
        assert len(remaining) == 1

    def test_delete_with_cancel_keeps_entry(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(key(pygame.K_DELETE))
        app.handle(key(pygame.K_ESCAPE))

        assert app._confirm_delete.open is False
        assert len(store.list_all()) == 2

    def test_clear_all_needs_both_confirms_first_only(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_DELETE, mod=pygame.KMOD_SHIFT))
        assert app._confirm_clear1.open is True
        app.handle(key(pygame.K_RETURN))  # only the first confirm so far
        assert app._confirm_clear1.open is False
        assert app._confirm_clear2.open is True
        # nothing deleted yet - still needs the second confirm
        assert len(store.list_all()) == 2

    def test_clear_all_needs_both_confirms_then_clears(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_DELETE, mod=pygame.KMOD_SHIFT))
        app.handle(key(pygame.K_RETURN))  # first confirm
        app.handle(key(pygame.K_RETURN))  # second confirm

        assert app._confirm_clear2.open is False
        assert store.list_all() == []

    def test_clear_all_cancel_on_second_dialog_keeps_entries(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app.handle(key(pygame.K_DELETE, mod=pygame.KMOD_SHIFT))
        app.handle(key(pygame.K_RETURN))  # first confirm
        app.handle(key(pygame.K_ESCAPE))  # cancel the second

        assert app._confirm_clear2.open is False
        assert len(store.list_all()) == 2


# ---------------------------------------------------------------- polling


class TestPolling:
    def test_selection_survives_external_board_change(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500), ("JO", 100))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        jo = next(e for e in app._all if e.name == "JO")
        app._edit_list.select(app._index_of_id(jo.id))

        # A change from the *other* process (the game window).
        store.add_entry("NEWCOMER", 999)
        now = settle(app, surface, now, dt=0.3)

        assert app._selected_entry() is not None
        assert app._selected_entry().id == jo.id

    def test_highlight_fires_when_highlight_seq_bumps(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        entry_id = store.list_all()[0].id

        store.announce(entry_id)
        now += POLL_INTERVAL * 2
        app.poll(now)

        assert app._highlight_id == entry_id
        assert app._highlight_seq == 1

    def test_highlight_refires_for_same_id_on_second_announce(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        entry_id = store.list_all()[0].id

        store.announce(entry_id)
        now += POLL_INTERVAL * 2
        app.poll(now)
        first_started = app._highlight_started

        now += 5.0
        store.announce(entry_id)  # same id, e.g. a returning player's best unchanged
        now += POLL_INTERVAL * 2
        app.poll(now)

        assert app._highlight_seq == 2
        assert app._highlight_started > first_started

    def test_board_rows_from_settings_is_respected(self, store, surface):
        for i in range(20):
            store.add_entry(f"P{i:02d}", 500 - i)
        settings = store.load_settings()
        store.save_settings(settings.with_changes(board_rows=5))

        app = make_app(store)
        now = settle(app, surface, 0.0)
        assert len(app._top) == 5

    def test_board_rows_change_picked_up_live(self, store, surface):
        for i in range(20):
            store.add_entry(f"P{i:02d}", 500 - i)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        assert len(app._top) == 10  # Settings() default

        settings = store.load_settings()
        store.save_settings(settings.with_changes(board_rows=6))  # min is 5
        now = settle(app, surface, now, dt=0.3)
        assert len(app._top) == 6

    def test_stale_live_is_treated_as_idle(self, store, surface):
        store.set_live(store_mod.PHASE_PLAYING, player_name="SAM", score=100,
                        lives=2, max_lives=3)
        # Backdate updated_at well past the staleness window.
        store._db.execute("UPDATE live_state SET updated_at = updated_at - 10")
        store._db.commit()

        app = make_app(store)
        now = settle(app, surface, 0.0)
        assert app._live_stale is True

    def test_fresh_live_is_not_stale(self, store, surface):
        store.set_live(store_mod.PHASE_PLAYING, player_name="SAM", score=100,
                        lives=2, max_lives=3)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        assert app._live_stale is False


# --------------------------------------------------------------- drawing


class TestDrawingDoesNotError:
    def test_empty_board(self, store, surface):
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)

    def test_three_entries(self, store, surface):
        seed(store, ("MAYA", 300), ("DEV", 234), ("JO", 120))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)

    def test_fifteen_entries_with_highlight(self, store, surface):
        for i in range(15):
            store.add_entry(f"P{i:02d}", 400 - i * 10)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        store.announce(app._top[3].id)
        now = settle(app, surface, now, dt=0.2)
        app.draw(surface, now)

    def test_now_playing_panel(self, store, surface):
        seed(store, ("MAYA", 300))
        store.set_live(store_mod.PHASE_PLAYING, player_name="SAM", score=157,
                        lives=2, max_lives=3)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)

    def test_getting_ready_panel(self, store, surface):
        seed(store, ("MAYA", 300))
        store.set_live(store_mod.PHASE_COUNTDOWN, player_name="ALEX", score=0,
                        lives=3, max_lives=3)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)

    def test_result_tag(self, store, surface):
        seed(store, ("MAYA", 300))
        store.set_live(store_mod.PHASE_RESULT, player_name="DEV", score=411,
                        lives=0, max_lives=3)
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)

    def test_1920x1080_surface(self, store):
        size = (1920, 1080)
        surface = pygame.Surface(size)
        seed(store, ("MAYA", 300), ("DEV", 234))
        app = make_app(store, size)
        now = settle(app, surface, 0.0)
        app.draw(surface, now)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)
        app.draw(surface, now)


# --------------------------------------------------------------- pointer


class TestMouseSelection:
    def test_click_selects_a_row(self, store, surface):
        seed(store, ("ALEX", 200), ("SAM", 500))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        row_rect = app._edit_list.rect
        second_row_point = (row_rect.left + 10,
                             row_rect.top + app._edit_list.row_height + 5)
        app.handle(mouse_down(second_row_point))
        assert app._edit_list.selected_index == 1

    def test_toast_button_click_deletes(self, store, surface):
        seed(store, ("ALEX", 200))
        app = make_app(store)
        now = settle(app, surface, 0.0)
        app.handle(key(pygame.K_e))
        now = settle(app, surface, now, dt=0.3)

        app._edit_list.select(0)
        app.handle(mouse_down(app._btn_delete.rect.center))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                       pos=app._btn_delete.rect.center, button=1))
        assert app._confirm_delete.open is True
        app.handle(mouse_down(app._confirm_delete._confirm.rect.center))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                       pos=app._confirm_delete._confirm.rect.center,
                                       button=1))
        assert store.list_all() == []
