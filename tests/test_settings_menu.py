"""Headless tests for the in-game settings overlay.

Same style as test_widgets.py: synthetic pygame events, no real window
(conftest.py sets SDL_VIDEODRIVER=dummy before pygame is imported).
"""

from __future__ import annotations

import pygame
import pytest

from searchlight.settings import DEFAULTS, GROUPS, Settings, matching_preset
from searchlight.settings_menu import SettingsMenu

pygame.init()


def key(k, mod=0, unicode=""):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=unicode)


def mouse_down(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def mouse_up(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def make_menu(settings=None, on_save=None, on_close=None):
    return SettingsMenu(settings or Settings(), on_save=on_save or (lambda s: None),
                        on_close=on_close or (lambda: None))


def render(menu, size=(1280, 720), now=1.0):
    """Runs one frame so widgets get real rects, needed before clicking."""
    surface = pygame.Surface(size)
    menu.update(now)
    menu.draw(surface)
    return surface


def click(menu, widget, size=(1280, 720)):
    render(menu, size)
    pos = widget.rect.center
    menu.handle(mouse_down(pos))
    menu.handle(mouse_up(pos))


# ------------------------------------------------------------------ arrows


def test_right_arrow_increases_focused_value():
    menu = make_menu()
    render(menu)
    lives_row = menu._rows[0]
    assert lives_row.spec.key == "lives"
    menu._focus.focus(lives_row.control)
    assert menu._working.lives == 3
    menu.handle(key(pygame.K_RIGHT))
    assert menu._working.lives == 4
    assert lives_row.control.value == 4


def test_left_right_clamp_to_setting_bounds():
    menu = make_menu()
    render(menu)
    lives_row = menu._rows[0]  # lives: min 1, max 9, step 1
    menu._focus.focus(lives_row.control)
    for _ in range(20):
        menu.handle(key(pygame.K_RIGHT))
    assert menu._working.lives == 9
    for _ in range(20):
        menu.handle(key(pygame.K_LEFT))
    assert menu._working.lives == 1


def test_up_down_move_focus_not_value():
    menu = make_menu()
    render(menu)
    lives_row = menu._rows[0]
    grace_row = menu._rows[1]
    menu._focus.focus(lives_row.control)
    menu.handle(key(pygame.K_DOWN))
    assert menu._focus.focused_widget() is grace_row.control
    assert menu._working.lives == 3  # unchanged - Down moved focus, not the value


# ------------------------------------------------------------------ presets


def test_default_settings_match_normal_preset():
    menu = make_menu()
    menu.update(1.0)
    assert menu._current_preset == "Normal"


def test_preset_click_applies_and_is_detected():
    menu = make_menu()
    render(menu)
    easy_btn = menu._preset_buttons[0]
    click(menu, easy_btn)
    menu.update(1.0)
    assert menu._current_preset == "Easy"
    assert menu._working.lives == 5  # Easy preset's lives value


def test_custom_values_show_no_matching_preset():
    menu = make_menu()
    render(menu)
    menu._on_row_change("lives", 7)
    menu.update(1.0)
    assert menu._current_preset is None
    assert matching_preset(menu._working) is None


# -------------------------------------------------------------- save/cancel


def test_save_calls_on_save_with_working_settings_then_closes():
    saved, closed = [], []
    menu = make_menu(on_save=saved.append, on_close=lambda: closed.append(1))
    render(menu)
    menu._on_row_change("lives", 5)
    menu.update(1.0)
    click(menu, menu._save_btn)
    assert saved == [menu._working]
    assert saved[0].lives == 5
    assert closed == [1]


def test_save_disabled_when_nothing_changed():
    saved = []
    menu = make_menu(on_save=saved.append)
    menu.update(1.0)
    assert menu._save_btn.enabled is False
    click(menu, menu._save_btn)
    assert saved == []


def test_cancel_without_changes_closes_immediately():
    saved, closed = [], []
    menu = make_menu(on_save=saved.append, on_close=lambda: closed.append(1))
    render(menu)
    click(menu, menu._cancel_btn)
    assert closed == [1]
    assert saved == []
    assert menu._discard_dialog.open is False


# --------------------------------------------------------------- discard flow


def test_cancel_with_changes_opens_discard_dialog_without_closing():
    closed = []
    menu = make_menu(on_close=lambda: closed.append(1))
    render(menu)
    menu._on_row_change("lives", 6)
    click(menu, menu._cancel_btn)
    assert menu._discard_dialog.open is True
    assert closed == []


def test_esc_with_changes_opens_discard_dialog():
    closed = []
    menu = make_menu(on_close=lambda: closed.append(1))
    render(menu)
    menu._on_row_change("lives", 6)
    menu.handle(key(pygame.K_ESCAPE))
    assert menu._discard_dialog.open is True
    assert closed == []


def test_esc_without_changes_closes_directly():
    closed = []
    menu = make_menu(on_close=lambda: closed.append(1))
    render(menu)
    menu.handle(key(pygame.K_ESCAPE))
    assert closed == [1]
    assert menu._discard_dialog.open is False


def test_discard_dialog_confirm_closes_menu():
    closed = []
    menu = make_menu(on_close=lambda: closed.append(1))
    render(menu)
    menu._on_row_change("lives", 6)
    menu._request_cancel()
    assert menu._discard_dialog.open is True
    menu.handle(key(pygame.K_RETURN))  # confirm "Discard"
    assert closed == [1]
    assert menu._discard_dialog.open is False


def test_discard_dialog_cancel_keeps_menu_open():
    closed = []
    menu = make_menu(on_close=lambda: closed.append(1))
    render(menu)
    menu._on_row_change("lives", 6)
    menu._request_cancel()
    menu.handle(key(pygame.K_ESCAPE))  # "Keep editing"
    assert closed == []
    assert menu._discard_dialog.open is False
    assert menu._working.lives == 6  # edit is preserved, nothing was discarded


# -------------------------------------------------------------- reset flow


def test_reset_all_dialog_confirm_restores_defaults():
    menu = make_menu()
    render(menu)
    menu._on_row_change("lives", 8)
    menu._confirm_reset_all()
    assert menu._reset_dialog.open is True
    menu.handle(key(pygame.K_RETURN))
    assert menu._working == DEFAULTS
    assert menu._rows[0].control.value == DEFAULTS.lives


def test_reset_focused_row_via_backspace():
    menu = make_menu()
    render(menu)
    lives_row = menu._rows[0]
    menu._focus.focus(lives_row.control)
    menu._on_row_change("lives", 8)
    assert menu._working.lives == 8
    menu.handle(key(pygame.K_BACKSPACE))
    assert menu._working.lives == DEFAULTS.lives
    assert lives_row.control.value == DEFAULTS.lives


# ------------------------------------------------------------------- tabs


def test_digit_hotkey_switches_tab_and_rebuilds_rows():
    menu = make_menu()
    render(menu)
    menu.handle(key(pygame.K_3))  # Detection is GROUPS[2]
    assert GROUPS[menu._tabs.active_index] == "Detection"
    assert len(menu._rows) == 9
    assert {row.spec.key for row in menu._rows} == {
        "mode", "long_range", "zoom", "min_confidence", "smoothing",
        "resolution", "camera_index", "mirror", "show_tracking"}


def test_tab_switch_hides_presets_outside_gameplay_square():
    menu = make_menu()
    render(menu)
    menu.handle(key(pygame.K_5))  # Leaderboard
    assert GROUPS[menu._tabs.active_index] == "Leaderboard"
    assert all(btn not in menu._focus._widgets for btn in menu._preset_buttons)


def test_tab_click_switches_tab():
    menu = make_menu()
    render(menu)
    tabs_rect = menu._tabs.rect
    # Middle of the third segment -> "Detection"
    pos = (tabs_rect.left + int(tabs_rect.width * 2.5 / len(GROUPS)), tabs_rect.centery)
    menu.handle(mouse_down(pos))
    assert GROUPS[menu._tabs.active_index] == "Detection"


# ------------------------------------------------------------ consistency


def test_consistency_refresh_pulls_min_side_down_and_resyncs_control():
    menu = make_menu()
    render(menu)
    menu.handle(key(pygame.K_2))  # Square tab
    render(menu)
    min_side_row = next(r for r in menu._rows if r.spec.key == "min_side")
    assert menu._working.start_side == pytest.approx(0.42)
    menu._on_row_change("min_side", 0.60)  # above start_side, and above the spec's own max
    # Setting.clamp caps min_side at its own max (0.50) first, then _consistent()
    # pulls it down further because it now exceeds start_side (0.42).
    assert menu._working.min_side == pytest.approx(0.42)
    assert menu._working.min_side <= menu._working.start_side
    assert min_side_row.control.value == pytest.approx(0.42)


def test_consistency_refresh_on_max_speed_below_start_speed():
    menu = make_menu()
    render(menu)
    menu.handle(key(pygame.K_2))  # Square tab
    render(menu)
    menu._on_row_change("start_speed", 0.5)
    assert menu._working.max_speed >= menu._working.start_speed
    max_speed_row = next(r for r in menu._rows if r.spec.key == "max_speed")
    assert max_speed_row.control.value == pytest.approx(menu._working.max_speed)


# --------------------------------------------------------------------- misc


def test_changed_keys_tracks_gold_dot_state():
    menu = make_menu()
    render(menu)
    menu.update(1.0)
    assert menu._changed == set()
    menu._on_row_change("lives", 5)
    menu.update(1.0)
    assert "lives" in menu._changed


def test_wheel_scrolls_detection_tab_rows():
    menu = make_menu()
    render(menu, size=(1280, 400))  # short window: forces scrolling on Detection
    menu.handle(key(pygame.K_3))
    render(menu, size=(1280, 400))
    max_scroll = menu._max_scroll()
    if max_scroll == 0:
        pytest.skip("viewport tall enough to fit all rows even at this size")
    wheel = pygame.event.Event(pygame.MOUSEWHEEL, y=-1, x=0, flipped=False, which=0,
                               pos=menu._viewport.center)
    menu.handle(wheel)
    assert menu._scroll == 1
