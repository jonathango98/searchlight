"""Headless tests for the widget kit: synthetic pygame events, no window.

conftest.py sets SDL_VIDEODRIVER=dummy before pygame is imported anywhere,
so `pygame.init()` here never touches a real display.
"""

from __future__ import annotations

import pygame
import pytest

from searchlight.ui import widgets as w

pygame.init()


def key(k, mod=0, unicode=""):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=unicode)


def mouse_down(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def mouse_up(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def mouse_move(pos):
    return pygame.event.Event(pygame.MOUSEMOTION, pos=pos)


def wheel(y, pos=None):
    e = pygame.event.Event(pygame.MOUSEWHEEL, y=y, x=0, flipped=False, which=0)
    if pos is not None:
        e.pos = pos
    return e


# ------------------------------------------------------------------ button


def test_button_click_via_mouse():
    calls = []
    b = w.Button("Go", on_click=lambda: calls.append(1))
    b.layout(pygame.Rect(0, 0, 100, 40))
    assert b.handle(mouse_down((50, 20))) is True
    assert b.handle(mouse_up((50, 20))) is True
    assert calls == [1]


def test_button_mouse_up_outside_does_not_click():
    calls = []
    b = w.Button("Go", on_click=lambda: calls.append(1))
    b.layout(pygame.Rect(0, 0, 100, 40))
    b.handle(mouse_down((50, 20)))
    b.handle(mouse_up((500, 500)))
    assert calls == []


def test_button_enter_and_space_when_focused():
    calls = []
    b = w.Button("Go", on_click=lambda: calls.append(1))
    b.layout(pygame.Rect(0, 0, 100, 40))
    b.focused = True
    assert b.handle(key(pygame.K_RETURN)) is True
    assert b.handle(key(pygame.K_SPACE)) is True
    assert calls == [1, 1]


def test_button_disabled_ignores_everything():
    calls = []
    b = w.Button("Go", on_click=lambda: calls.append(1), enabled=False)
    b.layout(pygame.Rect(0, 0, 100, 40))
    b.focused = True
    b.handle(mouse_down((50, 20)))
    b.handle(mouse_up((50, 20)))
    b.handle(key(pygame.K_RETURN))
    assert calls == []


# ------------------------------------------------------------------ toggle


def test_toggle_click_flips_and_notifies():
    values = []
    t = w.Toggle(False, on_change=values.append)
    t.layout(pygame.Rect(0, 0, 60, 30))
    t.handle(mouse_down((30, 15)))
    assert t.value is True
    assert values == [True]


def test_toggle_arrow_keys_when_focused():
    t = w.Toggle(False)
    t.layout(pygame.Rect(0, 0, 60, 30))
    t.focused = True
    t.handle(key(pygame.K_LEFT))
    assert t.value is True
    t.handle(key(pygame.K_RIGHT))
    assert t.value is False


def test_toggle_set_value_no_notify_when_unchanged():
    values = []
    t = w.Toggle(True, on_change=values.append)
    t.set_value(True)
    assert values == []


# ------------------------------------------------------------------ choice


def test_choice_click_selects_segment():
    changes = []
    c = w.Choice((("a", "Face"), ("b", "Body")), "a", on_change=changes.append)
    c.layout(pygame.Rect(0, 0, 200, 40))
    c.handle(mouse_down((150, 20)))  # right half -> "b"
    assert c.value == "b"
    assert changes == ["b"]


def test_choice_arrow_cycle_wraps():
    c = w.Choice((("a", "A"), ("b", "B"), ("c", "C")), "c")
    c.layout(pygame.Rect(0, 0, 200, 40))
    c.focused = True
    c.handle(key(pygame.K_RIGHT))
    assert c.value == "a"
    c.handle(key(pygame.K_LEFT))
    assert c.value == "c"


# ------------------------------------------------------------------ slider


def test_slider_arrow_keys_step_and_clamp():
    changes = []
    s = w.Slider(5, min_value=0, max_value=10, step=1, integer=True, on_change=changes.append)
    s.layout(pygame.Rect(0, 0, 200, 30))
    s.focused = True
    s.handle(key(pygame.K_RIGHT))
    assert s.value == 6
    for _ in range(10):
        s.handle(key(pygame.K_RIGHT))
    assert s.value == 10  # clamped to max
    s.handle(key(pygame.K_HOME))
    assert s.value == 0


def test_slider_wheel_only_when_hovered():
    s = w.Slider(5, min_value=0, max_value=10, step=1, integer=True)
    s.layout(pygame.Rect(0, 0, 200, 30))
    assert s.handle(wheel(1, pos=(100, 15))) is True
    assert s.value == 6
    assert s.handle(wheel(1, pos=(900, 900))) is False
    assert s.value == 6


def test_slider_drag_sets_value_from_position():
    s = w.Slider(0, min_value=0, max_value=100, step=1, integer=True)
    s.layout(pygame.Rect(0, 0, 200, 30))
    s.handle(mouse_down((0, 15)))
    assert s.value == 0
    s.handle(mouse_move((200, 15)))
    assert s.value == 100
    s.handle(mouse_up((200, 15)))
    assert s._dragging is False


class _FakeSpec:
    kind = "float"
    min = 0.0
    max = 1.0
    step = 0.1

    def clamp(self, v):
        v = max(self.min, min(self.max, v))
        return round(v, 4)

    def nudge(self, v, steps):
        return self.clamp(v + steps * self.step)

    def format(self, v):
        return f"{v*100:.0f}%"

    def fraction(self, v):
        return (v - self.min) / (self.max - self.min)

    def from_fraction(self, f):
        return self.clamp(self.min + f * (self.max - self.min))


def test_slider_spec_binding_uses_spec_math():
    s = w.Slider(0.5, spec=_FakeSpec())
    s.layout(pygame.Rect(0, 0, 200, 30))
    assert s._format(s.value) == "50%"
    s.focused = True
    s.handle(key(pygame.K_RIGHT))
    assert s.value == pytest.approx(0.6)
    s.handle(key(pygame.K_END))
    assert s.value == 1.0


# --------------------------------------------------------------- text field


def test_textfield_types_filters_and_max_length():
    field = w.TextField("", max_length=4, allowed_chars="AB12")
    field.layout(pygame.Rect(0, 0, 200, 30))
    field.focused = True
    for ch in "Ax1B29":
        field.handle(key(pygame.K_a, unicode=ch))
    # 'x' and the second '9' beyond max_length are rejected/dropped
    assert field.text == "A1B2"


def test_textfield_uppercase_option():
    field = w.TextField("", uppercase=True)
    field.layout(pygame.Rect(0, 0, 200, 30))
    field.focused = True
    field.handle(key(pygame.K_a, unicode="j"))
    field.handle(key(pygame.K_a, unicode="o"))
    assert field.text == "JO"


def test_textfield_backspace_and_caret_moves():
    field = w.TextField("HELLO")
    field.layout(pygame.Rect(0, 0, 200, 30))
    field.focused = True
    field.caret = 5
    field.handle(key(pygame.K_BACKSPACE))
    assert field.text == "HELL"
    field.handle(key(pygame.K_HOME))
    assert field.caret == 0
    field.handle(key(pygame.K_DELETE))
    assert field.text == "ELL"


def test_textfield_enter_commits_escape_reverts():
    committed = []
    field = w.TextField("OLD", on_commit=committed.append)
    field.layout(pygame.Rect(0, 0, 200, 30))
    field.focused = True
    field.caret = len(field.text)
    field.handle(key(pygame.K_a, unicode="!"))
    assert field.text == "OLD!"
    field.handle(key(pygame.K_ESCAPE))
    assert field.text == "OLD"
    field.handle(key(pygame.K_a, unicode="?"))
    field.handle(key(pygame.K_RETURN))
    assert field.text == "OLD?"
    assert committed == ["OLD?"]


# ------------------------------------------------------------- number field


def test_numberfield_valid_commit():
    committed = []
    field = w.NumberField(23.4, on_commit=committed.append)
    field.layout(pygame.Rect(0, 0, 150, 30))
    assert field.invalid is False
    field.focused = True
    field.caret = len(field.text)
    field.handle(key(pygame.K_a, unicode="5"))  # "23.45"
    assert field.invalid is False
    field.handle(key(pygame.K_RETURN))
    assert committed == [23.45]
    assert field.get_number() == 23.45


def test_numberfield_invalid_blocks_commit():
    committed = []
    field = w.NumberField(1.0, on_commit=committed.append)
    field.layout(pygame.Rect(0, 0, 150, 30))
    field.focused = True
    field.caret = len(field.text)
    field.handle(key(pygame.K_a, unicode="."))
    field.handle(key(pygame.K_a, unicode="."))  # "1.." -> unparsable
    assert field.invalid is True
    field.handle(key(pygame.K_RETURN))
    assert committed == []
    assert field.invalid is True


def test_numberfield_validator_range():
    field = w.NumberField(5.0, validator=lambda v: 0 <= v <= 10)
    field.layout(pygame.Rect(0, 0, 150, 30))
    field.focused = True
    field.text = "99"
    field.caret = 2
    field._check(field.text)
    assert field.invalid is True


# -------------------------------------------------------------- scroll list


def _items(n=20):
    return [{"name": f"P{i}"} for i in range(n)]


def render_noop(surface, rect, item, index, selected, hovered, now):
    pass


def test_scrolllist_arrow_keys_move_selection():
    lst = w.ScrollList(_items(), render_noop, row_height=20)
    lst.layout(pygame.Rect(0, 0, 200, 100))  # 5 visible rows
    lst.focused = True
    lst.select(0, notify=False)
    lst.handle(key(pygame.K_DOWN))
    assert lst.selected_index == 1
    lst.handle(key(pygame.K_PAGEDOWN))
    assert lst.selected_index == 6
    lst.handle(key(pygame.K_END))
    assert lst.selected_index == 19
    lst.handle(key(pygame.K_HOME))
    assert lst.selected_index == 0


def test_scrolllist_up_down_clamped_at_ends():
    lst = w.ScrollList(_items(3), render_noop, row_height=20)
    lst.layout(pygame.Rect(0, 0, 200, 100))
    lst.focused = True
    lst.select(0, notify=False)
    lst.handle(key(pygame.K_UP))
    assert lst.selected_index == 0
    lst.select(2, notify=False)
    lst.handle(key(pygame.K_DOWN))
    assert lst.selected_index == 2


def test_scrolllist_keeps_selection_visible():
    lst = w.ScrollList(_items(20), render_noop, row_height=20)
    lst.layout(pygame.Rect(0, 0, 200, 100))  # 5 visible rows
    lst.focused = True
    lst.select(19, notify=False)
    assert lst._scroll == 15  # last row visible at the bottom
    lst.select(0, notify=False)
    assert lst._scroll == 0


def test_scrolllist_click_selects_and_double_click_activates():
    selected, activated = [], []
    lst = w.ScrollList(_items(10), render_noop, row_height=20,
                       on_select=selected.append, on_activate=activated.append)
    lst.layout(pygame.Rect(0, 0, 200, 100))
    lst.select(None, notify=False)  # clear the default selection first
    lst.handle(mouse_down((50, 30)))  # row 1
    assert selected == [1]
    assert activated == []
    lst.handle(mouse_down((50, 30)))  # second click, same row -> activate
    assert activated == [1]


def test_scrolllist_wheel_scrolls_when_hovered():
    lst = w.ScrollList(_items(20), render_noop, row_height=20)
    lst.layout(pygame.Rect(0, 0, 200, 100))
    assert lst.handle(wheel(-1, pos=(50, 50))) is True
    assert lst._scroll == 1
    assert lst.handle(wheel(-1, pos=(900, 900))) is False
    assert lst._scroll == 1


def test_scrolllist_set_items_resets_when_empty():
    lst = w.ScrollList(_items(5), render_noop, row_height=20)
    lst.layout(pygame.Rect(0, 0, 200, 100))
    lst.set_items([])
    assert lst.selected_index is None
    lst.set_items(_items(3))
    assert lst.selected_index == 0


# ------------------------------------------------------------------- tabs


def test_tabs_click_selects():
    changes = []
    tabs = w.Tabs(["Gameplay", "Square", "Detection", "Leaderboard"], on_change=changes.append)
    tabs.layout(pygame.Rect(0, 0, 400, 40))
    tabs.handle(mouse_down((350, 20)))  # last segment
    assert tabs.active_index == 3
    assert changes == [3]


def test_tabs_arrow_keys_when_focused():
    tabs = w.Tabs(["A", "B", "C"], active=0)
    tabs.layout(pygame.Rect(0, 0, 300, 40))
    tabs.focused = True
    tabs.handle(key(pygame.K_LEFT))
    assert tabs.active_index == 2  # wraps backward
    tabs.handle(key(pygame.K_RIGHT))
    assert tabs.active_index == 0


def test_tabs_digit_hotkeys_work_unfocused():
    tabs = w.Tabs(["A", "B", "C"], active=0)
    tabs.layout(pygame.Rect(0, 0, 300, 40))
    tabs.handle(key(pygame.K_3))
    assert tabs.active_index == 2


# -------------------------------------------------------------- confirm dialog


def test_confirmdialog_enter_confirms_escape_cancels():
    confirmed, cancelled = [], []
    dlg = w.ConfirmDialog("Delete?", "Are you sure?",
                          on_confirm=lambda: confirmed.append(1),
                          on_cancel=lambda: cancelled.append(1))
    dlg.layout(pygame.Rect(0, 0, 800, 600))
    dlg.show()
    assert dlg.handle(key(pygame.K_RETURN)) is True
    assert confirmed == [1]
    assert dlg.open is False

    dlg.show()
    assert dlg.handle(key(pygame.K_ESCAPE)) is True
    assert cancelled == [1]
    assert dlg.open is False


def test_confirmdialog_closed_ignores_events():
    confirmed = []
    dlg = w.ConfirmDialog("Delete?", "Sure?", on_confirm=lambda: confirmed.append(1))
    dlg.layout(pygame.Rect(0, 0, 800, 600))
    assert dlg.handle(key(pygame.K_RETURN)) is False
    assert confirmed == []


def test_confirmdialog_button_click_confirms():
    confirmed = []
    dlg = w.ConfirmDialog("Delete?", "Sure?", on_confirm=lambda: confirmed.append(1))
    dlg.layout(pygame.Rect(0, 0, 800, 600))
    dlg.show()
    dlg.handle(mouse_down(dlg._confirm.rect.center))
    dlg.handle(mouse_up(dlg._confirm.rect.center))
    assert confirmed == [1]
    assert dlg.open is False


def test_confirmdialog_long_message_does_not_overlap_buttons():
    dlg = w.ConfirmDialog("Clear leaderboard?",
                          "This permanently deletes all fourteen entries currently on the "
                          "board and there is absolutely no way to undo it once confirmed.",
                          danger=True)
    dlg.layout(pygame.Rect(0, 0, 1280, 720))
    message_bottom = dlg._message_y + len(dlg._lines) * dlg._line_h
    assert message_bottom <= dlg._confirm.rect.top


# -------------------------------------------------------------- focus manager


def test_focusmanager_tab_cycles_forward_and_back():
    a, b, c = w.Button("A"), w.Button("B"), w.Button("C")
    fm = w.FocusManager([a, b, c])
    assert fm.focused_widget() is a
    fm.handle(key(pygame.K_TAB))
    assert fm.focused_widget() is b
    fm.handle(key(pygame.K_TAB))
    assert fm.focused_widget() is c
    fm.handle(key(pygame.K_TAB))  # wraps
    assert fm.focused_widget() is a
    fm.handle(key(pygame.K_TAB, mod=pygame.KMOD_SHIFT))
    assert fm.focused_widget() is c


def test_focusmanager_skips_disabled_and_invisible():
    a, b, c = w.Button("A"), w.Button("B"), w.Button("C")
    b.enabled = False
    fm = w.FocusManager([a, b, c])
    fm.handle(key(pygame.K_TAB))
    assert fm.focused_widget() is c


def test_focusmanager_click_focuses_widget_under_pointer():
    a, b = w.Button("A"), w.Button("B")
    a.layout(pygame.Rect(0, 0, 50, 50))
    b.layout(pygame.Rect(100, 0, 50, 50))
    fm = w.FocusManager([a, b])
    fm.handle(mouse_down((120, 20)))
    assert fm.focused_widget() is b
    assert a.focused is False
    assert b.focused is True


# ------------------------------------------------------------------- layout


def test_split_row_and_stack_rows():
    label_r, control_r = w.split_row(pygame.Rect(0, 0, 300, 40), label_width=100, gap=10)
    assert label_r == pygame.Rect(0, 0, 100, 40)
    assert control_r == pygame.Rect(110, 0, 190, 40)

    rows = w.stack_rows(pygame.Rect(0, 0, 100, 100), [20, 30], gap=5)
    assert rows[0] == pygame.Rect(0, 0, 100, 20)
    assert rows[1] == pygame.Rect(0, 25, 100, 30)


def test_wrap_text_breaks_on_width():
    lines = w.wrap_text("one two three four five", "mono", 16, 60)
    assert len(lines) > 1
    assert " ".join(lines).split() == "one two three four five".split()


# ------------------------------------------------------------------- toast


def test_toast_visibility_window():
    toast = w.Toast()
    toast.show("Hi", duration=1.0, now=10.0)
    assert toast.visible(10.0) is True
    assert toast.visible(10.5) is True
    assert toast.visible(11.0 + toast._fade + 0.01) is False
