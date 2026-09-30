"""Tests for searchlight.ui.window - the resizable fullscreen window."""

import pygame
import pytest

from searchlight.ui.window import Window, is_fullscreen_key, is_quit_key


@pytest.fixture(autouse=True)
def display():
    """Initialize and quit pygame display for each test."""
    pygame.display.init()
    yield
    pygame.display.quit()


def key(k, mod=0, unicode=""):
    """Helper to create a KEYDOWN event."""
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=unicode)


class TestIsFullscreenKey:
    """Test fullscreen toggle key detection."""

    def test_f11_is_fullscreen_key(self):
        event = key(pygame.K_F11)
        assert is_fullscreen_key(event) is True

    def test_meta_f_is_fullscreen_key(self):
        event = key(pygame.K_f, mod=pygame.KMOD_META)
        assert is_fullscreen_key(event) is True

    def test_ctrl_f_is_fullscreen_key(self):
        event = key(pygame.K_f, mod=pygame.KMOD_CTRL)
        assert is_fullscreen_key(event) is True

    def test_plain_f_is_not_fullscreen_key(self):
        """Plain F without modifiers should not toggle fullscreen (for typing names)."""
        event = key(pygame.K_f)
        assert is_fullscreen_key(event) is False

    def test_other_keys_not_fullscreen(self):
        assert is_fullscreen_key(key(pygame.K_q)) is False
        assert is_fullscreen_key(key(pygame.K_ESCAPE)) is False

    def test_non_keydown_event_not_fullscreen(self):
        event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600)
        assert is_fullscreen_key(event) is False


class TestIsQuitKey:
    """Test quit key detection (Cmd/Ctrl+Q)."""

    def test_cmd_q_is_quit_key(self):
        event = key(pygame.K_q, mod=pygame.KMOD_META)
        assert is_quit_key(event) is True

    def test_ctrl_q_is_quit_key(self):
        event = key(pygame.K_q, mod=pygame.KMOD_CTRL)
        assert is_quit_key(event) is True

    def test_plain_q_is_not_quit_key(self):
        """Plain Q without modifiers should not quit (for typing names)."""
        event = key(pygame.K_q)
        assert is_quit_key(event) is False

    def test_other_keys_not_quit(self):
        assert is_quit_key(key(pygame.K_ESCAPE)) is False
        assert is_quit_key(key(pygame.K_f)) is False

    def test_non_keydown_event_not_quit(self):
        event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=600)
        assert is_quit_key(event) is False


class TestWindowCreation:
    """Test Window initialization."""

    def test_window_creation_default_size(self):
        window = Window((640, 360))

        assert window.windowed_size == (640, 360)
        assert window.fullscreen is False
        assert window.display == 0

    def test_window_creation_with_display_index(self):
        window = Window((640, 360), display=0)

        assert window.display == 0

    def test_window_creation_windowed(self):
        window = Window((640, 360), fullscreen=False)

        assert window.fullscreen is False

    def test_window_has_surface(self):
        window = Window((640, 360))

        assert window.surface is not None
        assert isinstance(window.surface, pygame.Surface)


class TestWindowToggle:
    """Test fullscreen toggle functionality."""

    def test_toggle_from_windowed_to_fullscreen(self):
        window = Window((640, 360), fullscreen=False)
        assert window.fullscreen is False

        window.toggle()

        assert window.fullscreen is True

    def test_toggle_from_fullscreen_to_windowed(self):
        window = Window((640, 360), fullscreen=True)
        assert window.fullscreen is True

        window.toggle()

        assert window.fullscreen is False

    def test_toggle_twice_returns_to_original_state(self):
        window = Window((640, 360), fullscreen=False)
        original_fullscreen = window.fullscreen

        window.toggle()
        window.toggle()

        assert window.fullscreen == original_fullscreen


class TestWindowHandleResize:
    """Test VIDEORESIZE event handling."""

    def test_handle_videoresize_updates_windowed_size(self):
        window = Window((640, 360), fullscreen=False)

        event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=450)
        consumed = window.handle(event)

        assert consumed is True
        assert window.windowed_size == (800, 450)

    def test_handle_videoresize_in_fullscreen_not_consumed(self):
        window = Window((640, 360), fullscreen=True)

        event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=450)
        consumed = window.handle(event)

        assert consumed is False
        # windowed_size should not change when in fullscreen
        assert window.windowed_size == (640, 360)

    def test_handle_videoresize_preserves_size_after_toggle(self):
        window = Window((640, 360), fullscreen=False)

        # Resize to new size
        event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=450)
        window.handle(event)

        # Toggle to fullscreen
        window.toggle()
        assert window.fullscreen is True

        # Toggle back to windowed
        window.toggle()
        assert window.fullscreen is False

        # Should restore the resized size
        assert window.windowed_size == (800, 450)


class TestWindowHandleFullscreenToggle:
    """Test fullscreen toggle via event handling."""

    def test_handle_f11_toggles_fullscreen(self):
        window = Window((640, 360), fullscreen=False)

        event = key(pygame.K_F11)
        consumed = window.handle(event)

        assert consumed is True
        assert window.fullscreen is True

    def test_handle_meta_f_toggles_fullscreen(self):
        window = Window((640, 360), fullscreen=False)

        event = key(pygame.K_f, mod=pygame.KMOD_META)
        consumed = window.handle(event)

        assert consumed is True
        assert window.fullscreen is True

    def test_handle_ctrl_f_toggles_fullscreen(self):
        window = Window((640, 360), fullscreen=False)

        event = key(pygame.K_f, mod=pygame.KMOD_CTRL)
        consumed = window.handle(event)

        assert consumed is True
        assert window.fullscreen is True

    def test_handle_plain_f_not_consumed(self):
        """Plain F should not be consumed and should not toggle fullscreen."""
        window = Window((640, 360), fullscreen=False)

        event = key(pygame.K_f)
        consumed = window.handle(event)

        assert consumed is False
        assert window.fullscreen is False


class TestWindowHandleOtherEvents:
    """Test that other events are not consumed."""

    def test_handle_other_key_not_consumed(self):
        window = Window((640, 360))

        event = key(pygame.K_q)
        consumed = window.handle(event)

        assert consumed is False

    def test_handle_mousemotion_not_consumed(self):
        window = Window((640, 360))

        event = pygame.event.Event(pygame.MOUSEMOTION, x=100, y=100)
        consumed = window.handle(event)

        assert consumed is False

    def test_handle_quit_event_not_consumed(self):
        window = Window((640, 360))

        event = pygame.event.Event(pygame.QUIT)
        consumed = window.handle(event)

        assert consumed is False


class TestWindowToggleRoundTrip:
    """Test fullscreen toggle with resize round-trip."""

    def test_toggle_round_trip_restores_windowed_size(self):
        """From old test: Window((640,360)); VIDEORESIZE 800x450; F11 fullscreen; toggle back -> windowed_size == (800,450)"""
        window = Window((640, 360), fullscreen=False)

        # User resizes window to 800x450
        resize_event = pygame.event.Event(pygame.VIDEORESIZE, w=800, h=450)
        window.handle(resize_event)
        assert window.windowed_size == (800, 450)

        # User presses F11 to go fullscreen
        fullscreen_event = key(pygame.K_F11)
        window.handle(fullscreen_event)
        assert window.fullscreen is True

        # Toggle back to windowed
        window.toggle()
        assert window.fullscreen is False

        # Should restore to the remembered windowed size (800, 450)
        assert window.windowed_size == (800, 450)
