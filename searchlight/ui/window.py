"""A resizable window that can flip to fullscreen and back.

Both Searchlight windows open through here, so they behave the same: drag the
edges to resize, and press **F11** or **Cmd/Ctrl+F** to toggle fullscreen on
whichever display the window belongs to. Plain F is deliberately not a toggle,
since players type their names with it.

Toggling re-creates the display surface, so callers must fetch `window.surface`
every frame rather than holding on to the one from start-up.
"""

from __future__ import annotations

from typing import Tuple

import pygame


def is_fullscreen_key(event: pygame.event.Event) -> bool:
    if event.type != pygame.KEYDOWN:
        return False
    if event.key == pygame.K_F11:
        return True
    return event.key == pygame.K_f and bool(
        event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL))


def is_quit_key(event: pygame.event.Event) -> bool:
    """Cmd/Ctrl+Q - the one way out that works on every screen."""
    return (event.type == pygame.KEYDOWN and event.key == pygame.K_q
            and bool(event.mod & (pygame.KMOD_META | pygame.KMOD_CTRL)))


class Window:
    def __init__(self, size: Tuple[int, int], fullscreen: bool = False,
                 display: int = 0) -> None:
        self.windowed_size = size
        self.display = display
        self.fullscreen = False
        try:
            self._apply(fullscreen)
        except pygame.error:
            # Only one monitor attached: fall back rather than refusing to start.
            print(f"Display {display} unavailable; using display 0.")
            self.display = 0
            self._apply(fullscreen)

    @property
    def surface(self) -> pygame.Surface:
        return pygame.display.get_surface()

    def toggle(self) -> None:
        self._apply(not self.fullscreen)

    def handle(self, event: pygame.event.Event) -> bool:
        """Deal with resize and fullscreen events. True means it was consumed."""
        if is_fullscreen_key(event):
            self.toggle()
            return True
        if event.type == pygame.VIDEORESIZE and not self.fullscreen:
            # Remembered so leaving fullscreen restores the size the user chose.
            self.windowed_size = (event.w, event.h)
            return True
        return False

    def _apply(self, fullscreen: bool) -> None:
        if fullscreen:
            # (0, 0) means the display's own resolution - no mode switch.
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=self.display)
        else:
            pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE,
                                    display=self.display)
        self.fullscreen = fullscreen
        # Keep the pointer visible in a window; only hide it for a real kiosk.
        pygame.mouse.set_visible(not fullscreen)
