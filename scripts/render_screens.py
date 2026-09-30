#!/usr/bin/env python3
"""Render every Searchlight game screen to PNGs, headless.

    .venv/bin/python scripts/render_screens.py --out DIR [--size 1280x720]

No camera, no window and no real time: the game runs against a stub camera that
serves a synthetic room (a blurred gradient with a head-and-shoulders figure) and
a fake clock the script steps forward, so each screen is captured in exactly the
state it names. Use it to review layout at several resolutions after a change.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# pygame before anything that could import cv2 - see searchlight/game.py.
import pygame  # noqa: E402

import argparse  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from searchlight import game as g  # noqa: E402
from searchlight.store import Store  # noqa: E402
from searchlight.tracking.base import MODE_FACE, HeadDetection  # noqa: E402

Point = Tuple[float, float]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


class StubCamera:
    def __init__(self) -> None:
        self.detection = HeadDetection(ok=True, fps=30.0, mode=MODE_FACE)
        self.error: Optional[str] = None

    def latest(self) -> HeadDetection:
        return self.detection

    def configure(self, options) -> None:
        pass

    def reset_tracking(self) -> None:
        pass

    def start_recording(self, path) -> None:
        pass

    def stop_recording(self, keep: bool = True) -> None:
        return None

    def close(self) -> None:
        pass


# --------------------------------------------------------------- the room


def synthetic_frame(size: Tuple[int, int], head: Optional[Point]) -> np.ndarray:
    """A soft, camera-ish RGB frame: a room with a window, and optionally a person
    whose head centre sits at `head` (normalised)."""
    w, h = size
    sw, sh = max(16, w // 6), max(9, h // 6)
    small = pygame.Surface((sw, sh))
    for y in range(sh):
        t = y / (sh - 1)
        colour = (int(58 + 40 * t), int(62 + 30 * t), int(78 + 8 * t))
        pygame.draw.line(small, colour, (0, y), (sw, y))
    # A bright window, a door, a shelf, a lamp - enough to read as a room.
    pygame.draw.rect(small, (190, 200, 210), (int(sw * 0.66), int(sh * 0.12),
                                              int(sw * 0.22), int(sh * 0.36)))
    pygame.draw.rect(small, (150, 160, 175), (int(sw * 0.765), int(sh * 0.12), 2,
                                              int(sh * 0.36)))
    pygame.draw.rect(small, (96, 74, 58), (int(sw * 0.06), int(sh * 0.22),
                                           int(sw * 0.14), int(sh * 0.62)))
    pygame.draw.rect(small, (84, 70, 62), (int(sw * 0.30), int(sh * 0.30),
                                           int(sw * 0.16), int(sh * 0.035)))
    pygame.draw.circle(small, (230, 196, 140), (int(sw * 0.52), int(sh * 0.18)),
                       max(2, sh // 22))
    pygame.draw.rect(small, (70, 64, 70), (0, int(sh * 0.84), sw, sh))
    if head is not None:
        hx, hy = int(head[0] * sw), int(head[1] * sh)
        face_h = int(sh * 0.22)
        face_w = int(face_h * 0.74)
        body = pygame.Rect(0, 0, int(face_w * 3.4), int(face_h * 2.2))
        body.midtop = (hx, hy + int(face_h * 0.55))
        pygame.draw.ellipse(small, (40, 90, 150), body)
        pygame.draw.rect(small, (40, 90, 150), (body.left, body.centery, body.width, sh))
        pygame.draw.rect(small, (196, 150, 120), (hx - face_w // 5, hy + face_h // 3,
                                                  face_w // 2.5, face_h // 2))
        face = pygame.Rect(0, 0, face_w, face_h)
        face.center = (hx, hy)
        pygame.draw.ellipse(small, (214, 168, 136), face)
        hair = pygame.Rect(0, 0, int(face_w * 1.08), int(face_h * 0.62))
        hair.midtop = (hx, face.top - int(face_h * 0.06))
        pygame.draw.ellipse(small, (52, 38, 32), hair)
    soft = pygame.transform.smoothscale(small, (sw // 2, sh // 2))
    soft = pygame.transform.smoothscale(soft, (sw, sh))
    full = pygame.transform.smoothscale(soft, (w, h))
    return np.ascontiguousarray(pygame.surfarray.array3d(full).swapaxes(0, 1))


def face_box(head: Point, frame: Tuple[int, int]) -> Tuple[float, float, float, float]:
    box_h = 0.24
    box_w = box_h * 0.8 * frame[1] / frame[0]
    return (head[0] - box_w / 2, head[1] - box_h / 2, box_w, box_h)


# -------------------------------------------------------------- the harness


class Harness:
    def __init__(self, size: Tuple[int, int], db: Path, seed_board: bool = True) -> None:
        self.size = size
        self.frame_size = (1280, 720)
        self.surface = pygame.display.get_surface() or pygame.Surface(size)
        self.clock = FakeClock()
        self.camera = StubCamera()
        self.store = Store(db)
        if seed_board and not self.store.top(1):
            for name, seconds in (("MAYA", 30.0), ("DEV", 23.4), ("PRIYA", 19.7),
                                  ("JO", 12.0), ("KAI", 8.3)):
                self.store.add_entry(name, int(seconds * 10))
        self.game = g.Game(self.store, self.camera, seed=4, clock=self.clock)
        self.draw_ms: List[float] = []
        self.set_head(None)

    def set_head(self, head: Optional[Point], person: Optional[Point] = "same",
                 detected: bool = True) -> None:
        """Place the person in the frame and (optionally) report the head."""
        shown = head if person == "same" else person
        frame = synthetic_frame(self.frame_size, shown)
        self.camera.detection = HeadDetection(
            frame=frame, head=head if detected else None,
            box=face_box(head, self.frame_size) if (head and detected) else None,
            timestamp=self.clock.t, ok=True, confidence=0.9, mode=MODE_FACE, fps=30.0,
            searching=head is None)

    def track(self, head: Optional[Point]) -> None:
        """Move the reported head without re-rendering the frame (cheap)."""
        d = self.camera.detection
        self.camera.detection = HeadDetection(
            frame=d.frame, head=head,
            box=face_box(head, self.frame_size) if head else None,
            timestamp=d.timestamp, ok=True, confidence=0.9, mode=MODE_FACE, fps=30.0,
            searching=head is None)

    def step(self, seconds: float, fps: int = 30,
             each: Optional[Callable[[], None]] = None) -> None:
        steps = max(1, int(round(seconds * fps)))
        for _ in range(steps):
            self.clock.t += 1.0 / fps
            if each:
                each()
            self.game.update()
            started = time.perf_counter()
            self.game.draw(self.surface)
            self.draw_ms.append((time.perf_counter() - started) * 1000)

    def key(self, key: int, unicode: str = "") -> None:
        self.game.handle(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=unicode,
                                            mod=0))

    def type(self, text: str) -> None:
        for char in text:
            self.key(pygame.key.key_code(char.lower()) if char != " " else pygame.K_SPACE,
                     char)

    def save(self, out: Path, name: str) -> Path:
        path = out / f"{name}.png"
        pygame.image.save(self.surface, str(path))
        return path

    def close(self) -> None:
        self.store.close()


def follow_square(h: Harness, offset: Point = (0.0, 0.0)) -> Callable[[], None]:
    def each() -> None:
        sq = h.game.run.square
        h.track((sq.x + offset[0], sq.y + offset[1]))
    return each


# ------------------------------------------------------------------ scenes


def render_all(out: Path, size: Tuple[int, int]) -> List[Path]:
    tag = f"{size[0]}x{size[1]}"
    written: List[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="searchlight-render-"))

    def fresh(name: str, seed_board: bool = True) -> Harness:
        return Harness(size, tmp / f"{name}.db", seed_board)

    def save(h: Harness, name: str) -> None:
        written.append(h.save(out, f"{tag}_{name}"))

    # ---- attract
    h = fresh("attract")
    h.set_head((0.55, 0.42))
    h.step(1.5)
    save(h, "01_attract")
    if h.game.open_settings():
        h.step(0.5)
        save(h, "02_attract_settings")
    else:
        print("settings menu not importable - skipped attract_settings")
    h.close()

    h = fresh("attract_empty", seed_board=False)
    h.camera.error = None
    h.step(1.2)
    save(h, "01b_attract_empty_board")
    h.close()

    # ---- name entry
    h = fresh("name")
    h.set_head((0.5, 0.45))
    h.step(0.5)
    h.key(pygame.K_RETURN)
    h.type("SAM")
    h.step(0.8)
    save(h, "03_name_entry")
    for _ in range(3):
        h.key(pygame.K_BACKSPACE)
    h.type("JO")
    h.step(0.3)
    save(h, "04_name_entry_welcome_back")

    # ---- find player
    h.key(pygame.K_RETURN)
    h.set_head(None)
    h.step(1.0)
    save(h, "05_find_nobody")
    h.set_head((0.24, 0.40))
    h.step(0.8)
    save(h, "06_find_outside")
    h.set_head((0.51, 0.47))
    h.step(0.65)
    save(h, "07_find_locking")

    # ---- countdown
    h.step(0.6)
    assert h.game.screen == g.COUNTDOWN, h.game.screen
    h.step(0.15)
    save(h, "08_countdown")

    # ---- playing
    h.step(h.game.settings.countdown_seconds, each=follow_square(h, (0.01, 0.0)))
    assert h.game.screen == g.PLAYING, h.game.screen
    h.draw_ms.clear()
    h.step(11.0, each=follow_square(h, (0.01, -0.01)))
    timings = sorted(h.draw_ms)
    print(f"playing draw: median {timings[len(timings) // 2]:.1f} ms, "
          f"p95 {timings[int(len(timings) * 0.95)]:.1f} ms over {len(timings)} frames")
    sq = h.game.run.square
    h.set_head((sq.x + 0.01, sq.y - 0.01))
    h.step(0.05, each=follow_square(h, (0.01, -0.01)))
    save(h, "09_playing")
    # Drift out of the square: danger rises through the grace period.
    grace = h.game.run.config.grace_seconds
    out_by = sq.half_w + 0.06
    h.set_head((sq.x + out_by, sq.y))
    h.step(grace * 0.7, fps=60, each=follow_square(h, (out_by, 0.0)))
    save(h, "10_playing_danger")
    h.step(grace * 0.3 + 0.12, fps=60, each=follow_square(h, (out_by, 0.0)))
    save(h, "11_loss_flash")
    h.step(0.65, each=follow_square(h, (out_by, 0.0)))
    h.set_head((sq.x + out_by, sq.y))
    h.step(0.02)
    save(h, "12_playing_respawn")
    # Back in: the short countdown before the square moves again.
    h.set_head((sq.x + 0.01, sq.y))
    h.step(0.3)
    save(h, "12b_playing_resume")
    h.close()

    # ---- results
    for name, player, seconds, file in (("new_top", "SAM", 42.3, "13_result_new_top"),
                                        ("pb", "JO", 18.4, "14_result_personal_best"),
                                        ("not_best", "DEV", 9.8, "15_result_not_best"),
                                        ("first", "NEWBIE", 6.1, "16_result_first_run")):
        h = fresh(name)
        h.set_head((0.5, 0.45))
        h.game.name = player
        h.game.go(g.FIND_PLAYER)
        h.game.go(g.PLAYING, None)
        h.step(0.2)
        h.game.run.started_at = h.clock.t - seconds
        h.game._finish_run(h.clock.t)
        h.step(1.3)
        save(h, file)
        h.close()

    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for the PNGs")
    parser.add_argument("--size", default="1280x720", help="WxH, e.g. 1920x1080")
    args = parser.parse_args(argv)

    size = tuple(int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.display.set_mode(size)
    started = time.perf_counter()
    written = render_all(out, size)  # type: ignore[arg-type]
    for path in written:
        print(path)
    print(f"{len(written)} screens in {time.perf_counter() - started:.1f}s")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
