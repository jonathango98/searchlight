"""Every tunable parameter, defined once.

`SETTINGS` is the schema: key, label, range, step, help text and when a change
takes effect. The in-game settings menu builds its controls from it, the store
persists the values, and `Settings` turns them into the configs the pure game
modules want. Adding a knob means adding one `Setting` here plus the matching
field on `Settings` - nothing else has to learn about it.

Pure: no pygame, OpenCV or MediaPipe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from .rules import RunConfig
from .square import SquareConfig
from .tracking.base import MODE_FACE, MODE_UPPER_BODY, CameraOptions

# When a change takes effect - shown in the menu so nobody waits for a change
# that will only happen on the next run.
LIVE = "live"            # immediately
NEXT_RUN = "next_run"    # at the start of the next run, never mid-run

INT, FLOAT, BOOL, CHOICE = "int", "float", "bool", "choice"

# How a number is shown. "percent" multiplies by 100: sizes and speeds are in
# frame heights, and "42%" reads better than "0.42".
PLAIN, PERCENT, SECONDS = "plain", "percent", "seconds"

GROUPS: Tuple[str, ...] = ("Gameplay", "Square", "Detection", "Recording", "Leaderboard")


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    group: str
    kind: str                                   # INT | FLOAT | BOOL | CHOICE
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    # For CHOICE: ((stored value, label shown), ...)
    choices: Tuple[Tuple[str, str], ...] = ()
    help: str = ""
    applies: str = NEXT_RUN
    display: str = PLAIN
    unit: str = ""                              # appended after the number

    def clamp(self, value: Any) -> Any:
        """Coerce and clamp `value` to something valid; fall back to the default."""
        try:
            if self.kind == BOOL:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.kind == CHOICE:
                valid = [v for v, _ in self.choices]
                return value if value in valid else self.default
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        if number != number:                     # NaN
            return self.default
        if self.min is not None:
            number = max(self.min, number)
        if self.max is not None:
            number = min(self.max, number)
        if self.kind == INT:
            return int(round(number))
        # Snap to the step grid so the menu never shows 0.35000000001.
        if self.step:
            base = self.min or 0.0
            number = base + round((number - base) / self.step) * self.step
            number = min(self.max, number) if self.max is not None else number
        return round(number, 6)

    def nudge(self, value: Any, steps: int) -> Any:
        """`value` moved by `steps` (+/-) - arrow keys and slider clicks use this."""
        if self.kind == BOOL:
            return (not value) if steps % 2 else value
        if self.kind == CHOICE:
            valid = [v for v, _ in self.choices]
            index = valid.index(value) if value in valid else 0
            return valid[(index + steps) % len(valid)]
        return self.clamp(value + steps * (self.step or 1))

    def format(self, value: Any) -> str:
        """The value as the menu should show it."""
        if self.kind == BOOL:
            return "ON" if value else "OFF"
        if self.kind == CHOICE:
            return dict(self.choices).get(value, str(value))
        if self.display == PERCENT:
            shown = value * 100
            text = f"{shown:.1f}" if abs(shown - round(shown)) > 1e-6 else f"{shown:.0f}"
            return f"{text}%{self.unit}"
        if self.display == SECONDS:
            return f"{value:.2f}".rstrip("0").rstrip(".") + "s"
        if self.kind == INT:
            return f"{value}{self.unit}"
        return f"{value:g}{self.unit}"

    def fraction(self, value: Any) -> float:
        """Where `value` sits between min and max, 0-1, for drawing a slider."""
        if self.min is None or self.max is None or self.max == self.min:
            return 0.0
        return max(0.0, min(1.0, (float(value) - self.min) / (self.max - self.min)))

    def from_fraction(self, fraction: float) -> Any:
        """Inverse of `fraction` - for dragging a slider."""
        if self.min is None or self.max is None:
            return self.default
        return self.clamp(self.min + fraction * (self.max - self.min))


SETTINGS: Tuple[Setting, ...] = (
    # ---------------------------------------------------------------- gameplay
    Setting("lives", "Lives", "Gameplay", INT, 3, 1, 9, 1,
            help="How many times a player can leave the light before the run ends."),
    Setting("grace_seconds", "Grace time", "Gameplay", FLOAT, 0.35, 0.10, 2.00, 0.05,
            display=SECONDS,
            help="How long the head may be outside the square before a life is lost."),
    Setting("respawn_seconds", "Respawn countdown", "Gameplay", INT, 2, 1, 5, 1, unit="s",
            help="After losing a life the run pauses until the player is back in the "
                 "square, then counts down this long before the square moves again."),
    Setting("face_lock_seconds", "Lock-on time", "Gameplay", FLOAT, 1.0, 0.3, 3.0, 0.1,
            display=SECONDS,
            help="How long a head must stay in the start square before the countdown."),
    Setting("countdown_seconds", "Countdown", "Gameplay", INT, 3, 1, 5, 1, unit="s",
            help="Length of the 3-2-1 before the square starts moving."),

    # ------------------------------------------------------------------ square
    Setting("start_side", "Starting size", "Square", FLOAT, 0.42, 0.20, 0.70, 0.02,
            display=PERCENT, help="Square side at the start, as a share of frame height."),
    Setting("min_side", "Smallest size", "Square", FLOAT, 0.16, 0.08, 0.50, 0.01,
            display=PERCENT, help="The square never shrinks below this."),
    Setting("shrink_per_second", "Shrink rate", "Square", FLOAT, 0.004, 0.0, 0.02, 0.001,
            display=PERCENT, unit="/s", help="How fast the square shrinks as the run goes on."),
    Setting("start_speed", "Starting speed", "Square", FLOAT, 0.10, 0.0, 0.80, 0.02,
            display=PERCENT, unit="/s",
            help="How fast the square moves at the start (frame heights per second)."),
    Setting("max_speed", "Top speed", "Square", FLOAT, 0.45, 0.05, 1.20, 0.05,
            display=PERCENT, unit="/s", help="The square never moves faster than this."),
    Setting("speed_ramp", "Speed-up rate", "Square", FLOAT, 0.006, 0.0, 0.03, 0.001,
            display=PERCENT, unit="/s²", help="How quickly the square gets faster."),
    Setting("turn_rate", "Wobble", "Square", FLOAT, 1.6, 0.0, 4.0, 0.1,
            help="How sharply the square changes direction. 0 = straight lines."),

    # --------------------------------------------------------------- detection
    Setting("mode", "Detect", "Detection", CHOICE, MODE_FACE,
            choices=((MODE_FACE, "Face"), (MODE_UPPER_BODY, "Upper body")),
            applies=LIVE,
            help="Face is precise up close. Upper body works from further away "
                 "but needs the shoulders in view."),
    Setting("long_range", "Long-range assist", "Detection", BOOL, True, applies=LIVE,
            help="Zooms in around the player so small, distant faces are still found."),
    Setting("zoom", "Camera zoom", "Detection", FLOAT, 1.0, 1.0, 3.0, 0.1, applies=LIVE,
            unit="x", help="Crops into the middle of the camera. Use when players stand far back."),
    Setting("min_confidence", "Detection confidence", "Detection", FLOAT, 0.5, 0.2, 0.9,
            0.05, applies=LIVE,
            help="Lower finds more faces but may mistake other things for one."),
    Setting("smoothing", "Smoothing", "Detection", FLOAT, 0.4, 0.0, 0.9, 0.05,
            applies=LIVE, help="Steadies a jittery head marker. Higher adds a little lag."),
    Setting("resolution", "Camera resolution", "Detection", CHOICE, "1280x720",
            choices=(("960x540", "960 x 540"), ("1280x720", "1280 x 720"),
                     ("1920x1080", "1920 x 1080")),
            applies=LIVE, help="Higher helps distant players but costs frame rate."),
    Setting("camera_index", "Camera", "Detection", INT, 0, 0, 5, 1, applies=LIVE,
            help="Which camera to use, if more than one is connected."),
    Setting("mirror", "Mirror image", "Detection", BOOL, True, applies=LIVE,
            help="Flip the picture so moving left moves you left on screen."),
    Setting("show_tracking", "Show tracking box", "Detection", BOOL, True, applies=LIVE,
            help="Outline what the camera thinks is the player's head."),

    # --------------------------------------------------------------- recording
    Setting("record_runs", "Record runs", "Recording", BOOL, False,
            help="Save the raw camera video of each run to the recordings folder."),

    # ------------------------------------------------------------- leaderboard
    Setting("board_rows", "Rows shown", "Leaderboard", INT, 10, 5, 15, 1, applies=LIVE,
            help="How many players the leaderboard screen lists."),
)

SETTINGS_BY_KEY: Dict[str, Setting] = {s.key: s for s in SETTINGS}


def settings_in(group: str) -> Tuple[Setting, ...]:
    return tuple(s for s in SETTINGS if s.group == group)


@dataclass(frozen=True)
class Settings:
    """A complete, valid set of values. Build with `from_dict` or `with_changes`."""

    lives: int = 3
    grace_seconds: float = 0.35
    respawn_seconds: int = 2
    face_lock_seconds: float = 1.0
    countdown_seconds: int = 3

    start_side: float = 0.42
    min_side: float = 0.16
    shrink_per_second: float = 0.004
    start_speed: float = 0.10
    max_speed: float = 0.45
    speed_ramp: float = 0.006
    turn_rate: float = 1.6

    mode: str = MODE_FACE
    long_range: bool = True
    zoom: float = 1.0
    min_confidence: float = 0.5
    smoothing: float = 0.4
    resolution: str = "1280x720"
    camera_index: int = 0
    mirror: bool = True
    show_tracking: bool = True

    record_runs: bool = False

    board_rows: int = 10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Settings":
        """Clamp every known key; ignore unknown ones; default missing ones."""
        values = {}
        for spec in SETTINGS:
            values[spec.key] = spec.clamp(data[spec.key]) if spec.key in data else spec.default
        return cls(**values)._consistent()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_changes(self, **changes: Any) -> "Settings":
        """A copy with some values changed - clamped, like `from_dict`."""
        unknown = set(changes) - set(SETTINGS_BY_KEY)
        if unknown:
            raise KeyError(f"unknown settings: {sorted(unknown)}")
        merged = self.to_dict()
        merged.update(changes)
        return Settings.from_dict(merged)

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def changed_keys(self, other: "Settings") -> set:
        return {f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)}

    def _consistent(self) -> "Settings":
        """Fix pairs that are each valid alone but contradict each other."""
        fixed = self
        if fixed.min_side > fixed.start_side:
            fixed = replace(fixed, min_side=fixed.start_side)
        if fixed.max_speed < fixed.start_speed:
            fixed = replace(fixed, max_speed=fixed.start_speed)
        return fixed

    # ---------------------------------------------------- configs for the game

    def square_config(self, aspect: float = 16 / 9) -> SquareConfig:
        return SquareConfig(aspect=aspect, start_side=self.start_side,
                            min_side=self.min_side,
                            shrink_per_second=self.shrink_per_second,
                            start_speed=self.start_speed, max_speed=self.max_speed,
                            speed_ramp=self.speed_ramp, turn_rate=self.turn_rate)

    def run_config(self) -> RunConfig:
        return RunConfig(lives=self.lives, grace_seconds=self.grace_seconds,
                         respawn_seconds=self.respawn_seconds)

    def camera_options(self) -> CameraOptions:
        width, height = (int(v) for v in self.resolution.split("x"))
        return CameraOptions(camera_index=self.camera_index, width=width, height=height,
                             mode=self.mode, long_range=self.long_range, zoom=self.zoom,
                             min_confidence=self.min_confidence,
                             smoothing=self.smoothing, mirror=self.mirror)


DEFAULTS = Settings()

# Presets touch difficulty only - never detection or display.
PRESETS: Dict[str, Dict[str, Any]] = {
    "Easy": dict(lives=5, grace_seconds=0.6, start_side=0.50, min_side=0.24,
                 shrink_per_second=0.003, start_speed=0.08, max_speed=0.30,
                 speed_ramp=0.004, turn_rate=1.2),
    "Normal": {k: getattr(DEFAULTS, k) for k in (
        "lives", "grace_seconds", "start_side", "min_side", "shrink_per_second",
        "start_speed", "max_speed", "speed_ramp", "turn_rate")},
    "Hard": dict(lives=2, grace_seconds=0.25, start_side=0.36, min_side=0.12,
                 shrink_per_second=0.006, start_speed=0.16, max_speed=0.60,
                 speed_ramp=0.009, turn_rate=2.0),
}


def apply_preset(settings: Settings, name: str) -> Settings:
    return settings.with_changes(**PRESETS[name])


def matching_preset(settings: Settings) -> Optional[str]:
    """The preset `settings` currently equals, if any - so the menu can mark it."""
    for name, values in PRESETS.items():
        if all(abs(getattr(settings, k) - v) < 1e-9 for k, v in values.items()):
            return name
    return None
