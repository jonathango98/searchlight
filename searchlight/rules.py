"""One run of Searchlight: lives, grace, respawn and the clock.

Pure, like the square it drives: feed it timestamps and head positions and it says
what happened, so every rule can be tested without a camera.

The rules:

* The head has to be inside the square. A head the tracker cannot find counts as
  outside - otherwise ducking out of frame would be the safest place to stand.
* Leaving is forgiven for `grace_seconds`. Face tracking drops the odd frame and
  jitters at the edges, and losing a life to a dropped frame feels like a bug.
* Staying out longer costs a life and pauses the run: the square holds still and
  the clock stops until the player is back inside. Then a `respawn_seconds`
  countdown runs, and the square moves again when it ends. Leaving for longer than
  the grace time during that countdown cancels it - back to waiting, no life lost.
* No lives left ends the run. The score is time survived, not counting pauses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .square import Square

Point = Tuple[float, float]


@dataclass
class RunConfig:
    lives: int = 3
    grace_seconds: float = 0.35
    respawn_seconds: float = 2.0


@dataclass(frozen=True)
class RunResult:
    inside: bool
    lives: int
    survived: float
    # How far through the grace period the player is, 0-1; drawn as a warning.
    danger: float
    # Paused after a lost life: waiting for the player, then counting down.
    respawning: bool
    # Seconds left on the countdown back to play; 0 while still waiting for the
    # player to get back in the square.
    resume_in: float
    life_lost: bool
    # The countdown just finished and the square is moving again.
    resumed: bool
    over: bool

    @property
    def waiting(self) -> bool:
        return self.respawning and self.resume_in <= 0


class Run:
    def __init__(self, square: Square, config: Optional[RunConfig] = None) -> None:
        self.square = square
        self.config = config or RunConfig()
        self.start(0.0)

    def start(self, t: float) -> None:
        self.square.reset()
        self.lives = self.config.lives
        self.started_at = t
        self.ended_at: Optional[float] = None
        self._last_t = t
        self._outside_since: Optional[float] = None
        self._paused_at: Optional[float] = None      # set while respawning
        self._paused_total = 0.0
        self._resume_at: Optional[float] = None      # set while counting down

    @property
    def over(self) -> bool:
        return self.ended_at is not None

    @property
    def respawning(self) -> bool:
        return self._paused_at is not None

    def survived(self, t: float) -> float:
        end = t
        if self.ended_at is not None:
            end = self.ended_at
        elif self._paused_at is not None:
            end = self._paused_at
        return max(0.0, end - self.started_at - self._paused_total)

    def update(self, t: float, head: Optional[Point]) -> RunResult:
        life_lost = resumed = False
        danger = 0.0

        if self.respawning:
            resumed = self._update_respawn(t, head)
        dt = max(0.0, t - self._last_t)
        self._last_t = t
        if not self.over and not self.respawning:
            self.square.update(dt, self.survived(t))

        inside = head is not None and self.square.contains(*head)

        if not self.over and not self.respawning:
            if inside:
                self._outside_since = None
            else:
                if self._outside_since is None:
                    self._outside_since = t
                away = t - self._outside_since
                danger = min(1.0, away / self.config.grace_seconds)
                if away > self.config.grace_seconds:
                    life_lost = True
                    danger = 0.0
                    self._lose_life(t)

        resume_in = 0.0
        if self._resume_at is not None:
            resume_in = max(0.0, self._resume_at - t)
        return RunResult(inside=inside, lives=self.lives, survived=self.survived(t),
                         danger=danger, respawning=self.respawning, resume_in=resume_in,
                         life_lost=life_lost, resumed=resumed, over=self.over)

    def _lose_life(self, t: float) -> None:
        self.lives -= 1
        self._outside_since = None
        if self.lives <= 0:
            self.lives = 0
            self.ended_at = t
        else:
            self._paused_at = t

    def _update_respawn(self, t: float, head: Optional[Point]) -> bool:
        """Wait for the head to come back, count down, resume. True on resuming."""
        inside = head is not None and self.square.contains(*head)
        if inside:
            self._outside_since = None
            if self._resume_at is None:
                self._resume_at = t + self.config.respawn_seconds
        elif self._resume_at is not None:
            # Wandering off mid-countdown is forgiven for the grace time, like in
            # play; any longer and the countdown starts over once they are back.
            if self._outside_since is None:
                self._outside_since = t
            if t - self._outside_since > self.config.grace_seconds:
                self._resume_at = None
                self._outside_since = None

        if self._resume_at is None or t < self._resume_at:
            return False
        # Resume from the moment the countdown ended, so no time is lost or gained
        # to the frame boundary. A head that has just slipped out keeps its grace
        # timer running into play.
        assert self._paused_at is not None
        self._paused_total += self._resume_at - self._paused_at
        self._last_t = self._resume_at
        self._paused_at = self._resume_at = None
        return True
