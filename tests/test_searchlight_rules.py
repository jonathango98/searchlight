"""Lives, grace and respawn, driven by synthetic head positions."""

import pytest

from searchlight.rules import Run, RunConfig
from searchlight.square import Square, SquareConfig

DT = 1 / 30
# A square that never moves or shrinks, so the tests are about the rules alone.
STILL = SquareConfig(start_speed=0, max_speed=0, speed_ramp=0, shrink_per_second=0)
INSIDE = (0.5, 0.5)
OUTSIDE = (0.02, 0.02)


@pytest.fixture
def run():
    r = Run(Square(STILL, seed=0), RunConfig(lives=3, grace_seconds=0.35,
                                             respawn_seconds=2.0))
    r.start(0.0)
    return r


def feed(run, start, seconds, head):
    t, results = start, []
    while t < start + seconds - 1e-9:
        t += DT
        results.append(run.update(t, head))
    return t, results


def test_staying_inside_never_costs_a_life(run):
    _, results = feed(run, 0.0, 30, INSIDE)
    assert all(r.inside for r in results)
    assert results[-1].lives == 3
    assert results[-1].survived == pytest.approx(30, abs=DT)


def test_a_blip_shorter_than_grace_is_forgiven(run):
    t, _ = feed(run, 0.0, 1, INSIDE)
    t, blip = feed(run, t, 0.2, OUTSIDE)
    assert 0 < blip[-1].danger < 1
    t, back = feed(run, t, 1, INSIDE)
    assert back[-1].lives == 3 and back[-1].danger == 0


def test_a_sustained_exit_costs_exactly_one_life(run):
    _, results = feed(run, 0.0, 1.0, OUTSIDE)
    assert sum(r.life_lost for r in results) == 1
    assert results[-1].lives == 2
    assert results[-1].respawning


def test_missing_head_counts_as_outside(run):
    _, results = feed(run, 0.0, 1.0, None)
    assert results[-1].lives == 2


def lose_life_and_return(run, t):
    """Step out until a life goes, come back and sit out the countdown."""
    while not run.update(t := t + DT, OUTSIDE).life_lost:
        pass
    t, back = feed(run, t, 2.0 + 2 * DT, INSIDE)
    assert back[-1].resumed or not back[-1].respawning
    return t


def test_after_a_lost_life_the_run_waits_for_the_player(run):
    t, _ = feed(run, 0.0, 0.5, OUTSIDE)            # life lost at ~0.37s
    t, waiting = feed(run, t, 20, OUTSIDE)
    assert all(r.respawning and r.waiting for r in waiting)
    assert not any(r.life_lost for r in waiting)
    assert waiting[-1].lives == 2


def test_the_clock_stops_while_paused(run):
    t, before = feed(run, 0.0, 0.5, OUTSIDE)
    paused_at = before[-1].survived
    t, waiting = feed(run, t, 5, None)
    t, counting = feed(run, t, 1.9, INSIDE)
    assert all(r.survived == paused_at for r in waiting + counting)
    t, after = feed(run, t, 1.0, INSIDE)
    assert not after[-1].respawning
    assert after[-1].survived == pytest.approx(paused_at + 0.9, abs=2 * DT)


def test_coming_back_starts_a_countdown_then_play_resumes(run):
    t, _ = feed(run, 0.0, 0.5, OUTSIDE)
    t, _ = feed(run, t, 1.0, OUTSIDE)
    t, counting = feed(run, t, 1.9, INSIDE)
    assert counting[0].resume_in == pytest.approx(2.0, abs=DT)
    assert all(r.respawning and not r.waiting for r in counting)
    remaining = [r.resume_in for r in counting]
    assert remaining == sorted(remaining, reverse=True)
    t, after = feed(run, t, 0.5, INSIDE)
    assert sum(r.resumed for r in after) == 1
    assert not after[-1].respawning and after[-1].lives == 2


def test_leaving_during_the_countdown_resets_it_without_costing_a_life(run):
    t, _ = feed(run, 0.0, 0.5, OUTSIDE)
    t, _ = feed(run, t, 1.5, INSIDE)                # most of the countdown
    t, blip = feed(run, t, 0.2, OUTSIDE)            # within grace: still counting
    assert not blip[-1].waiting
    t, gone = feed(run, t, 1.0, OUTSIDE)            # past grace: back to waiting
    assert gone[-1].waiting and gone[-1].lives == 2
    t, again = feed(run, t, 0.1, INSIDE)
    assert again[-1].resume_in == pytest.approx(2.0, abs=0.1)


def test_play_resumes_where_the_player_left_off(run):
    t = lose_life_and_return(run, 0.0)
    t, out = feed(run, t, 1.0, OUTSIDE)
    assert sum(r.life_lost for r in out) == 1
    assert out[-1].lives == 1


def test_square_holds_still_while_respawning():
    moving = SquareConfig(start_speed=0.3, max_speed=0.3, speed_ramp=0)
    r = Run(Square(moving, seed=1), RunConfig())
    r.start(0.0)
    t, _ = feed(r, 0.0, 0.4, OUTSIDE)
    assert r.update(t + DT, OUTSIDE).respawning
    where = (r.square.x, r.square.y)
    feed(r, t + DT, 1.0, OUTSIDE)
    feed(r, t + 1.0 + DT, 1.5, where)               # counting down, not yet moving
    assert (r.square.x, r.square.y) == where


def test_game_over_freezes_the_clock(run):
    t = 0.0
    for _ in range(2):
        t = lose_life_and_return(run, t)
    t, results = feed(run, t, 2, OUTSIDE)
    over = [r for r in results if r.over]
    assert over and over[0].lives == 0
    assert over[0].life_lost
    assert results[-1].survived == over[0].survived


def test_survived_time_never_goes_backwards(run):
    t, results = 0.0, []
    for _ in range(2):
        t, part = feed(run, t, 1.0, OUTSIDE)
        results += part
        t, part = feed(run, t, 3.0, INSIDE)
        results += part
    times = [r.survived for r in results]
    assert times == sorted(times)
