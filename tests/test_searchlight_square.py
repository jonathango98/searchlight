"""The wandering square, stepped through simulated time."""

from searchlight.square import Square, SquareConfig


def run_for(square, seconds, dt=1 / 60):
    t = 0.0
    while t < seconds:
        t += dt
        square.update(dt, t)
        yield t


def test_starts_centred_at_full_size():
    square = Square(seed=1)
    assert (square.x, square.y) == (0.5, 0.5)
    assert square.side == square.config.start_side
    assert square.contains(0.5, 0.5)


def test_never_leaves_the_frame():
    square = Square(SquareConfig(max_speed=1.5, speed_ramp=0.1), seed=2)
    margin = square.config.margin
    for _ in run_for(square, 180):
        left, top, right, bottom = square.bounds()
        assert left >= margin / square.config.aspect - 1e-9
        assert right <= 1 - margin / square.config.aspect + 1e-9
        assert top >= margin - 1e-9
        assert bottom <= 1 - margin + 1e-9


def test_actually_moves():
    square = Square(seed=3)
    for _ in run_for(square, 5):
        pass
    assert (square.x, square.y) != (0.5, 0.5)


def test_speeds_up_and_shrinks_within_caps():
    cfg = SquareConfig()
    square = Square(cfg, seed=4)
    for _ in run_for(square, 10):
        pass
    early_side, early_speed = square.side, square.speed
    assert early_side < cfg.start_side and early_speed > cfg.start_speed
    for _ in run_for(square, 1000, dt=0.5):
        pass
    assert square.side == cfg.min_side
    assert square.speed == cfg.max_speed


def test_same_seed_same_path():
    a, b = Square(seed=9), Square(seed=9)
    for _ in run_for(a, 20):
        pass
    for _ in run_for(b, 20):
        pass
    assert (a.x, a.y) == (b.x, b.y)


def test_square_is_square_on_screen():
    """In normalised x units it is narrower, so on a 16:9 frame it draws square."""
    square = Square(seed=5)
    assert abs(square.half_w * square.config.aspect - square.half_h) < 1e-12
