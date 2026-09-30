"""The pure tracking geometry: crops, tiles, remapping, smoothing, target choice."""

import math

import pytest

from searchlight.tracking.base import Target
from searchlight.tracking.geometry import (
    FpsMeter, OneEuroFilter, TargetSmoother, box_iou, boxes_agree, challengers, choose_target,
    clamp_rect, crop_to_frame_box, crop_to_frame_point, crop_to_frame_target, dedupe,
    frame_to_crop_box, frame_to_crop_point, rect_covers, roi_rect, search_tiles,
    smoothing_cutoff, zoom_rect)

W, H = 1280, 720


def inside(rect, w=W, h=H):
    x, y, rw, rh = rect
    return x >= 0 and y >= 0 and x + rw <= w and y + rh <= h and rw > 0 and rh > 0


# ------------------------------------------------------------------- rects

def test_clamp_rect_slides_instead_of_shrinking():
    assert clamp_rect(10, 10, 200, 100, W, H) == (0, 0, 200, 100)
    assert clamp_rect(W, H, 200, 100, W, H) == (W - 200, H - 100, 200, 100)


def test_clamp_rect_shrinks_only_when_larger_than_frame():
    assert clamp_rect(640, 360, 5000, 5000, W, H) == (0, 0, W, H)


def test_roi_is_square_scaled_and_centred():
    box = (0.5, 0.5, 40 / W, 40 / H)                     # 40 px face at (660, 380)
    x, y, w, h = roi_rect(box, W, H, scale=4.0, min_side=64)
    assert (w, h) == (160, 160)
    assert (x + w / 2, y + h / 2) == pytest.approx((660, 380), abs=1)


def test_roi_respects_min_side():
    tiny = (0.5, 0.5, 5 / W, 5 / H)
    assert roi_rect(tiny, W, H, scale=4.0, min_side=128)[2:] == (128, 128)


@pytest.mark.parametrize("box", [(0.0, 0.0, 0.05, 0.08), (0.97, 0.95, 0.05, 0.08),
                                 (-0.02, 0.5, 0.05, 0.08), (0.4, 0.2, 0.9, 0.9)])
def test_roi_is_clamped_at_frame_edges(box):
    assert inside(roi_rect(box, W, H))


def test_roi_at_edge_keeps_the_face_inside():
    box = (0.96, 0.9, 0.04, 0.07)                        # face in the bottom-right corner
    x, y, w, h = roi_rect(box, W, H)
    assert inside((x, y, w, h))
    assert x <= box[0] * W and (box[0] + box[2]) * W <= x + w + 1


def test_rect_covers():
    assert rect_covers((0, 0, W, H), W, H)
    assert not rect_covers((0, 0, H, H), W, H)          # a full-height square on 16:9


# ------------------------------------------------------------------- tiles

def test_tiles_lie_inside_and_are_square():
    tiles = search_tiles(W, H)
    assert tiles and all(inside(t) and t[2] == t[3] for t in tiles)


def test_tiles_cover_the_width_with_overlap():
    for side in {t[2] for t in search_tiles(W, H)}:
        row = sorted(t for t in search_tiles(W, H) if t[2] == side)
        xs = sorted({t[0] for t in row})
        assert xs[0] == 0 and xs[-1] + side == W
        assert all(b - a < side for a, b in zip(xs, xs[1:]))   # neighbours overlap


def test_tiles_have_two_zoom_levels_and_three_coarse():
    tiles = search_tiles(W, H)
    sides = sorted({t[2] for t in tiles})
    assert len(sides) == 2 and sides[0] == sides[1] // 2
    assert sum(t[2] == sides[1] for t in tiles) == 3
    assert search_tiles(W, H, fine_rows=0) == [t for t in tiles if t[2] == sides[1]]


def test_tiles_cover_the_head_band():
    tiles = search_tiles(W, H)
    top = min(t[1] for t in tiles)
    bottom = max(t[1] + t[3] for t in tiles)
    assert top <= 0.15 * H and bottom >= 0.65 * H


def test_tiles_on_small_and_portrait_frames():
    for w, h in [(640, 360), (360, 640), (200, 200)]:
        assert all(inside(t, w, h) for t in search_tiles(w, h))


# -------------------------------------------------------------------- zoom

def test_zoom_one_is_the_whole_frame():
    assert zoom_rect(W, H, 1.0) == (0, 0, W, H)
    assert zoom_rect(W, H, 0.5) == (0, 0, W, H)


def test_zoom_two_is_the_centre_half_same_aspect():
    x, y, w, h = zoom_rect(W, H, 2.0)
    assert (x, y, w, h) == (320, 180, 640, 360)
    assert w / h == pytest.approx(W / H)


def test_odd_zoom_stays_inside():
    rect = zoom_rect(W, H, 2.7)
    assert inside(rect) and rect[2] / rect[3] == pytest.approx(W / H, rel=0.01)


# ----------------------------------------------------------------- remapping

CROPS = [(0, 0, W, H), (100, 50, 256, 256), (1024, 464, 256, 256), (320, 180, 640, 360)]


@pytest.mark.parametrize("crop", CROPS)
def test_point_round_trip(crop):
    for point in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.13, 0.87)]:
        there = crop_to_frame_point(point, crop, W, H)
        assert frame_to_crop_point(there, crop, W, H) == pytest.approx(point)


@pytest.mark.parametrize("crop", CROPS)
def test_box_round_trip(crop):
    box = (0.2, 0.3, 0.25, 0.1)
    there = crop_to_frame_box(box, crop, W, H)
    assert frame_to_crop_box(there, crop, W, H) == pytest.approx(box)


def test_crop_corners_map_to_the_crop_in_the_frame():
    crop = (100, 50, 256, 256)
    assert crop_to_frame_point((0, 0), crop, W, H) == pytest.approx((100 / W, 50 / H))
    assert crop_to_frame_point((1, 1), crop, W, H) == pytest.approx((356 / W, 306 / H))
    assert crop_to_frame_box((0, 0, 1, 1), crop, W, H) == pytest.approx(
        (100 / W, 50 / H, 256 / W, 256 / H))


def test_target_remap_moves_head_and_box_together():
    crop = (640, 0, 320, 320)
    t = crop_to_frame_target(Target(head=(0.5, 0.5), box=(0.25, 0.25, 0.5, 0.5),
                                    confidence=0.7), crop, W, H)
    assert t.head == pytest.approx((800 / W, 160 / H))
    assert t.box == pytest.approx((720 / W, 80 / H, 160 / W, 160 / H))
    assert t.confidence == 0.7


def test_iou_and_dedupe():
    a, b, far = (0, 0, 0.2, 0.2), (0.01, 0.01, 0.2, 0.2), (0.6, 0.6, 0.1, 0.1)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, far) == 0.0
    assert dedupe([a, b, far], box=lambda r: r) == [a, far]     # first one wins


def test_boxes_agree():
    face = (0.50, 0.30, 0.02, 0.035)
    assert boxes_agree(face, face)
    # Tiny and a little off: IoU is poor but it is clearly the same face.
    nudged = (0.51, 0.318, 0.016, 0.028)
    assert box_iou(nudged, face) < 0.3 and boxes_agree(nudged, face)
    # A whole head-and-shoulders "face" around a small real one does not agree.
    blob = (0.46, 0.20, 0.10, 0.18)
    assert not boxes_agree(face, blob)
    assert not boxes_agree(face, (0.8, 0.8, 0.02, 0.035))


# --------------------------------------------------------------- smoothing

def test_filter_passes_through_at_zero():
    smoother = TargetSmoother(0.0)
    for i, x in enumerate([0.1, 0.9, 0.2, 0.7]):
        t = Target(head=(x, x), box=(x, x, 0.1, 0.1))
        assert smoother(t, i / 30) == t


def test_filter_steadies_jitter_when_smoothing():
    raw, smooth = [], []
    smoother = TargetSmoother(0.4)
    for i in range(90):
        x = 0.5 + (0.01 if i % 2 else -0.01)                    # +-1% jitter
        raw.append(x)
        smooth.append(smoother(Target(head=(x, 0.5), box=(x, 0.5, 0.1, 0.1)), i / 30).head[0])
    spread = lambda v: max(v[30:]) - min(v[30:])
    assert spread(smooth) < 0.3 * spread(raw)


def test_filter_catches_up_with_movement():
    smoother = TargetSmoother(0.9)                              # the steadiest setting
    for i in range(60):
        x = min(0.2 + i * 0.02, 0.8)                            # moves, then holds at 0.8
        out = smoother(Target(head=(x, 0.5), box=(x, 0.5, 0.1, 0.1)), i / 30)
    assert out.head[0] == pytest.approx(0.8, abs=0.03)


def test_stronger_smoothing_lowers_the_cutoff():
    assert smoothing_cutoff(0.0) > smoothing_cutoff(0.4) > smoothing_cutoff(0.9) > 0


def test_filter_restarts_after_a_gap():
    smoother = TargetSmoother(0.8, reset_after=0.5)
    smoother(Target(head=(0.1, 0.1), box=(0.1, 0.1, 0.1, 0.1)), 0.0)
    assert smoother(None, 0.1) is None
    after = smoother(Target(head=(0.9, 0.9), box=(0.9, 0.9, 0.1, 0.1)), 2.0)
    assert after.head == pytest.approx((0.9, 0.9))                # no glide from 0.1


def test_box_never_inverts():
    smoother = TargetSmoother(0.6)
    for i, w in enumerate([0.3, 0.01, 0.3, 0.01, 0.3]):
        out = smoother(Target(head=(0.5, 0.5), box=(0.5 - w / 2, 0.4, w, w)), i / 30)
        assert out.box[2] > 0 and out.box[3] > 0


def test_one_euro_first_sample_and_repeated_time():
    f = OneEuroFilter(min_cutoff=1.0)
    assert f(5.0, 1.0) == 5.0
    assert math.isfinite(f(6.0, 1.0))                            # same timestamp: no div by 0


# ------------------------------------------------------------ target choice

def face(x, y, side):
    return Target(head=(x, y), box=(x - side / 2, y - side / 2, side, side))


def choose(candidates, previous):
    return choose_target(candidates, previous, size=lambda t: t.box[2] * t.box[3],
                         head=lambda t: t.head, switch_ratio=1.8)


def test_no_candidates():
    assert choose([], None) is None
    assert choose([], (0.5, 0.5)) is None


def test_largest_without_a_previous_player():
    small, big = face(0.2, 0.3, 0.05), face(0.7, 0.3, 0.1)
    assert choose([small, big], None) is big


def test_sticks_to_the_player_near_the_last_position():
    player, passerby = face(0.2, 0.3, 0.05), face(0.7, 0.3, 0.06)   # 1.44x area
    assert choose([player, passerby], (0.21, 0.3)) is player


def test_hands_over_to_a_much_bigger_newcomer():
    player, newcomer = face(0.2, 0.3, 0.05), face(0.7, 0.3, 0.08)   # 2.56x area
    assert choose([player, newcomer], (0.21, 0.3)) is newcomer


def test_challengers_only_admits_much_bigger():
    faces = [face(0.1, 0.1, 0.05), face(0.5, 0.5, 0.06), face(0.8, 0.8, 0.1)]
    current = 0.05 * 0.05
    assert challengers(faces, current, lambda t: t.box[2] * t.box[3]) == [faces[2]]


# --------------------------------------------------------------------- fps

def test_fps_meter():
    meter = FpsMeter(window=1.0)
    assert meter.fps == 0.0
    for i in range(61):
        meter.tick(i / 30)
    assert meter.fps == pytest.approx(30.0, rel=0.05)
    meter.reset()
    assert meter.fps == 0.0
