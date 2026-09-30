"""The run recorder: frames in at a varying rate, a constant-rate video out."""

import pygame  # noqa: F401 - before cv2; see the import-order note in camera.py

import cv2
import numpy as np

from searchlight.recording import Recorder

SIZE = (320, 180)


def frame(value, size=SIZE):
    return np.full((size[1], size[0], 3), value, np.uint8)


def read_back(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


def record(path, stamped_frames, keep=True):
    recorder = Recorder(path, fps=10)
    for image, timestamp in stamped_frames:
        recorder.add(image, timestamp)
    recorder.finish(keep)
    recorder.join(5.0)
    return recorder


def test_playback_takes_as_long_as_the_capture(tmp_path):
    # Uneven capture times covering 2 s: bursts and a long stall.
    stamps = [0.0, 0.05, 0.1, 0.12, 0.3, 1.5, 1.6, 2.0]
    path = tmp_path / "run.mp4"
    record(path, [(frame(20 * i), t) for i, t in enumerate(stamps)])
    frames = read_back(path)
    assert len(frames) == 21                       # 2 s at 10 fps, plus frame 0
    assert frames[0].shape[:2] == (SIZE[1], SIZE[0])


def test_raw_frames_are_written_unchanged(tmp_path):
    image = frame(0)
    image[:, : SIZE[0] // 2] = 255                 # white left half: a mirror would show
    path = tmp_path / "run.mp4"
    record(path, [(image, 0.0), (image, 0.1)])
    shown = read_back(path)[0]
    assert shown[:, : SIZE[0] // 4].mean() > 200
    assert shown[:, -SIZE[0] // 4:].mean() < 50


def test_a_mid_run_resolution_change_is_resized(tmp_path):
    path = tmp_path / "run.mp4"
    record(path, [(frame(0), 0.0), (frame(0, (640, 360)), 0.1)])
    frames = read_back(path)
    assert len(frames) == 2 and all(f.shape[:2] == (SIZE[1], SIZE[0]) for f in frames)


def test_a_discarded_recording_leaves_no_file(tmp_path):
    path = tmp_path / "run.mp4"
    record(path, [(frame(0), 0.0), (frame(0), 0.5)], keep=False)
    assert not path.exists()


def test_a_recording_with_no_frames_leaves_no_file(tmp_path):
    path = tmp_path / "sub" / "run.mp4"
    record(path, [])
    assert not path.exists()


def test_frames_after_finish_are_ignored(tmp_path):
    path = tmp_path / "run.mp4"
    recorder = record(path, [(frame(0), 0.0)])
    recorder.add(frame(0), 5.0)
    assert len(read_back(path)) == 1
