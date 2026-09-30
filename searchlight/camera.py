"""Camera capture and head tracking, off the render thread.

A worker thread owns the camera and the tracker and publishes only the newest
result, so the game never waits on either and never draws a stale backlog
(`CAP_PROP_BUFFERSIZE=1` keeps the driver from queueing frames too).

Per frame: capture -> mirror -> digital zoom (a centre crop) -> RGB -> tracker ->
smoothing -> publish. Everything published - frame, head, box - is in the
coordinates of the mirrored, zoomed frame the game draws, so the game never
converts between spaces.

Settings change while the game runs (`configure`). The worker applies them
between frames and does only what the change needs: reopen the camera for a new
index or resolution, rebuild the tracker for a new mode, otherwise just update
numbers in place.

While a recording is running (`start_recording`), each captured frame is also
handed, untouched, to a `recording.Recorder` - before mirroring and zoom, so the
file holds exactly what the camera saw.

Nothing here dies on a missing camera or model: `error` explains the problem in
words a person at the kiosk can act on, and the worker keeps retrying, so fixing
it (granting camera access, running `scripts/fetch_models.sh`) needs no restart.

Import order: pygame and OpenCV both bundle SDL2, and their ObjC classes clash on
macOS unless pygame is imported first. Entry points must import pygame before
this module.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp

from .recording import Recorder
from .tracking.base import MODE_UPPER_BODY, CameraOptions, HeadDetection, Tracker
from .tracking.geometry import FpsMeter, TargetSmoother, zoom_rect

__all__ = ["CameraThread", "CameraStats", "HeadDetection", "build_tracker",
           "check_mediapipe_version"]

CAMERA_HINT = ("Could not open the camera. On macOS, grant Camera access to your "
               "terminal in System Settings > Privacy & Security > Camera.")
RETRY_SECONDS = 2.0
READ_FAILURES_BEFORE_REOPEN = 60        # ~2 s of failed reads: the camera went away


def check_mediapipe_version() -> None:
    """Refuse mediapipe 1.x, which crashes on macOS arm64."""
    version = getattr(mp, "__version__", "0")
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return
    if major >= 1:
        raise RuntimeError(
            f"mediapipe {version} crashes on macOS (\"DrishtiMetalHelper ... Service "
            "is unavailable\"); this project needs the 0.10 line. You are probably "
            "running the system python instead of the project venv - use "
            "`.venv/bin/python`, or reinstall with `pip install -r requirements.txt`."
        )


def build_tracker(options: CameraOptions) -> Tracker:
    """The tracker for `options.mode`. Raises FileNotFoundError if its model is missing."""
    # Imported here so `searchlight.camera` loads even before models are fetched.
    if options.mode == MODE_UPPER_BODY:
        from .tracking.pose import PoseTracker
        tracker = PoseTracker(options.min_confidence)
        tracker.long_range = options.long_range
        return tracker
    from .tracking.face import FaceTracker
    return FaceTracker(options.min_confidence, options.long_range)


@dataclass(frozen=True)
class CameraStats:
    """Diagnostics beyond `HeadDetection`, for `scripts/check_camera.py`."""

    capture_size: Tuple[int, int] = (0, 0)  # what the camera actually delivers
    state: str = "search"                   # tracker state: search | track | full
    roi: Optional[Tuple[int, int, int, int]] = None   # tracking crop, published-frame px
    calls: int = 0                          # detector calls on the last frame
    process_ms: float = 0.0                 # tracker time on the last frame


class CameraThread:
    """Background capture + tracking. Start it, read `latest()`, close it."""

    def __init__(self, options: CameraOptions = CameraOptions()) -> None:
        self._options = options
        self._pending: Optional[CameraOptions] = None
        self._latest = HeadDetection(mode=options.mode)
        self._stats = CameraStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recorder: Optional[Recorder] = None
        self._finishing: List[Recorder] = []     # finished, possibly still writing
        self.error: Optional[str] = None

    # ------------------------------------------------------------ public API

    def start(self) -> "CameraThread":
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def latest(self) -> HeadDetection:
        with self._lock:
            return self._latest

    def stats(self) -> CameraStats:
        with self._lock:
            return self._stats

    def configure(self, options: CameraOptions) -> None:
        """Use `options` from the next frame on. Safe from any thread."""
        with self._lock:
            self._pending = options

    def reset_tracking(self) -> None:
        """Forget the locked player - call before each new player."""
        self._reset.set()

    def start_recording(self, path: Path) -> None:
        """Save raw frames to `path` from the next frame on, replacing any recording."""
        self.stop_recording()
        recorder = Recorder(path)
        with self._lock:
            self._recorder = recorder

    def stop_recording(self, keep: bool = True) -> Optional[Path]:
        """End the recording; delete the file unless `keep`. Returns the kept path."""
        with self._lock:
            recorder, self._recorder = self._recorder, None
        if recorder is None:
            return None
        recorder.finish(keep)
        self._finishing = [r for r in self._finishing if r.is_alive()] + [recorder]
        return recorder.path if keep else None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.stop_recording()
        for recorder in self._finishing:
            recorder.join(timeout=5.0)          # let the files finish writing

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        try:
            check_mediapipe_version()
        except RuntimeError as exc:
            self.error = str(exc)
            return

        options = self._options
        capture: Optional[cv2.VideoCapture] = None
        tracker: Optional[Tracker] = None
        tracker_error: Optional[str] = None
        next_tracker_try = 0.0
        smoother = TargetSmoother(options.smoothing)
        fps = FpsMeter()
        start = time.monotonic()
        last_stamp = -1
        read_failures = 0

        try:
            while not self._stop.is_set():
                # -- settings
                with self._lock:
                    pending, self._pending = self._pending, None
                if pending is not None and pending != options:
                    if (pending.camera_index, pending.width, pending.height) != \
                            (options.camera_index, options.width, options.height):
                        if capture is not None:
                            capture.release()
                        capture = None
                    if pending.mode != options.mode:
                        if tracker is not None:
                            tracker.close()
                        tracker, next_tracker_try = None, 0.0
                        smoother.reset()
                        fps.reset()
                    elif tracker is not None:
                        tracker.min_confidence = pending.min_confidence
                        tracker.long_range = pending.long_range
                    smoother.strength = pending.smoothing
                    options = pending

                if self._reset.is_set():
                    self._reset.clear()
                    if tracker is not None:
                        tracker.reset()
                    smoother.reset()

                # -- camera
                if capture is None:
                    capture = self._open(options)
                    if capture is None:
                        self.error = CAMERA_HINT
                        self._stop.wait(RETRY_SECONDS)
                        continue
                    read_failures = 0

                # -- tracker (a missing model still lets the camera show)
                if tracker is None and time.monotonic() >= next_tracker_try:
                    try:
                        tracker = build_tracker(options)
                        tracker_error = None
                    except Exception as exc:          # missing model, bad bundle, ...
                        tracker_error = str(exc)
                        next_tracker_try = time.monotonic() + RETRY_SECONDS

                ok, frame = capture.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= READ_FAILURES_BEFORE_REOPEN:
                        capture.release()
                        capture = None
                    time.sleep(0.01)
                    continue
                read_failures = 0
                self.error = tracker_error
                now = time.monotonic()

                with self._lock:
                    recorder = self._recorder
                if recorder is not None:
                    recorder.add(frame, now)    # raw: before flip and crop

                if options.mirror:
                    frame = cv2.flip(frame, 1)
                x, y, w, h = zoom_rect(frame.shape[1], frame.shape[0], options.zoom)
                rgb = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)

                # MediaPipe's VIDEO mode insists on strictly increasing stamps.
                stamp = max(int((now - start) * 1000), last_stamp + 1)
                last_stamp = stamp

                target = None
                began = time.perf_counter()
                if tracker is not None:
                    try:
                        target = tracker.process(rgb, stamp)
                    except Exception:
                        target = None
                elapsed = (time.perf_counter() - began) * 1000
                target = smoother(target, now)

                detection = HeadDetection(
                    frame=rgb,
                    head=target.head if target else None,
                    box=target.box if target else None,
                    timestamp=now,
                    ok=True,
                    confidence=target.confidence if target else 0.0,
                    mode=options.mode,
                    fps=fps.tick(now),
                    searching=bool(getattr(tracker, "searching", target is None)),
                )
                stats = CameraStats(
                    capture_size=(frame.shape[1], frame.shape[0]),
                    state=getattr(tracker, "state", "none"),
                    roi=getattr(tracker, "roi", None),
                    calls=getattr(tracker, "calls", 0),
                    process_ms=elapsed,
                )
                with self._lock:
                    self._latest, self._stats = detection, stats
        finally:
            if capture is not None:
                capture.release()
            if tracker is not None:
                tracker.close()

    @staticmethod
    def _open(options: CameraOptions) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(options.camera_index)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, options.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, options.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
