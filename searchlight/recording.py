"""Saving the raw camera feed of a run to a video file.

Raw means straight off the camera: full capture resolution, not mirrored, not
zoomed, nothing drawn on it - so a run can be watched back, or fed through the
tracker again, exactly as the camera saw it.

The camera thread hands each frame to `Recorder.add`, which only queues it; a
thread of its own does the encoding, so recording never slows tracking. If the
encoder falls behind, frames are dropped rather than queued without limit.

Video files have a fixed frame rate but the camera does not: the capture loop
runs as fast as the tracker allows, which varies. So the writer places each
frame by its capture time, repeating the previous frame over gaps and skipping
frames that arrive early. A ten-second run plays back in ten seconds.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["Recorder"]

FPS = 30.0
QUEUE_FRAMES = 60                       # ~2 s at 30 fps before frames are dropped
# H.264 plays everywhere; mp4v is the fallback where no H.264 encoder is built in.
CODECS: Tuple[str, ...] = ("avc1", "mp4v")


class Recorder:
    """Encodes BGR frames to `path` on a background thread. `add`, then `finish`."""

    def __init__(self, path: Path, fps: float = FPS) -> None:
        self.path = path
        self.fps = fps
        self.error: Optional[str] = None
        self._queue: "queue.Queue[Optional[Tuple[np.ndarray, float]]]" = \
            queue.Queue(QUEUE_FRAMES)
        self._keep = True
        self._finished = False
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray, timestamp: float) -> None:
        """Queue a BGR frame captured at `timestamp` (seconds). Never blocks."""
        if self._finished:
            return
        try:
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:
            pass                        # the writer fills the gap by repeating a frame

    def finish(self, keep: bool = True) -> None:
        """Stop taking frames and close the file - or delete it if not `keep`.

        Returns at once; the writer drains what is queued on its own thread.
        Call `join` to wait for the file to be complete.
        """
        if self._finished:
            return
        self._finished = True
        self._keep = keep
        try:
            self._queue.put(None, timeout=2.0)      # waits only while the writer drains
        except queue.Full:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    # ---------------------------------------------------------------- writer

    def _run(self) -> None:
        writer: Optional[cv2.VideoWriter] = None
        size: Tuple[int, int] = (0, 0)
        start = 0.0
        written = 0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if self.error is not None:
                    continue            # keep draining so `finish` never blocks
                frame, timestamp = item
                if writer is None:
                    size = (frame.shape[1], frame.shape[0])
                    writer = self._open(size)
                    if writer is None:
                        self.error = f"Could not open a video encoder for {self.path}"
                        print(self.error)
                        continue
                    start = timestamp
                if (frame.shape[1], frame.shape[0]) != size:
                    # The resolution setting changed mid-run; a file has one size.
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                # Every frame slot up to this capture time shows this frame.
                due = int((timestamp - start) * self.fps) + 1
                for _ in range(due - written):
                    writer.write(frame)
                written = max(written, due)
        finally:
            if writer is not None:
                writer.release()
            if not self._keep or writer is None:
                self.path.unlink(missing_ok=True)

    def _open(self, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for codec in CODECS:
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec),
                                     self.fps, size)
            if writer.isOpened():
                return writer
            writer.release()
        return None
