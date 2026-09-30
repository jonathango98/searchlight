"""Check head detection from the terminal before an event - no window needed.

Stand at 1 m, 3 m and 5 m (or ask someone to) and watch the live line: is the
head found, how big is the face box in pixels, and is the tracker locked
(`track`) or still scanning (`search`)? The face detector needs the tracking
crop to lock on; if a spot only ever shows `search`, try `--zoom`, a higher
`--resolution`, or `--mode upper_body`.

    .venv/bin/python scripts/check_camera.py
    .venv/bin/python scripts/check_camera.py --mode upper_body --seconds 30
    .venv/bin/python scripts/check_camera.py --zoom 1.5 --resolution 1920x1080

Runs the same `CameraThread` as the game, so what it reports is what the game
will see. Ctrl-C ends early and still prints the summary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"

# Re-run under the project venv if started with another python: the system
# python usually lacks mediapipe, or has a 1.x that crashes on macOS.
if (VENV / "bin" / "python").exists() and Path(sys.prefix).resolve() != VENV.resolve() \
        and not os.environ.get("SEARCHLIGHT_NO_REEXEC"):
    os.environ["SEARCHLIGHT_NO_REEXEC"] = "1"
    python = str(VENV / "bin" / "python")
    os.execv(python, [python, *sys.argv])

sys.path.insert(0, str(ROOT))

import argparse                                           # noqa: E402
import statistics                                         # noqa: E402
import time                                               # noqa: E402

from searchlight.camera import CameraThread               # noqa: E402
from searchlight.tracking.base import (MODE_FACE, MODE_UPPER_BODY,  # noqa: E402
                                       CameraOptions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=(MODE_FACE, MODE_UPPER_BODY), default=MODE_FACE)
    parser.add_argument("--no-long-range", action="store_true",
                        help="plain whole-frame detection (no tracking crop or tiles)")
    parser.add_argument("--zoom", type=float, default=1.0, help="digital zoom, 1.0-3.0")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--seconds", type=float, default=20.0, help="how long to run")
    parser.add_argument("--resolution", default="1280x720", help="WIDTHxHEIGHT")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="minimum detection confidence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        width, height = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        print(f"--resolution must look like 1280x720, not {args.resolution!r}")
        return 2
    options = CameraOptions(camera_index=args.camera, width=width, height=height,
                            mode=args.mode, long_range=not args.no_long_range,
                            zoom=args.zoom, min_confidence=args.confidence,
                            smoothing=0.0)       # raw numbers: we are measuring
    print(f"mode={args.mode} long_range={not args.no_long_range} zoom={args.zoom}x "
          f"camera={args.camera} requested={width}x{height} for {args.seconds:.0f}s")

    camera = CameraThread(options).start()
    frames = found = locked = 0
    heights_px, fps_seen = [], []
    last_stamp = 0.0
    last_error = None
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            time.sleep(0.03)
            if camera.error != last_error:
                last_error = camera.error
                if last_error:
                    print(f"\n{last_error}")
            detection = camera.latest()
            if not detection.ok or detection.timestamp == last_stamp:
                continue
            last_stamp = detection.timestamp
            stats = camera.stats()
            frame_h = detection.frame.shape[0]
            frames += 1
            locked += not detection.searching
            if detection.fps:
                fps_seen.append(detection.fps)
            if detection.box is not None:
                found += 1
                box_px = detection.box[3] * frame_h
                heights_px.append(box_px)
                seen = f"YES  box {box_px:5.0f}px {100 * detection.box[3]:5.1f}% of height"
            else:
                seen = "no   box     -          -        "
            roi = f"roi {stats.roi[2]}px" if stats.roi else "roi   -  "
            sys.stdout.write(
                f"\r{detection.fps:5.1f} fps  {seen}  {stats.state:<6} {roi:<9} "
                f"calls {stats.calls}  {stats.process_ms:5.1f}ms  "
                f"capture {stats.capture_size[0]}x{stats.capture_size[1]}   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()

    print("\n\nsummary")
    if not frames:
        print(f"  no frames received. {camera.error or ''}")
        return 1
    print(f"  frames      {frames} (sampled), tracking fps "
          f"{statistics.median(fps_seen) if fps_seen else 0:.1f} median")
    print(f"  detected    {100 * found / frames:.0f}% of frames, "
          f"locked {100 * locked / frames:.0f}%")
    if heights_px:
        print(f"  box height  median {statistics.median(heights_px):.0f}px, "
              f"min {min(heights_px):.0f}px, max {max(heights_px):.0f}px "
              f"(of {frame_h}px frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
