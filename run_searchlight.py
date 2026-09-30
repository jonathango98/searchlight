#!/usr/bin/env python3
"""Start both Searchlight windows and keep their lifetimes tied together.

This script re-launches itself under the project venv if needed, then spawns
the game and leaderboard as separate processes and ties their lifetimes together.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"


def should_reexec(prefix: str, already_reexeced: bool, venv_python_exists: bool) -> bool:
    """Decide whether to hand over to the venv interpreter.

    Split out so the decision can be tested without the exec itself.
    """
    if already_reexeced:
        return False  # guard against an exec loop
    if not venv_python_exists:
        return False
    # Compare prefixes, not interpreter paths. `.venv/bin/python` is a symlink chain
    # ending at the system interpreter, so resolving both sides makes them compare
    # equal and the check never fires.
    return Path(prefix).resolve() != VENV_DIR.resolve()


def reexec_in_venv() -> None:
    """Re-launch under the venv interpreter if needed."""
    if not should_reexec(sys.prefix, bool(os.environ.get("SEARCHLIGHT_REEXEC")),
                         VENV_PYTHON.exists()):
        return
    print(f"Re-launching under {VENV_PYTHON}", flush=True)
    os.environ["SEARCHLIGHT_REEXEC"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()),
                                *sys.argv[1:]])


def build_command(module: str, extra: list[str]) -> list[str]:
    """Build a command to run a searchlight module."""
    return [sys.executable, "-m", f"searchlight.{module}", *extra]


def main() -> int:
    reexec_in_venv()

    # Pre-flight: check for models
    model_path = ROOT / "models" / "blaze_face_short_range.tflite"
    if not model_path.exists():
        print("run scripts/fetch_models.sh", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Run the Searchlight kiosk")
    parser.add_argument("--game-display", type=int, default=0)
    parser.add_argument("--board-display", type=int, default=1)
    parser.add_argument("--db", default=str(ROOT / "searchlight.db"))
    parser.add_argument("--fullscreen", action="store_true",
                        help="run both fullscreen instead of in windows")
    parser.add_argument("--game-size", default="1280x720")
    parser.add_argument("--board-size", default="1280x720")
    parser.add_argument("--dev", action="store_true",
                        help="the mouse stands in for your head")
    parser.add_argument("--game-pos", default="40,60", help="game window X,Y")
    parser.add_argument("--board-pos", default="140,220", help="leaderboard window X,Y")
    parser.add_argument("--recordings", default=str(ROOT / "recordings"),
                        help="folder for run videos (when Record runs is on)")
    args = parser.parse_args()

    shared = ["--db", args.db] + (["--fullscreen"] if args.fullscreen else [])
    if not args.fullscreen:
        game_pos, board_pos = ["--pos", args.game_pos], ["--pos", args.board_pos]
    else:
        game_pos = board_pos = []

    game_args = shared + game_pos + [
        "--display", str(args.game_display),
        "--size", args.game_size,
        "--recordings", args.recordings,
    ] + (["--dev"] if args.dev else [])
    board_args = shared + board_pos + [
        "--display", str(args.board_display),
        "--size", args.board_size,
    ]

    processes = {
        "leaderboard": subprocess.Popen(build_command("leaderboard", board_args), cwd=ROOT),
        "game": subprocess.Popen(build_command("game", game_args), cwd=ROOT),
    }

    def shutdown(*_args) -> None:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, lambda *a: (shutdown(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (shutdown(), sys.exit(0)))

    try:
        while True:
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    print(f"{name} exited (code {code}); shutting down Searchlight.")
                    shutdown()
                    for other in processes.values():
                        try:
                            other.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            other.kill()
                    return code or 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
