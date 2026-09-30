"""Tests for run_searchlight.py command building and re-execution logic."""

from pathlib import Path
import sys

import pytest

# Import the module to test
sys.path.insert(0, str(Path(__file__).parent.parent))
from run_searchlight import should_reexec, build_command


class TestShouldReexec:
    """Tests for the should_reexec decision logic."""

    def test_already_reexeced_returns_false(self):
        """Guard against exec loops."""
        assert should_reexec("/some/prefix", already_reexeced=True, venv_python_exists=True) is False

    def test_venv_missing_returns_false(self):
        """If venv doesn't exist, don't try to exec into it."""
        assert should_reexec("/some/prefix", already_reexeced=False, venv_python_exists=False) is False

    def test_same_prefix_returns_false(self):
        """If already in the venv prefix, don't re-exec."""
        venv_root = Path(__file__).parent.parent / ".venv"
        prefix = str(venv_root.resolve())
        # When sys.prefix is the venv root itself, don't re-exec
        assert should_reexec(prefix, already_reexeced=False, venv_python_exists=True) is False

    def test_different_prefix_returns_true(self):
        """If in a different interpreter, re-exec."""
        venv_root = Path(__file__).parent.parent / ".venv"
        # Use a prefix that is definitely not under the venv
        assert should_reexec("/usr/bin", already_reexeced=False, venv_python_exists=True) is True
        assert should_reexec("/opt/homebrew", already_reexeced=False, venv_python_exists=True) is True


class TestBuildCommand:
    """Tests for command building."""

    def test_build_game_command(self):
        """Build a game module command."""
        cmd = build_command("game", ["--display", "0", "--size", "1280x720"])
        assert cmd == [sys.executable, "-m", "searchlight.game", "--display", "0", "--size", "1280x720"]

    def test_build_leaderboard_command(self):
        """Build a leaderboard module command."""
        cmd = build_command("leaderboard", ["--display", "1", "--size", "1280x720"])
        assert cmd == [sys.executable, "-m", "searchlight.leaderboard", "--display", "1", "--size", "1280x720"]

    def test_build_command_with_no_args(self):
        """Build a command with no extra arguments."""
        cmd = build_command("game", [])
        assert cmd == [sys.executable, "-m", "searchlight.game"]
