#!/usr/bin/env bash
# One-step setup for Searchlight. Run once from anywhere:
#
#   ./install.sh
#
# It finds (or installs) a Python that the pinned packages support, builds .venv,
# installs the requirements and downloads the MediaPipe models. Safe to re-run: a
# good .venv and models already on disk are kept.
#
# Python 3.10-3.12 only. mediapipe 0.10 needs numpy < 2, and numpy 1.x has no
# builds for 3.13 or newer, so a newer default python3 fails part-way through
# `pip install`. Set PYTHON=/path/to/python3.x to choose one yourself.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
SUPPORTED="3.10 3.11 3.12"
PREFERRED="3.11"          # the version Searchlight is developed and tested on

say()  { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Prints "3.11" for a working interpreter, nothing otherwise.
version_of() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

supported() {
    [[ " $SUPPORTED " == *" $1 "* ]]
}

find_python() {
    local candidate version
    if [[ -n "${PYTHON:-}" ]]; then
        version="$(version_of "$PYTHON")"
        [[ -n "$version" ]] || fail "PYTHON=$PYTHON does not run."
        supported "$version" || fail "PYTHON=$PYTHON is Python $version; Searchlight needs one of: $SUPPORTED."
        echo "$PYTHON"
        return
    fi
    # Newest-first after the preferred one, then whatever python3 is.
    for candidate in "python$PREFERRED" python3.12 python3.10 python3 \
                     "/opt/homebrew/opt/python@$PREFERRED/bin/python$PREFERRED" \
                     "/usr/local/opt/python@$PREFERRED/bin/python$PREFERRED"; do
        command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]] || continue
        version="$(version_of "$candidate")"
        if [[ -n "$version" ]] && supported "$version"; then
            command -v "$candidate" 2>/dev/null || echo "$candidate"
            return
        fi
    done
}

# ---------------------------------------------------------------- python

say "Looking for Python ($SUPPORTED)"
PY="$(find_python)"
if [[ -z "$PY" ]]; then
    if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        say "No supported Python found - installing python@$PREFERRED with Homebrew"
        brew install "python@$PREFERRED"
        PY="$(find_python)"
    fi
fi
if [[ -z "$PY" ]]; then
    fail "No Python $SUPPORTED found.
Install Python $PREFERRED, then run ./install.sh again:
  macOS:  brew install python@$PREFERRED   (or the installer from https://www.python.org/downloads/)
  Ubuntu: sudo apt install python$PREFERRED python$PREFERRED-venv
Or point the installer at one: PYTHON=/path/to/python3.x ./install.sh"
fi
echo "Using $PY (Python $(version_of "$PY"))"

# ------------------------------------------------------------------ venv

if [[ -x "$VENV/bin/python" ]]; then
    venv_version="$(version_of "$VENV/bin/python")"
    if ! supported "$venv_version"; then
        say "Existing .venv is Python ${venv_version:-unknown} - rebuilding it"
        rm -rf "$VENV"
    fi
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    say "Creating .venv"
    "$PY" -m venv "$VENV" || fail "Could not create a virtual environment.
On Ubuntu/Debian install the venv module first: sudo apt install python3-venv"
fi
VPY="$VENV/bin/python"

# ---------------------------------------------------------- requirements

say "Installing packages (the first run downloads ~150 MB)"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet -r "$ROOT/requirements.txt"

# The regular OpenCV wheels bundle their own SDL2, which collides with pygame's.
# mediapipe depends on the regular build anyway, so pip always brings it in; only
# the headless one belongs here. Both share the cv2 folder, so after removing the
# regular one, put the headless one back whole at the version requirements.txt pins.
HEADLESS="$(grep -E '^opencv-contrib-python-headless' "$ROOT/requirements.txt" | sed 's/[[:space:]]*#.*//')"
for pkg in opencv-python opencv-contrib-python; do
    if "$VPY" -m pip show "$pkg" >/dev/null 2>&1; then
        say "Removing $pkg (conflicts with pygame; the headless build is used instead)"
        "$VPY" -m pip uninstall --quiet -y "$pkg"
        "$VPY" -m pip install --quiet --force-reinstall --no-deps "$HEADLESS"
    fi
done

# ----------------------------------------------------------- data files

say "Downloading detection models"
bash "$ROOT/scripts/fetch_models.sh"

shopt -s nullglob
fonts=("$ROOT"/assets/fonts/*.ttf)
if (( ${#fonts[@]} < 4 )); then
    say "Downloading fonts"
    bash "$ROOT/scripts/fetch_fonts.sh"
fi

# ---------------------------------------------------------------- check

say "Checking the install"
# pygame first, as in the app: it and OpenCV both carry SDL2.
"$VPY" - <<'EOF' || fail "The packages installed but do not import - see the error above."
import os, sys
import pygame, cv2, mediapipe, numpy
print(f"pygame {pygame.version.ver}, opencv {cv2.__version__}, "
      f"mediapipe {mediapipe.__version__}, numpy {numpy.__version__}")
# os.walk, not glob: the bundled libraries live in a hidden .dylibs folder.
if any("sdl" in f.lower() for _, _, files in os.walk(os.path.dirname(cv2.__file__))
       for f in files):
    sys.exit(f"OpenCV {cv2.__version__} bundles its own SDL2, which clashes with pygame's.")
EOF

cat <<EOF

Searchlight is installed. Start it with:

  .venv/bin/python run_searchlight.py              # game + leaderboard windows
  .venv/bin/python run_searchlight.py --dev        # no camera: the mouse is your head
  .venv/bin/python run_searchlight.py --fullscreen # kiosk mode on two displays

EOF
if [[ "$(uname)" == "Darwin" ]]; then
    cat <<EOF
macOS: allow your terminal to use the camera in System Settings > Privacy &
Security > Camera. Without it the preview stays black and no error is shown.

EOF
fi
