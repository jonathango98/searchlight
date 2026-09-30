#!/bin/bash
# Download Open Font License fonts from google/fonts for Searchlight.
# Idempotent: safe to run multiple times.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FONT_DIR="$PROJECT_ROOT/assets/fonts"

# Create the font directory if it doesn't exist
mkdir -p "$FONT_DIR"

# Base URL for the google/fonts raw GitHub repo
BASE_URL="https://github.com/google/fonts/raw/main"

# Font files to download
FONTS=(
    "ofl/bebasneue/BebasNeue-Regular.ttf"
    "ofl/barlow/Barlow-SemiBold.ttf"
    "ofl/barlow/Barlow-Bold.ttf"
    "ofl/spacemono/SpaceMono-Bold.ttf"
)

# Download each font
for font in "${FONTS[@]}"; do
    filename=$(basename "$font")
    dest="$FONT_DIR/$filename"
    echo "Downloading $filename..."
    curl -fsSL "$BASE_URL/$font" -o "$dest"
done

# Download license files with renamed output
echo "Downloading license files..."
curl -fsSL "$BASE_URL/ofl/bebasneue/OFL.txt" -o "$FONT_DIR/OFL-BebasNeue.txt"
curl -fsSL "$BASE_URL/ofl/barlow/OFL.txt" -o "$FONT_DIR/OFL-Barlow.txt"
curl -fsSL "$BASE_URL/ofl/spacemono/OFL.txt" -o "$FONT_DIR/OFL-SpaceMono.txt"

# Verify fonts load with pygame
echo "Verifying fonts..."
python3 -c "
import pygame
pygame.font.init()
fonts = ['BebasNeue-Regular.ttf', 'Barlow-SemiBold.ttf', 'Barlow-Bold.ttf', 'SpaceMono-Bold.ttf']
for f in fonts:
    pygame.font.Font('$FONT_DIR/' + f, 20)
    print(f'✓ {f} loads')
"

echo "Done."
