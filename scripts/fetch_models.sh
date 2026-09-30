#!/usr/bin/env bash
# Download the MediaPipe models Searchlight needs into models/.
#
# Idempotent: a model that is already present (and non-empty) is left alone, so
# this is safe to run before every event. Downloads go to a temp file first and
# are moved into place only when complete, so an interrupted run never leaves a
# truncated model that would fail to load later.
#
#   face  - BlazeFace short-range detector ("Detect: Face")
#   pose  - PoseLandmarker full ("Detect: Upper body")

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

fetch() {
    local url="$1" name="$2"
    local dest="$MODELS/$name"
    if [[ -s "$dest" ]]; then
        echo "ok       $name (already present)"
        return
    fi
    echo "fetching $name"
    local tmp
    tmp="$(mktemp "$MODELS/.$name.XXXXXX")"
    if curl -fL --retry 3 --progress-bar -o "$tmp" "$url"; then
        chmod 644 "$tmp"
        mv "$tmp" "$dest"
        echo "ok       $name ($(du -h "$dest" | cut -f1 | tr -d ' '))"
    else
        rm -f "$tmp"
        echo "FAILED   $name from $url" >&2
        return 1
    fi
}

fetch "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
      "blaze_face_short_range.tflite"
fetch "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" \
      "pose_landmarker_full.task"
