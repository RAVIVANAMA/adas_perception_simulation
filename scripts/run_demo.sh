#!/usr/bin/env bash
# run_demo.sh – Build (if needed) and launch the ADAS demo.
# Optional: pass model paths as env vars before calling this script.
#
# Examples:
#   # With real models:
#   DETECTOR_MODEL=models/detector.onnx  \
#   LANE_MODEL=models/lane.onnx          \
#   ./scripts/run_demo.sh
#
#   # Synthetic-data only (no models required):
#   ./scripts/run_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build/Release"
BINARY="$BUILD_DIR/adas_demo"

if [[ ! -f "$BINARY" ]]; then
    echo "==> Binary not found, building first…"
    "$SCRIPT_DIR/build.sh" Release
fi

DETECTOR_MODEL="${DETECTOR_MODEL:-}"
LANE_MODEL="${LANE_MODEL:-}"

echo "==> Starting ADAS demo"
echo "    Detector model : ${DETECTOR_MODEL:-<none – using synthetic detections>}"
echo "    Lane model     : ${LANE_MODEL:-<none – using synthetic lane data>}"
echo ""

exec "$BINARY" "$DETECTOR_MODEL" "$LANE_MODEL"
