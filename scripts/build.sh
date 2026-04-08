#!/usr/bin/env bash
# build.sh – Configure and build the ADAS Perception Stack
# Usage:  ./scripts/build.sh [Debug|Release] [extra cmake args...]

set -euo pipefail

BUILD_TYPE=${1:-Release}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build/$BUILD_TYPE"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "==> Build type : $BUILD_TYPE"
echo "==> Build dir  : $BUILD_DIR"
echo "==> Parallel   : $JOBS jobs"
echo ""

# Optional: point to ONNX Runtime and TensorRT if installed
# export ONNXRUNTIME_ROOT=/opt/onnxruntime
# export TENSORRT_ROOT=/opt/tensorrt

cmake \
    -S "$ROOT_DIR" \
    -B "$BUILD_DIR" \
    -G "Ninja" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    "${@:2}"

cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo ""
echo "==> Build complete. Binaries:"
echo "      $BUILD_DIR/adas_demo"
echo "      $BUILD_DIR/tests/adas_tests"
echo ""
echo "==> Run demo:   $BUILD_DIR/adas_demo"
echo "==> Run tests:  cd $BUILD_DIR && ctest --output-on-failure"
