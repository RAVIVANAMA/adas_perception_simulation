# ADAS Perception Stack

A production-oriented C++17 / CUDA / Python demonstration project covering the
full stack of an Advanced Driver Assistance System (ADAS): sensor input,
deep-learning inference, multi-sensor fusion, object tracking, trajectory
prediction, and closed-loop planning functions (ACC, AEB, LKA, traffic-light
handling).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Sensor Inputs                                │
│   Camera (BGR)   ──►  ObjectDetector  ──► 2-D Detections            │
│   Radar targets  ──────────────────────┐                            │
│   LiDAR cloud    ──────────────────────┤                            │
│   Camera         ──►  LaneDetector  ──►│ LaneInfo                   │
└─────────────────────────────────────────────────────────────────────┘
                              │                │
                    ┌─────────▼──────────┐     │
                    │   SensorFusion     │     │
                    │  (EKF late fusion) │     │
                    └─────────┬──────────┘     │
                              │                │
                 ┌────────────▼───────┐        │
                 │ MultiObjectTracker │        │
                 │    (SORT + KF)     │        │
                 └────────────┬───────┘        │
                              │                │
                 ┌────────────▼───────┐        │
                 │ TrajectoryPredictor│        │
                 │ (CV / CTRA / Poly) │        │
                 └────────────┬───────┘        │
                              │                │
          ┌───────────────────┼────────────────▼──────────┐
          │             Planning Layer                     │
          │  ┌──────────┐  ┌──────┐  ┌─────┐  ┌────────┐  │
          │  │    ACC   │  │  AEB │  │ LKA │  │  TL    │  │
          │  │ (PID +   │  │(TTC) │  │(PID)│  │handler │  │
          │  │ headway) │  │      │  │     │  │        │  │
          │  └────┬─────┘  └──┬───┘  └──┬──┘  └───┬────┘  │
          │       └───────────┴──────────┴──────────┘       │
          │                 VehicleControl                  │
          └─────────────────────────────────────────────────┘
```

---

## Directory Structure

```
adas_perception_stack/
├── CMakeLists.txt               # Root CMake – builds lib + exe + tests
├── include/
│   ├── common/
│   │   ├── types.hpp            # Shared data types (DetectedObject, EgoState…)
│   │   ├── logger.hpp           # Thread-safe singleton logger + macros
│   │   └── math_utils.hpp       # PID, KF1D, NMS, TTC, IoU, angle utils
│   ├── inference/
│   │   └── onnx_runner.hpp      # IInferenceRunner + OnnxRunner + TRTRunner
│   ├── perception/
│   │   ├── object_detector.hpp  # YOLO-style 2-D object detection wrapper
│   │   ├── lane_detector.hpp    # DNN-based lane boundary detector
│   │   └── sensor_fusion.hpp    # EKF multi-sensor (camera + radar + lidar)
│   ├── tracking/
│   │   ├── multi_object_tracker.hpp  # SORT (Kalman + Hungarian)
│   │   └── hungarian.hpp        # O(n³) Hungarian assignment algorithm
│   ├── prediction/
│   │   └── trajectory_predictor.hpp  # CV / CTRA / polynomial prediction
│   └── planning/
│       └── adas_controllers.hpp # ACC, AEB, LKA, TrafficLightHandler
│
├── src/
│   ├── common/       logger.cpp
│   ├── inference/    onnx_runner.cpp   tensorrt_runner.cpp
│   ├── perception/   object_detector.cpp  lane_detector.cpp  sensor_fusion.cpp
│   ├── tracking/     multi_object_tracker.cpp  hungarian.cpp  kalman_filter.cpp
│   ├── prediction/   trajectory_predictor.cpp
│   ├── planning/     acc_controller.cpp  aeb_controller.cpp
│   │                 lane_keeping_assist.cpp  traffic_light_handler.cpp
│   ├── cuda/         nms_kernel.cu  preprocess_kernel.cu
│   └── main.cpp                 # Full pipeline demo (synthetic data)
│
├── tests/
│   ├── CMakeLists.txt
│   ├── test_aeb_controller.cpp  # 8 AEB unit tests
│   ├── test_acc_controller.cpp  # 6 ACC unit tests
│   ├── test_math_utils.cpp      # PID, NMS, TTC, IoU, Kalman1D tests
│   ├── test_sensor_fusion.cpp   # EKF fusion tests
│   ├── test_tracker.cpp         # SORT tracker tests
│   └── test_lka_predictor.cpp   # LKA + trajectory predictor tests
│
├── python/
│   ├── model_export/
│   │   ├── export_to_onnx.py    # PyTorch → ONNX export + ONNX Runtime validation
│   │   └── quantize_model.py    # INT8/FP16 quantisation (ORT + TensorRT)
│   ├── training/
│   │   ├── dataset.py           # DetectionDataset, LaneDataset, MultiTaskDataset
│   │   └── train_detector.py    # DDP training loop with cosine LR + AMP + W&B
│   └── eval/
│       └── evaluate_adas.py     # mAP, lane accuracy, AEB/ACC metrics, latency
│
└── scripts/
    ├── build.sh                 # CMake configure + build (Ninja)
    └── run_demo.sh              # Build (if needed) + launch demo
```

---

## Prerequisites

| Requirement      | Version   | Notes |
|------------------|-----------|-------|
| CMake            | ≥ 3.18    | |
| C++ compiler     | GCC 11 / Clang 14 | C++17 required |
| Eigen3           | ≥ 3.4     | `apt install libeigen3-dev` |
| Google Test      | auto-fetched | via `FetchContent` |
| CUDA Toolkit     | ≥ 11.8    | Optional – GPU kernels |
| ONNX Runtime     | ≥ 1.16    | Optional – DNN inference |
| TensorRT         | ≥ 8.6     | Optional – TRT backend |
| Python           | ≥ 3.10    | For tooling scripts |
| PyTorch          | ≥ 2.1     | For training/export |

---

## Build

```bash
# Release build
./scripts/build.sh Release

# Debug build with ASAN/UBSAN
./scripts/build.sh Debug

# With explicit ONNX Runtime path
./scripts/build.sh Release \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.18.0

# With TensorRT
./scripts/build.sh Release \
    -DTENSORRT_ROOT=/usr/local/tensorrt
```

---

## Run the Demo

```bash
# Synthetic data only (no model files required):
./scripts/run_demo.sh

# With real ONNX models:
DETECTOR_MODEL=models/yolov8n.onnx \
LANE_MODEL=models/ufld.onnx        \
./scripts/run_demo.sh
```

The demo runs a 30 Hz synthetic pipeline and logs:
- Object detections and fused track counts
- AEB state (Inactive / Warning / PartialBrake / FullBrake)
- ACC throttle/brake commands
- LKA steering corrections
- Traffic-light state
- Predicted lead-vehicle trajectory (every 5 s)

---

## Unit Tests

```bash
cd build/Release
ctest --output-on-failure        # run all

# Or directly:
./tests/adas_tests --gtest_filter="AEBController.*"
./tests/adas_tests --gtest_filter="ACCController.*"
./tests/adas_tests --gtest_filter="SensorFusion.*"
./tests/adas_tests --gtest_filter="MultiObjectTracker.*"
./tests/adas_tests --gtest_filter="MathUtils.*"
```

---

## Python Tooling

### Export a PyTorch model to ONNX

```bash
python python/model_export/export_to_onnx.py \
    --model  path/to/yolov8n.pt \
    --output models/detector.onnx \
    --imgsz  640 \
    --simplify \
    --dynamic
```

### Quantise (INT8 / FP16)

```bash
# Dynamic INT8 (CPU-friendly, no calibration data)
python python/model_export/quantize_model.py \
    --model   models/detector.onnx \
    --mode    dynamic_int8 \
    --output  models/detector_int8.onnx \
    --benchmark

# TensorRT FP16 engine
python python/model_export/quantize_model.py \
    --model   models/detector.onnx \
    --mode    trt_fp16 \
    --output  models/detector.engine \
    --device  0
```

### Train a detector

```bash
# Single GPU
python python/training/train_detector.py \
    --data   data/coco \
    --epochs 50 \
    --batch  16 \
    --device cuda:0 \
    --wandb

# Multi-GPU (4 × A100)
torchrun --nproc_per_node=4 python/training/train_detector.py \
    --data data/coco --epochs 100 --batch 64
```

### Evaluate

```bash
python python/eval/evaluate_adas.py \
    --detector-model models/detector.onnx \
    --dataset        data/val \
    --output         reports/eval.json
```

---

## Key Design Decisions

| Area | Choice | Rationale |
|------|--------|-----------|
| C++ standard | C++17 | `std::optional`, `if constexpr`, structured bindings |
| Memory management | RAII + smart pointers | No raw `new/delete`; PIMPL for ABI stability |
| Linear algebra | Eigen3 | Header-only, zero extra deps, fast |
| Inference | ONNX Runtime (primary) / TensorRT (GPU) | Portable + highest GPU throughput |
| Tracking | SORT (Kalman + Hungarian) | Real-time capable, well-understood |
| Prediction | CV / CTRA / Poly | Graduated complexity per scenario |
| Planning | PID controllers | Deterministic, tunable, safety-provable |
| Logging | Custom thread-safe singleton | No external dep, supports file sinks |
| Build | CMake + FetchContent | Reproducible, no manual GTest install |
| Tests | Google Test | De-facto C++ standard |
| Python tooling | PyTorch + ONNX Runtime | Industry standard |

---

## Safety Notes

This project is a **demonstration**. For production/road deployment:

- All ADAS functions must be validated against ISO 26262 / SOTIF.
- Static analysis (clang-tidy, cppcheck, Polyspace) is mandatory.
- Failure modes, watchdog timers, and fallback states must be implemented.
- Models must be evaluated for distributional robustness and edge cases.

---

## License

MIT – see LICENSE file.
