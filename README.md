<div align="center">

# 🚗 ADAS Perception Simulation Stack

**A full-stack, production-oriented demonstration of an Advanced Driver Assistance System (ADAS) — from raw sensor data to real-time vehicle control decisions.**

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-green.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-Optional-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

[**Quick Start**](#-quick-start) · [**Architecture**](#%EF%B8%8F-system-architecture) · [**Visualizer**](#%EF%B8%8F-real-time-visualizer) · [**Docs**](docs/) · [**Contributing**](CONTRIBUTING.md)

</div>

---

## 📌 What Is This?

This repository implements a **complete ADAS perception and planning pipeline** that mirrors the architecture of real automotive systems. It is designed as an engineering portfolio demonstration covering:

| Domain | What is demonstrated |
|---|---|
| **C++ Engineering** | Modern C++17, RAII, smart pointers, templates, PIMPL, modular libraries |
| **Deep Learning Deployment** | PyTorch → ONNX export, INT8/FP16 quantisation, ONNX Runtime, TensorRT |
| **Perception** | YOLOv8-style 2-D object detection, DNN lane detection, letterbox pre/post-processing |
| **Sensor Fusion** | Extended Kalman Filter (EKF) fusing camera, radar, and LiDAR |
| **Multi-Object Tracking** | SORT — Kalman filter prediction + Hungarian data association |
| **Trajectory Prediction** | Constant-Velocity, CTRA, and polynomial prediction (3-second horizon) |
| **ADAS Functions** | ACC, Automatic Emergency Braking (AEB), Lane Keeping Assist (LKA), Traffic-light handler |
| **GPU Programming** | CUDA kernels for NMS and image preprocessing |
| **Python Tooling** | Training (DDP + AMP + W&B), dataset loading, ONNX export, quantisation, evaluation |
| **Real-Time Visualisation** | Pygame GUI — camera view, bird's-eye map, dashboard, live metric plots |
| **Unit Testing** | 30+ Google Test unit tests covering all major components |
| **Build System** | CMake 3.18+, FetchContent, optional CUDA / ONNX Runtime / TensorRT guards |

> **⚠️ Safety Disclaimer:** This is a **demonstration project only** and is _not_ validated for or suitable for deployment in real vehicles. See [Safety Notes](#-safety-notes).

---

## 🖥️ Real-Time Visualizer

Launch a 1280 × 720 live simulation with **zero external models or build tools required** — all sensor data is synthetically generated:

```bash
pip install pygame numpy
python python/visualization/adas_visualizer.py
```

The window is divided into four panels:

```
┌──────────────────────────────────┬─────────────────────────────┐
│  A: CAMERA VIEW                  │  B: BIRD'S-EYE VIEW         │
│  • Detected object bounding boxes│  • Top-down ego + objects   │
│  • Lane boundaries (left/right)  │  • Radar sweep arc          │
│  • Class labels + confidence %   │  • Predicted trajectories   │
│  • Distance markers              │  • Distance rings (10/20m)  │
├──────────────────────────────────┼─────────────────────────────┤
│  C: VEHICLE DASHBOARD            │  D: TIME-SERIES PLOTS       │
│  • Animated speedometer dial     │  • Rolling speed chart      │
│  • Throttle / Brake bar gauges   │  • TTC (Time-to-Collision)  │
│  • AEB / ACC / LKA / TL lights   │  • Lead-vehicle distance    │
│  • Steering wheel indicator      │                             │
└──────────────────────────────────┴─────────────────────────────┘
```

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume simulation |
| `R` | Reset simulation state |
| `H` | Toggle help overlay |
| `Q` / `ESC` | Quit |

---

## ⚙️ System Architecture

The pipeline runs at **30 Hz** and processes data in the following order each frame:

```
┌────────────────────────────────────────────────────────────────────┐
│                         SENSOR LAYER                               │
│                                                                    │
│  Camera (BGR)  ──►  ObjectDetector  ──►  2-D Detections             │
│  Camera        ──►  LaneDetector    ──►  LaneInfo                   │
│  Radar targets ─────────────────────────────────────┐              │
│  LiDAR cloud   ─────────────────────────────────────┤              │
└─────────────────────────────────────────────────────┼──────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼──────────────┐
│                         FUSION LAYER                               │
│                                                                    │
│  SensorFusion (EKF)                                                │
│   • Associates camera detections, radar targets, LiDAR clusters    │
│   • Maintains per-track state [x, y, vx, vy, ax, ay]              │
│   • Outputs confirmed FusionTrack objects                          │
└─────────────────────────────────────────────────────┬──────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼──────────────┐
│                        TRACKING LAYER                               │
│                                                                    │
│  MultiObjectTracker (SORT)                                         │
│   • Kalman filter predicts each track position between frames      │
│   • Hungarian algorithm assigns detections to tracks optimally     │
│   • Tracks are confirmed after minHits, deleted after maxAge       │
└─────────────────────────────────────────────────────┬──────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼──────────────┐
│                       PREDICTION LAYER                              │
│                                                                    │
│  TrajectoryPredictor  (3-second rolling horizon)                   │
│   • ConstantVelocity  — linear extrapolation                       │
│   • ConstantTurnRate  — CTRA curved-path model                     │
│   • PolynomialFit     — cubic spline on position history           │
└─────────────────────────────────────────────────────┬──────────────┘
                                                      │
┌─────────────────────────────────────────────────────▼──────────────┐
│                        PLANNING LAYER                               │
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────┐  ┌───────────────┐  │
│  │     ACC     │  │     AEB      │  │  LKA  │  │ TL Handler    │  │
│  │  Speed PID  │  │ TTC threshld │  │ Lat   │  │ Red/amber     │  │
│  │  Headway    │  │ Full/Partial │  │ PID   │  │ stop control  │  │
│  └──────┬──────┘  └──────┬───────┘  └───┬───┘  └──────┬────────┘  │
│         └────────────────┴──────────────┴──────────────┘           │
│                               ↓                                    │
│                    VehicleControl output                            │
│             { throttle, brake, steering,                           │
│               aebActive, lkaActive, accActive }                    │
└────────────────────────────────────────────────────────────────────┘
```

For a deeper dive see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🗂️ Repository Structure

```
adas_perception_simulation/
│
├── 📄 CMakeLists.txt            # Root CMake — builds adas_core lib + exe + tests
├── 📄 README.md                 # ← You are here
├── 📄 CONTRIBUTING.md           # How to contribute
├── 📄 CHANGELOG.md              # Version history
│
├── 📁 docs/
│   ├── GETTING_STARTED.md       # Step-by-step setup (Linux / macOS / Windows)
│   ├── ARCHITECTURE.md          # EKF, SORT, CUDA, PID deep-dive
│   ├── ADAS_FUNCTIONS.md        # How ACC, AEB, LKA, TL handler each work
│   └── PYTHON_TOOLING.md        # Training, export, quantisation, eval guide
│
├── 📁 include/                  # Public C++ headers (all documented inline)
│   ├── common/
│   │   ├── types.hpp            # DetectedObject, EgoState, CameraFrame, etc.
│   │   ├── logger.hpp           # Thread-safe singleton logger + LOG_* macros
│   │   └── math_utils.hpp       # PID<T>, Kalman1D, NMS, TTC, IoU, wrapAngle
│   ├── inference/
│   │   └── onnx_runner.hpp      # IInferenceRunner, OnnxRunner, TensorRTRunner
│   ├── perception/
│   │   ├── object_detector.hpp  # YOLO-style 2-D detection + pre/post-processing
│   │   ├── lane_detector.hpp    # DNN lane boundary detector + error computation
│   │   └── sensor_fusion.hpp    # EKF multi-sensor fusion
│   ├── tracking/
│   │   ├── multi_object_tracker.hpp  # SORT tracker
│   │   └── hungarian.hpp        # O(n³) Hungarian assignment (header-only)
│   ├── prediction/
│   │   └── trajectory_predictor.hpp  # CV / CTRA / polynomial prediction
│   └── planning/
│       └── adas_controllers.hpp # ACC, AEB, LKA, TrafficLightHandler
│
├── 📁 src/                      # C++ implementation files
│   ├── common/        logger.cpp
│   ├── inference/     onnx_runner.cpp · tensorrt_runner.cpp
│   ├── perception/    object_detector.cpp · lane_detector.cpp
│   │                  sensor_fusion.cpp
│   ├── tracking/      multi_object_tracker.cpp
│   ├── prediction/    trajectory_predictor.cpp
│   ├── planning/      acc_controller.cpp · aeb_controller.cpp
│   │                  lane_keeping_assist.cpp · traffic_light_handler.cpp
│   ├── cuda/          nms_kernel.cu · preprocess_kernel.cu
│   └── main.cpp       # Full 30 Hz pipeline demo (synthetic data)
│
├── 📁 tests/                    # Google Test unit tests (FetchContent auto-setup)
│   ├── CMakeLists.txt
│   ├── test_aeb_controller.cpp  # AEB state machine & TTC-based braking
│   ├── test_acc_controller.cpp  # Speed tracking & safe headway
│   ├── test_math_utils.cpp      # PID, NMS, TTC, IoU, Kalman1D, angle wrap
│   ├── test_sensor_fusion.cpp   # EKF track lifecycle & radar fusion
│   ├── test_tracker.cpp         # SORT ID persistence & track management
│   └── test_lka_predictor.cpp   # LKA steering + CV/CTRA/poly prediction
│
├── 📁 python/
│   ├── visualization/
│   │   └── adas_visualizer.py   # 🖥️ Real-time Pygame GUI (1280×720, 4 panels)
│   ├── model_export/
│   │   ├── export_to_onnx.py    # PyTorch → ONNX + graph simplification
│   │   └── quantize_model.py    # INT8/FP16 quantisation (ORT + TensorRT)
│   ├── training/
│   │   ├── dataset.py           # DetectionDataset, LaneDataset, MultiTaskDataset
│   │   └── train_detector.py    # DDP + AMP + cosine LR warm-up + W&B logging
│   ├── eval/
│   │   └── evaluate_adas.py     # mAP@50, lane accuracy, AEB/ACC metrics, latency
│   └── requirements.txt         # Python dependencies
│
└── 📁 scripts/
    ├── build.sh                 # CMake configure + Ninja build (Linux/macOS)
    └── run_demo.sh              # Auto-build + launch C++ pipeline demo
```

---

## 🚀 Quick Start

### ▶ Option 1 — Python Visualizer (recommended, no build tools needed)

Works on Windows, Linux, and macOS with just Python installed.

```bash
# 1. Clone the repository
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

# 2. Install minimal Python dependencies
pip install pygame numpy

# 3. Launch the live 30 Hz simulation
python python/visualization/adas_visualizer.py

# Optional: set frame-rate and RNG seed
python python/visualization/adas_visualizer.py --fps 60 --seed 123
```

### ▶ Option 2 — Full C++ Pipeline

#### Linux / macOS

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y cmake ninja-build libeigen3-dev g++

# 2. Clone
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

# 3. Build (Release — no GPU/ONNX needed for stub mode)
./scripts/build.sh Release

# 4. Run the 30 Hz synthetic pipeline
./build/Release/adas_demo

# 5. Run unit tests
cd build/Release && ctest --output-on-failure
```

#### Windows (PowerShell)

```powershell
# Requires: Visual Studio 2022, CMake, Eigen3 via vcpkg or manual install
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation

cmake -S . -B build\Release -G "Visual Studio 17 2022" `
      -DCMAKE_BUILD_TYPE=Release `
      -DEIGEN3_DIR="C:\path\to\eigen3\cmake"

cmake --build build\Release --config Release
.\build\Release\adas_demo.exe
```

> See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for full platform-specific installation including optional ONNX Runtime and TensorRT.

---

## 🔩 Prerequisites

### C++ Build

| Dependency | Version | Required | Install |
|---|---|---|---|
| CMake | ≥ 3.18 | ✅ Yes | [cmake.org](https://cmake.org) |
| GCC / Clang / MSVC | GCC 11 / Clang 14 / VS 2022 | ✅ Yes | system package manager |
| Eigen3 | ≥ 3.4 | ✅ Yes | `apt install libeigen3-dev` |
| Ninja | any | ✅ Recommended | `apt install ninja-build` |
| Google Test | v1.14 | ✅ Auto-fetched | CMake FetchContent |
| CUDA Toolkit | ≥ 11.8 | ⚪ Optional | [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads) |
| ONNX Runtime | ≥ 1.16 | ⚪ Optional | [GitHub releases](https://github.com/microsoft/onnxruntime/releases) |
| TensorRT | ≥ 8.6 | ⚪ Optional | [nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) |

> 💡 **Without GPU deps:** the project fully builds and runs in stub mode using synthetic data. All 30+ unit tests pass on CPU.

### Python

```bash
# Full tooling
pip install -r python/requirements.txt

# Visualizer only (minimal)
pip install pygame numpy
```

---

## 🧪 Unit Tests

```bash
./scripts/build.sh Release
cd build/Release

# Run all 30+ tests
ctest --output-on-failure

# Filter by component
./tests/adas_tests --gtest_filter="AEBController.*"
./tests/adas_tests --gtest_filter="ACCController.*"
./tests/adas_tests --gtest_filter="SensorFusion.*"
./tests/adas_tests --gtest_filter="MultiObjectTracker.*"
./tests/adas_tests --gtest_filter="MathUtils.*"
./tests/adas_tests --gtest_filter="LKA.*:TrajectoryPredictor.*"
```

| Test File | Component Tested | Tests |
|---|---|---|
| `test_aeb_controller.cpp` | AEB state machine, TTC thresholds, in-path check | 8 |
| `test_acc_controller.cpp` | Speed control, headway, PID output | 6 |
| `test_math_utils.cpp` | PID, NMS, TTC, IoU, Kalman1D, angle wrap | 12 |
| `test_sensor_fusion.cpp` | Track creation, confirmation, staleness, radar | 7 |
| `test_tracker.cpp` | SORT ID persistence, multi-object, lifecycle | 7 |
| `test_lka_predictor.cpp` | LKA steering, lane errors, prediction models | 9 |

---

## 🐍 Python Tooling

### Real-Time Visualizer

```bash
python python/visualization/adas_visualizer.py [--fps 30] [--seed 42]
```

### Export PyTorch → ONNX

```bash
python python/model_export/export_to_onnx.py \
    --model  path/to/yolov8n.pt \
    --output models/detector.onnx \
    --imgsz  640 --simplify --dynamic
```

### Quantise for Deployment

```bash
# Dynamic INT8 — no calibration data needed
python python/model_export/quantize_model.py \
    --model models/detector.onnx --mode dynamic_int8 \
    --output models/detector_int8.onnx --benchmark

# TensorRT FP16 engine (requires NVIDIA GPU)
python python/model_export/quantize_model.py \
    --model models/detector.onnx --mode trt_fp16 \
    --output models/detector.engine --device 0
```

### Train a Detector

```bash
# Single GPU with Weights & Biases logging
python python/training/train_detector.py \
    --data data/coco --epochs 50 --batch 16 \
    --device cuda:0 --wandb

# Multi-GPU (torchrun DDP)
torchrun --nproc_per_node=4 python/training/train_detector.py \
    --data data/coco --epochs 100 --batch 64
```

### Evaluate

```bash
python python/eval/evaluate_adas.py \
    --detector-model models/detector.onnx \
    --dataset data/val --output reports/eval.json
```

> Full guide: [docs/PYTHON_TOOLING.md](docs/PYTHON_TOOLING.md)

---

## 🔑 Key Design Decisions

| Area | Choice | Rationale |
|---|---|---|
| C++ standard | C++17 | `std::optional`, structured bindings, `if constexpr` |
| Memory | RAII + `unique_ptr` / PIMPL | No raw `new/delete`; stable ABI |
| Linear algebra | Eigen3 | Header-only, SIMD-optimised, zero runtime deps |
| Inference | ONNX Runtime + TensorRT | Portable; TRT for peak GPU throughput |
| Tracking | SORT (Kalman + Hungarian) | Real-time capable, well-understood, extensible |
| Prediction | CV → CTRA → Polynomial | Graduated complexity per motion type |
| Controllers | PID | Deterministic, tunable, analytically bounded |
| Logger | Custom thread-safe singleton | No heavy dep; console + file sinks |
| Build | CMake + FetchContent + Ninja | Reproducible cross-platform |
| Tests | Google Test (auto-fetched) | Industry standard; zero manual install |

---

## 📚 Documentation Index

| Document | Description |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Full installation guide for Linux, macOS, Windows |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | EKF sensor fusion, SORT tracking, CUDA kernels, PID design |
| [docs/ADAS_FUNCTIONS.md](docs/ADAS_FUNCTIONS.md) | How ACC, AEB, LKA, and Traffic-light handler each work |
| [docs/PYTHON_TOOLING.md](docs/PYTHON_TOOLING.md) | Training, export, quantisation, evaluation step-by-step |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Code style guide, branch strategy, PR checklist |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🛡️ Safety Notes

> This is a **research and demonstration project only**.

For production road deployment:
- All safety functions **must** be validated per **ISO 26262** and **ISO 21448 (SOTIF)**.
- Static analysis (clang-tidy, cppcheck, Polyspace) is mandatory.
- FMEA, fault injection testing, and hardware-in-the-loop validation are required.
- Models must be evaluated for sensor degradation, edge cases, and ODD boundaries.
- Watchdog timers, fallback states, and minimum-risk conditions must be implemented.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the code style guide, branch strategy, and PR checklist.

---

## 📝 License

This project is licensed under the **MIT License**.

---

<div align="center">

*Built as an ADAS engineering portfolio demonstration · April 2026*

</div>
