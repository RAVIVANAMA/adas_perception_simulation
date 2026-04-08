# 🏗️ System Architecture

This document provides a deep technical description of each layer in the ADAS Perception Simulation Stack.

---

## Table of Contents

1. [High-Level Data Flow](#1-high-level-data-flow)
2. [Common Types and Utilities](#2-common-types-and-utilities)
3. [Inference Backend](#3-inference-backend)
4. [Perception Layer](#4-perception-layer)
5. [EKF Sensor Fusion](#5-ekf-sensor-fusion)
6. [SORT Multi-Object Tracker](#6-sort-multi-object-tracker)
7. [Trajectory Prediction](#7-trajectory-prediction)
8. [ADAS Planning Controllers](#8-adas-planning-controllers)
9. [CUDA Kernels](#9-cuda-kernels)
10. [Build System Design](#10-build-system-design)

---

## 1. High-Level Data Flow

```
Frame N (33 ms at 30 Hz)
│
├─► CameraFrame  ──►  ObjectDetector  ──►  vector<DetectedObject>  ──►─┐
├─► CameraFrame  ──►  LaneDetector    ──►  LaneInfo                    ├─► SensorFusion (EKF)
├─► RadarFrame   ──►  target list     ──►  vector<RadarDetection>  ──►─┤
└─► LidarFrame   ──►  cloud           ──►  vector<LidarPoint>      ──►─┘
                                                                         │
                                                                         ▼
                                                               vector<FusionTrack>
                                                                         │
                                                                         ▼
                                                         MultiObjectTracker (SORT)
                                                                         │
                                                                         ▼
                                                               vector<Track>
                                                                         │
                                                                         ▼
                                                               TrajectoryPredictor
                                                                         │
                                                                         ▼
                                                          map<id, Trajectory>
                                                                         │
                                                    ┌────────────────────┴────────────────────┐
                                                    ▼        ▼            ▼                   ▼
                                              ACC        AEB           LKA           TL Handler
                                                    └────────────────────┬────────────────────┘
                                                                         ▼
                                                                  VehicleControl
                                                       { throttle, brake, steering,
                                                         aebActive, lkaActive, accActive }
```

---

## 2. Common Types and Utilities

### Key Shared Types (`include/common/types.hpp`)

| Type | Fields | Used By |
|---|---|---|
| `BoundingBox2D` | `x, y, w, h, confidence, class_id` | Detector → Tracker |
| `DetectedObject` | `bbox`, `position_3d`, `velocity`, `object_class`, `track_id` | Fusion → Planning |
| `EgoState` | `position`, `velocity`, `acceleration`, `yaw`, `yaw_rate` | All controllers |
| `LaneInfo` | `left_boundary`, `right_boundary`, `lateral_error`, `heading_error`, `curvature` | LKA |
| `VehicleControl` | `throttle [0,1]`, `brake [0,1]`, `steering [-1,1]`, `aebActive`, `lkaActive`, `accActive` | Output |
| `Trajectory` | `vector<TrajectoryPoint>` with `position`, `velocity`, `timestamp` | Predictor → Visualizer |

### PID Controller (`include/common/math_utils.hpp`)

```
Template: PID<T>

u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·(de/dt)

Anti-windup: integral is clamped to [-windup_limit, +windup_limit]
Output:      clamped to [min_output, max_output]
Reset:       flushes integral and previous error to zero
```

### Kalman Filter 1D

```
State:  x (scalar)
Predict: x̂ = F·x  ;  P = F·P·Fᵀ + Q
Update:  K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
         x = x̂ + K·(z - H·x̂)
         P = (I - K·H)·P

F = 1, H = 1 (direct measurement)
Q = process noise, R = measurement noise (user-configurable)
```

---

## 3. Inference Backend

### Design Pattern: PIMPL + Interface

The inference layer uses two design patterns in tandem:

- **Interface (`IInferenceRunner`)** — allows the rest of the stack to be backend-agnostic. Swap between ONNX Runtime and TensorRT with zero changes upstream.
- **PIMPL (`Impl` class)** — hides heavy ONNX Runtime / TensorRT headers from public API. Keeps compile times low and enables a stable ABI.

```cpp
class IInferenceRunner {
public:
    virtual ~IInferenceRunner() = default;
    virtual bool loadModel(const std::string& model_path) = 0;
    virtual bool run(const std::vector<float>& input,
                     const TensorInfo& input_info,
                     std::vector<float>& output,
                     TensorInfo& output_info) = 0;
};
```

### Provider Selection

```
makeRunner() call
│
├─ HAVE_TENSORRT + .engine file?  ──► TensorRTRunner
├─ HAVE_ONNXRUNTIME + .onnx file? ──► OnnxRunner (+ CUDA provider if GPU available)
└─ Neither                        ──► StubRunner (returns synthetic output)
```

### ONNX Runtime Implementation

- `Ort::Session` loaded with `OrtCUDAProviderOptions` when a compatible GPU is present; falls back to CPU execution provider automatically.
- Dynamic axes (`-1` dimensions) resolved from `GetInputTypeInfo()` on first call.
- Output tensor copied to `std::vector<float>` for zero-cost downstream processing.

### TensorRT Implementation

- Engine deserialized with `nvinfer1::createInferRuntime()`.
- Bindings set via `getBindingIndex()` for named input/output tensors.
- `context->executeV2()` for multi-stream compatibility.
- Engine serialized to `.engine` cache file on first load to avoid re-building.

---

## 4. Perception Layer

### Object Detector

```
Input:  CameraFrame (BGR, arbitrary resolution)
Output: vector<DetectedObject>

Pipeline:
1. Letterbox resize → (640×640), maintain aspect ratio, grey padding
2. BGR → RGB, normalize to [0,1], HWC → CHW, float32 tensor
3. IInferenceRunner::run() → [1, 84, 8400] output tensor
   (84 = 4 box coords + 80 class probabilities)
4. Decode: cx,cy,w,h → x1,y1,x2,y2 in input space
5. Confidence filter: max(class_probs) × objectness > threshold
6. Non-Maximum Suppression (IoU threshold configurable)
7. Scale boxes back to original frame resolution
8. Map class index → COCO label string
```

### Lane Detector

```
Input:  CameraFrame (BGR, arbitrary resolution)
Output: LaneInfo

Pipeline:
1. Resize to (512×256), float32 + ImageNet normalisation
2. IInferenceRunner::run() → [1, 2, H, W] segmentation logits
   Channel 0 = left lane, channel 1 = right lane
3. Sigmoid → binary masks
4. Each mask scanned column-by-column for lane pixel centroids
5. Linear regression (Eigen least-squares) on centroids
   → slope, intercept for each boundary
6. computeErrors():
   lateral_error  = image_centre_x − average of left/right lane at ego row
   heading_error  = atan(slope) of lane midline
```

### Sensor Fusion (EKF)

See [Section 5](#5-ekf-sensor-fusion) below.

---

## 5. EKF Sensor Fusion

### State Vector

```
x = [px, py, vx, vy, ax, ay]ᵀ   (6 × 1)
```

where `(px, py)` is position in metres, `(vx, vy)` velocity in m/s, `(ax, ay)` acceleration in m/s².

### Process Model (Constant Acceleration)

```
F(dt) = | 1  0  dt  0   dt²/2  0     |
        | 0  1  0   dt  0      dt²/2  |
        | 0  0  1   0   dt     0      |
        | 0  0  0   1   0      dt     |
        | 0  0  0   0   1      0      |
        | 0  0  0   0   0      1      |

x_pred = F·x_prev
P_pred = F·P_prev·Fᵀ + Q
```

`Q` (process noise) is tuned via `SensorFusionConfig::qAccel` which scales an identity block.

### Camera Measurement Model

Camera detections are associated by 2-D IoU. For an associated detection:

```
H_camera = | 1  0  0  0  0  0 |   (px only)
           | 0  1  0  0  0  0 |   (py only)

z_camera = [px_measured, py_measured]ᵀ
```

### Radar Measurement Model

Radar provides range and velocity. Euclidean gating radius is configurable (`SensorFusionConfig::radarGateRadius` default 5 m).

```
H_radar = | 1  0  0  0  0  0 |   (px)
          | 0  1  0  0  0  0 |   (py)
          | 0  0  1  0  0  0 |   (vx)
          | 0  0  0  1  0  0 |   (vy)
```

### Track Lifecycle

```
minHits = 3   → track goes from Tentative → Confirmed
maxAge  = 5   → frames without update before track deletion
```

---

## 6. SORT Multi-Object Tracker

SORT (Simple, Online and Realtime Tracking) combines:

1. **Per-track Kalman prediction** (8-D state: `[cx, cy, w, h, vcx, vcy, vw, vh]`)  
2. **IoU cost matrix** construction (all tracks × all detections)  
3. **Hungarian algorithm** to find minimum-cost assignment  
4. **Track lifecycle management**

### Kalman State

```
State:    s = [cx, cy, w, h, vcx, vcy, vw, vh]ᵀ
Observation: z = [cx, cy, w, h]ᵀ

F = block diagonal: [constant-velocity model for each dimension]
H = [I₄ | 0₄]  (observe position/size, not velocity)
```

### Assignment

```
cost(i, j) = 1 − IoU(track_i.predicted_bbox, detection_j.bbox)

Threshold: cost > 0.7 → unacceptable match (track and detection treated as new)
```

### Track Lifecycle

```
New detection (no match)    → create Tentative track (hits=1)
Tentative, matched again    → hits++
hits >= minHits             → Confirmed (assigned stable ID)
Confirmed, no match         → age++
age >= maxAge               → delete track
```

---

## 7. Trajectory Prediction

### Constant Velocity (CV)

```
For step k at time t = k·dt:
  px(t) = px_0 + vx_0 · t
  py(t) = py_0 + vy_0 · t

Suitable for: straight highway driving, distant vehicles
```

### Constant Turn Rate and Acceleration (CTRA)

```
State: [px, py, v, yaw, yaw_rate]

For non-zero yaw_rate ω:
  px(t) = px_0 + (v/ω) · ( sin(yaw_0 + ω·t) − sin(yaw_0) )
  py(t) = py_0 + (v/ω) · ( cos(yaw_0) − cos(yaw_0 + ω·t) )

For ω ≈ 0 (straight line):
  px(t) = px_0 + v·cos(yaw_0)·t
  py(t) = py_0 + v·sin(yaw_0)·t

Suitable for: curved roads, intersection turns, lane changes
```

### Polynomial Fit

```
Fits a separate cubic polynomial through the n-point position history:
  x(t) = a₀ + a₁·t + a₂·t² + a₃·t³
  y(t) = b₀ + b₁·t + b₂·t² + b₃·t³

Solved via Eigen least-squares: A·c = b  →  c = (AᵀA)⁻¹·Aᵀb

Suitable for: complex manoeuvres, behaviour-agnostic prediction
Limitation: can diverge at prediction horizon if history is short
```

---

## 8. ADAS Planning Controllers

See [docs/ADAS_FUNCTIONS.md](ADAS_FUNCTIONS.md) for the full functional deep-dive. Summary:

| Controller | Inputs | Key Algorithm | Output |
|---|---|---|---|
| ACC | ego speed, lead vehicle distance/speed | PID (speed error) + headway gap | throttle, brake |
| AEB | ego speed, TTC to all in-path objects | TTC threshold state machine | brake, aebActive flag |
| LKA | lateral_error, heading_error, ego speed | Dual-term PID (lateral + heading) | steering |
| TL Handler | traffic light color, ego distance to stop line | PID (distance to stop) | brake when red/amber |

---

## 9. CUDA Kernels

### NMS Kernel (`src/cuda/nms_kernel.cu`)

```
Grid: 1 block per candidate box pair
Shared memory: IoU matrix cached in shared memory for coalesced access

Algorithm:
1. Candidates sorted by score (host-side, before kernel)
2. Each thread computes IoU(boxes[i], boxes[j])
3. If IoU > threshold and score[j] < score[i] → suppress[j] = true
4. Host reads suppress[] array back and filters output list

Launched via: launchNMS(boxes, scores, nms_threshold, d_output, n)
```

### Preprocess Kernel (`src/cuda/preprocess_kernel.cu`)

```
Grid: ceil(H*W/256) blocks × 1, Threads: 256
Per-thread:
  1. Compute (u, v) from global thread index
  2. Map to source pixel with letterbox offset and scale
  3. Bilinear-interpolate source (clamp to valid region)
  4. Reorder BGR → RGB
  5. Normalise: pixel /= 255.0f; apply per-channel mean/std subtraction
  6. Write to CHW float tensor (output layout for ONNX/TRT)

Launched via: launchPreprocess(src_device, dst_device, src_dims, target_dims, mean, std)
```

Both kernels are conditionally compiled only when `-DHAVE_CUDA=ON` is passed to CMake.

---

## 10. Build System Design

### Library / Executable Split

```
adas_core   (STATIC library)
├── COMMON_SOURCES   (logger, types)
├── INFERENCE_SOURCES
├── PERCEPTION_SOURCES
├── TRACKING_SOURCES
├── PREDICTION_SOURCES
├── PLANNING_SOURCES
└── CUDA_SOURCES (only if HAVE_CUDA)

adas_demo   (EXECUTABLE)
└── src/main.cpp + links adas_core

adas_tests  (EXECUTABLE — tests/ subdirectory)
└── test_*.cpp + links adas_core + GTest::gtest_main
```

### Optional Dependency Guards

```cmake
# Usage example (from CMakeLists.txt):
find_package(onnxruntime QUIET)
if (onnxruntime_FOUND)
    target_link_libraries(adas_core PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(adas_core PUBLIC HAVE_ONNXRUNTIME)
endif()
```

The same pattern applies for CUDA (`enable_language(CUDA)`) and TensorRT. This ensures the project builds cleanly to a functional demo even with no GPU or optional libraries installed.

### FetchContent (Google Test)

```cmake
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
```

No manual GTest installation is needed — CMake downloads and builds it automatically on first configure.
