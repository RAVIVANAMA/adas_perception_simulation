# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-04-01

### Added

#### Core Infrastructure
- Thread-safe singleton `Logger` with `LOG_INFO/DEBUG/WARN/ERROR/FATAL` macros and optional file sink
- Template `PID<T>` controller with anti-windup and derivative kick suppression
- `KalmanFilter1D` — single-axis scalar Kalman filter
- IOU computation (`iou(AABB, AABB)`) and templated `nms<Box>` non-maximum suppression
- `computeTTC()`, `wrapAngle()`, `euclidean2D/3D()`, `lerp()`, `clamp()` math utilities
- Shared types header: `DetectedObject`, `EgoState`, `VehicleControl`, `CameraFrame`, `RadarFrame`, `LidarFrame`, `LaneInfo`, `BoundingBox2D/3D`, `ObjectClass`, `TrafficLightColor`, `Trajectory`

#### Inference Backend
- `IInferenceRunner` abstract interface for swappable inference backends
- `OnnxRunner` — full ONNX Runtime 1.16+ implementation with CUDA Execution Provider, dynamic-axis handling, and warmup
- `TensorRTRunner` — TensorRT 8.6+ engine load / serialize / run with `TrtLogger` bridge (conditionally compiled)
- `makeRunner()` factory function with automatic provider selection
- Soft fallback to synthetic output when ONNX Runtime is not installed

#### Perception
- `ObjectDetector` wrapping `IInferenceRunner` — letterbox preprocessing (BGR→RGB, normalise, pad), YOLOv8-format `[B,84,8400]` head decoding, COCO 80-class label mapping, confidence + NMS filtering
- `LaneDetector` wrapping `IInferenceRunner` — resize + ImageNet normalisation preprocessing, segmentation mask postprocessing, linear regression for lateral and heading errors
- `SensorFusion` EKF — 6-D state `[x, y, vx, vy, ax, ay]`, per-step F/Q/H matrices, camera-IoU association, radar Euclidean gate, LiDAR scan integration, track lifecycle (minHits / maxAge)

#### Tracking
- `MultiObjectTracker` (SORT) — 8-D Kalman filter `[cx,cy,w,h,vcx,vcy,vw,vh]` per track
- Hungarian assignment algorithm (header-only O(n³) implementation) on 1-IoU cost matrix
- Track spawning, confirmation, occlusion prediction, and deletion lifecycle management

#### Prediction
- `TrajectoryPredictor` supporting three models:
  - `ConstantVelocity` — linear position extrapolation
  - `ConstantTurnRate` (CTRA) — yaw-rate arc integration
  - `PolynomialFit` — cubic Eigen least-squares fit on position history
- 3-second rolling horizon at configurable time-step resolution

#### ADAS Planning
- `ACCController` — PID-based speed regulation, desired gap = `minGap + timeHeadway × egoSpeed`, throttle / brake output with authority saturation
- `AEBController` — lateral `isInPath()` ±1.8 m gate, TTC thresholds → `Inactive / Warning / PartialBrake / FullBrake` state machine
- `LaneKeepingAssist` — combined lateral-error + heading-error PID, minimum-speed gate, steering bound clamp
- `TrafficLightHandler` — red / amber detection with 30 m lookahead and PID stop-line deceleration

#### CUDA Acceleration (optional)
- `nmsKernel` — GPU non-maximum suppression kernel + `launchNMS` host launcher
- `preprocessKernel` — GPU letterbox + BGR→RGB + normalise + `launchPreprocess` host launcher
- Conditionally compiled via `HAVE_CUDA` CMake option

#### Python Tooling
- `adas_visualizer.py` — 1280×720 Pygame real-time simulation GUI, 4-panel layout (camera view, bird's-eye map, dashboard, time-series plots), keyboard controls, all-synthetic data generators
- `export_to_onnx.py` — PyTorch → ONNX export with `onnxsim` graph simplification and shape validation
- `quantize_model.py` — dynamic INT8, static INT8 (calibration dataset), and TensorRT FP16 engine generation; ORT benchmark harness
- `dataset.py` — `DetectionDataset` (YOLO format), `LaneDataset` (binary masks), `MultiTaskDataset`, offline augmentation builder, detection collate function
- `train_detector.py` — DDP training loop with `torch.nn.parallel.DistributedDataParallel`, AMP `GradScaler`, cosine LR warm-up `LambdaLR`, optional W&B logging, checkpoint saving
- `evaluate_adas.py` — `box_iou_np()`, `compute_ap()`, `ONNXModel` wrapper, `postprocess_detections()`, custom `_nms()`, `evaluate_detector()` with `EvalResults` dataclass

#### Testing
- 30+ Google Test unit tests across 6 test files
- FetchContent auto-setup of GoogleTest v1.14 — no manual installation required

#### Build System
- CMake 3.18+ root file with source sets: `COMMON_SOURCES`, `INFERENCE_SOURCES`, `PERCEPTION_SOURCES`, `TRACKING_SOURCES`, `PREDICTION_SOURCES`, `PLANNING_SOURCES`, `CUDA_SOURCES`
- Optional dependency guards: `HAVE_CUDA`, `HAVE_ONNXRUNTIME`, `HAVE_TENSORRT`
- `adas_core` static library + `adas_demo` executable + `tests/` CTest integration
- `build.sh` and `run_demo.sh` convenience scripts

#### Documentation
- Comprehensive `README.md` with architecture diagram, visualizer layout, quick-start (Python + C++), prerequisites table, test coverage summary, design decisions table
- `docs/GETTING_STARTED.md` — platform-specific installation (Linux / macOS / Windows + vcpkg / MinGW)
- `docs/ARCHITECTURE.md` — EKF maths, SORT algorithm walkthrough, CUDA kernel design, PIMPL pattern rationale
- `docs/ADAS_FUNCTIONS.md` — ACC headway equation, AEB TTC formula and state machine, LKA PID tuning, TL handler braking profile
- `docs/PYTHON_TOOLING.md` — training guide, dataset format, ONNX export options, quantisation mode comparison, evaluation metrics
- `CONTRIBUTING.md`, `CHANGELOG.md`

---

*Older versions: no prior releases — this is the initial public release.*
