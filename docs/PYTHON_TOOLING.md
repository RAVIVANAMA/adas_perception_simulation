# 🐍 Python Tooling Guide

This document covers all Python scripts in the repository: the **real-time visualizer**, **model export**, **quantisation**, **training**, and **evaluation utilities**.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Real-Time Visualizer](#2-real-time-visualizer)
3. [Dataset Format](#3-dataset-format)
4. [Training a Detector](#4-training-a-detector)
5. [Exporting to ONNX](#5-exporting-to-onnx)
6. [Model Quantisation](#6-model-quantisation)
7. [Evaluation](#7-evaluation)
8. [Dependency Reference](#8-dependency-reference)

---

## 1. Setup

### Minimal (Visualizer Only)

```bash
pip install pygame numpy
```

### Full Tooling

```bash
git clone https://github.com/RAVIVANAMA/adas_perception_simulation.git
cd adas_perception_simulation
pip install -r python/requirements.txt
```

`python/requirements.txt` includes:

```
pygame>=2.5.0
numpy>=1.24
torch>=2.1
torchvision>=0.16
onnx>=1.15
onnxruntime>=1.16
onnxsim>=0.4
wandb                   # optional — comment out if not using W&B
```

Use a virtual environment to keep dependencies isolated:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
pip install -r python/requirements.txt
```

---

## 2. Real-Time Visualizer

**File:** `python/visualization/adas_visualizer.py`

### Launch

```bash
python python/visualization/adas_visualizer.py [--fps 30] [--seed 42]
```

### What It Shows

The 1280×720 window is split into four panels:

| Panel | Content |
|---|---|
| A — Camera View | Bounding boxes, lane overlay, object labels, confidence scores, distance markers |
| B — Bird's-Eye View | Top-down ego + tracked objects, radar arc, trajectory predictions, 10/20 m rings |
| C — Dashboard | Animated speedometer, throttle/brake bars, AEB/ACC/LKA/TL status lights, steering indicator |
| D — Time-Series | Rolling speed plot, TTC (time-to-collision) chart, lead-vehicle distance chart |

### Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume |
| `R` | Reset simulation |
| `H` | Toggle help overlay |
| `Q` / `ESC` | Quit |

### Data Source

All data is **synthetically generated** — no model file or hardware required. The synthetic generators create:

- 2–5 random `DetectedObject` instances per frame with realistic kinematics
- Lane boundaries with natural curvature noise
- Radar sweep with azimuth variation
- Ego vehicle speed profile (acceleration / steady-state / braking scenarios)

### Extending the Visualizer

To connect the visualizer to a live ONNX model output, replace the `_generate_*` methods in the `ADASVisualizer` class with calls to your own perception pipeline. The data structures used are the same `dict` format described in the source file's module docstring.

---

## 3. Dataset Format

**File:** `python/training/dataset.py`

### Detection Dataset (YOLO Format)

```
data/
├── images/
│   ├── train/
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── frame_000001.txt    # One line per object: class cx cy w h (normalised)
    │   └── ...
    └── val/
        └── ...
```

Label file format (per line):
```
<class_id>  <cx_norm>  <cy_norm>  <w_norm>  <h_norm>
```

All values are normalised to [0, 1] relative to image width/height.

### Lane Dataset

```
data/
├── lane_images/
│   └── train/  ...
└── lane_masks/
    └── train/
        └── frame_000001.png    # Binary PNG: 0=background, 1=left lane, 2=right lane
```

### Multi-Task Dataset

`MultiTaskDataset` wraps both datasets and returns:

```python
{
    "image":        torch.Tensor,   # [3, H, W] float32 normalised
    "det_boxes":    torch.Tensor,   # [N, 5] — [class, cx, cy, w, h]
    "lane_mask":    torch.Tensor,   # [H, W] long
}
```

---

## 4. Training a Detector

**File:** `python/training/train_detector.py`

### Single GPU

```bash
python python/training/train_detector.py \
    --data      data/coco              \
    --epochs    50                     \
    --batch     16                     \
    --imgsz     640                    \
    --device    cuda:0                 \
    --lr        0.01                   \
    --output    runs/train/exp1
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 python/training/train_detector.py \
    --data      data/coco              \
    --epochs    100                    \
    --batch     64                     \
    --imgsz     640                    \
    --device    cuda                   \
    --output    runs/train/exp_ddp
```

### With Weights & Biases Logging

```bash
pip install wandb
wandb login
python python/training/train_detector.py ... --wandb --wandb-project adas-detector
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to dataset root |
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size per GPU |
| `--imgsz` | 640 | Input image size (square) |
| `--device` | `cpu` | `cpu`, `cuda:0`, `cuda` (DDP) |
| `--lr` | 0.01 | Initial learning rate |
| `--output` | `runs/` | Output directory for checkpoints |
| `--resume` | — | Path to checkpoint to resume from |
| `--wandb` | False | Enable W&B logging |
| `--wandb-project` | `adas` | W&B project name |

### Training Loop Features

- **AMP** (Automatic Mixed Precision) via `torch.cuda.amp.GradScaler`
- **Cosine LR warm-up** via `torch.optim.lr_scheduler.LambdaLR`
- **DDP** via `torch.nn.parallel.DistributedDataParallel` when `LOCAL_RANK` is set
- Checkpoint saved every epoch as `checkpoint_epoch_N.pt` + `best.pt`

---

## 5. Exporting to ONNX

**File:** `python/model_export/export_to_onnx.py`

### Basic Export

```bash
python python/model_export/export_to_onnx.py \
    --model   runs/train/exp1/best.pt     \
    --output  models/detector.onnx        \
    --imgsz   640
```

### With Graph Simplification

```bash
python python/model_export/export_to_onnx.py \
    --model   runs/train/exp1/best.pt     \
    --output  models/detector.onnx        \
    --imgsz   640                         \
    --simplify
```

`onnxsim` removes redundant ops (e.g., fused BatchNorm → Conv) and generally reduces the graph by 20–40% in node count.

### Dynamic Axes (variable batch / image size at runtime)

```bash
python python/model_export/export_to_onnx.py \
    --model   runs/train/exp1/best.pt     \
    --output  models/detector_dyn.onnx    \
    --imgsz   640                         \
    --dynamic
```

With `--dynamic`, input and output shapes have a `-1` batch dimension, allowing the ONNX Runtime to run variable batch sizes without re-loading the session.

### Validation

The script automatically runs `onnxruntime` inference on a random input after export to verify numerical correctness against PyTorch output (max absolute error logged).

---

## 6. Model Quantisation

**File:** `python/model_export/quantize_model.py`

### Quantisation Mode Comparison

| Mode | Accuracy | Size reduction | Requires GPU | Calibration data |
|---|---|---|---|---|
| FP32 (baseline) | ✅ Highest | — | No | No |
| Dynamic INT8 | ✅ Good | ~4× | No | No |
| Static INT8 | ✅ Best INT8 | ~4× | No | Yes (100–1000 imgs) |
| TRT FP16 | ✅ Good | ~2× | ✅ Yes | No |
| TRT INT8 | ✅ Good | ~4× | ✅ Yes | Yes |

### Dynamic INT8 (No calibration — easiest)

```bash
python python/model_export/quantize_model.py \
    --model   models/detector.onnx         \
    --mode    dynamic_int8                  \
    --output  models/detector_int8.onnx    \
    --benchmark
```

### Static INT8 (With calibration images)

```bash
python python/model_export/quantize_model.py \
    --model         models/detector.onnx    \
    --mode          static_int8             \
    --calib-images  data/val/images/        \
    --output        models/detector_static_int8.onnx \
    --benchmark
```

### TensorRT FP16 (NVIDIA GPU required)

```bash
python python/model_export/quantize_model.py \
    --model   models/detector.onnx         \
    --mode    trt_fp16                      \
    --output  models/detector.engine       \
    --device  0                             \
    --benchmark
```

### Benchmark Output Example

```
[ORT Benchmark] models/detector_int8.onnx
  Warmup: 10 runs
  Benchmark: 100 runs
  Mean latency:  6.3 ms
  P99 latency:  7.1 ms
  Throughput:   158 FPS
```

---

## 7. Evaluation

**File:** `python/eval/evaluate_adas.py`

### Object Detection Evaluation

```bash
python python/eval/evaluate_adas.py \
    --detector-model  models/detector.onnx  \
    --dataset         data/val              \
    --conf-threshold  0.25                  \
    --iou-threshold   0.50                  \
    --output          reports/eval.json
```

### Metrics Reported

| Metric | Description |
|---|---|
| mAP@50 | Mean Average Precision at IoU threshold 0.50 |
| mAP@50-95 | COCO-style mean over IoU thresholds [0.50, 0.55 … 0.95] |
| Per-class AP | AP breakdown for each of the 80 COCO classes |
| Mean latency | Average inference time per image (ms) |
| P99 latency | 99th-percentile inference time |

### How mAP is Computed

1. For each class, detections are sorted by confidence descending.
2. Each detection is matched to a ground-truth box by IoU ≥ threshold.
3. Precision–Recall curve is computed.
4. AP = area under the PR curve (11-point interpolation).
5. mAP = mean AP across all classes.

### ONNX Model Inference in Evaluator

`ONNXModel` wraps `onnxruntime.InferenceSession` with the same letterbox preprocessing used in the C++ `ObjectDetector`:

```python
model = ONNXModel("models/detector.onnx")
results: EvalResults = evaluate_detector(
    model          = model,
    dataset_path   = "data/val",
    conf_threshold = 0.25,
    iou_threshold  = 0.50,
)
print(f"mAP@50: {results.map50:.4f}")
print(f"Mean latency: {results.mean_latency_ms:.1f} ms")
```

---

## 8. Dependency Reference

| Package | Version | Used In | Notes |
|---|---|---|---|
| `pygame` | ≥ 2.5.0 | Visualizer | GUI rendering |
| `numpy` | ≥ 1.24 | All | Array operations |
| `torch` | ≥ 2.1 | Training, Export | DDP, AMP, export |
| `torchvision` | ≥ 0.16 | Training | Dataset utilities |
| `onnx` | ≥ 1.15 | Export | Graph manipulation |
| `onnxruntime` | ≥ 1.16 | Quantise, Eval | CPU/GPU inference |
| `onnxsim` | ≥ 0.4 | Export | Graph simplification |
| `wandb` | any | Training | Optional experiment tracking |

Install all at once:
```bash
pip install -r python/requirements.txt
```
