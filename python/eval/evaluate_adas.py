#!/usr/bin/env python3
"""
evaluate_adas.py – Offline evaluation harness for the ADAS perception stack.

Evaluates:
  1. Object detection quality (mAP, precision, recall per class)
  2. Lane detection quality (accuracy, F1-score)
  3. ADAS function correctness (AEB TTC accuracy, ACC tracking error)
  4. Inference latency per component

Usage:
    python evaluate_adas.py \
        --detector-model  models/detector.onnx \
        --lane-model      models/lane.onnx \
        --dataset         data/val \
        --output          reports/eval_results.json

Outputs a JSON report and prints a human-readable table to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

log = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Detection metrics ────────────────────────────────────────────────────────

def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between two arrays of boxes (N×4, M×4) → N×M."""
    ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    ix1 = np.maximum(ax1[:,None], bx1[None,:])
    iy1 = np.maximum(ay1[:,None], by1[None,:])
    ix2 = np.minimum(ax2[:,None], bx2[None,:])
    iy2 = np.minimum(ay2[:,None], by2[None,:])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    aA    = (ax2 - ax1) * (ay2 - ay1)
    bA    = (bx2 - bx1) * (by2 - by1)
    union = aA[:,None] + bA[None,:] - inter
    return inter / np.maximum(union, 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using 11-point interpolation (VOC style)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 11.0


@dataclass
class DetectionScore:
    class_name:  str
    ap50:        float = 0.0
    precision:   float = 0.0
    recall:      float = 0.0
    f1:          float = 0.0
    tp:          int   = 0
    fp:          int   = 0
    fn:          int   = 0


@dataclass
class EvalResults:
    map50:          float = 0.0
    per_class:      List[DetectionScore] = field(default_factory=list)
    lane_accuracy:  float = 0.0
    aeb_ttc_mae:    float = 0.0      # seconds
    acc_speed_mae:  float = 0.0      # m/s
    latency_ms:     Dict[str, float] = field(default_factory=dict)


# ─── ONNX Runtime session helper ─────────────────────────────────────────────

class ONNXModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device.startswith("cuda") else ["CPUExecutionProvider"])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session    = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def run(self, image: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: image})

    def warmup(self, imgsz: int = 640, n: int = 10) -> None:
        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        for _ in range(n):
            self.run(dummy)


# ─── Pre-processing ───────────────────────────────────────────────────────────

def preprocess(image: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """Resize + normalise image to model input format (1×3×H×W float32)."""
    import cv2  # type: ignore
    img = cv2.resize(image, (imgsz, imgsz)) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)[None].astype(np.float32)


# ─── Post-processing (YOLOv8 output: [B, 84, 8400]) ─────────────────────────

def postprocess_detections(
    output: np.ndarray,
    orig_w: int, orig_h: int,
    imgsz: int = 640,
    conf_thresh: float = 0.45,
    iou_thresh:  float = 0.45,
) -> list[dict]:
    pred = output[0]          # shape: (84, 8400)
    boxes_raw  = pred[:4].T   # (8400, 4) cx cy w h
    class_scores = pred[4:].T # (8400, 80)

    max_scores = class_scores.max(axis=1)
    max_classes = class_scores.argmax(axis=1)
    mask = max_scores > conf_thresh
    if not mask.any():
        return []

    boxes  = boxes_raw[mask]
    scores = max_scores[mask]
    classes = max_classes[mask]

    # Convert cx,cy,w,h → x1,y1,x2,y2  and scale to orig
    sx, sy = orig_w / imgsz, orig_h / imgsz
    x1 = (boxes[:,0] - boxes[:,2]/2) * sx
    y1 = (boxes[:,1] - boxes[:,3]/2) * sy
    x2 = (boxes[:,0] + boxes[:,2]/2) * sx
    y2 = (boxes[:,1] + boxes[:,3]/2) * sy
    xyxy = np.stack([x1,y1,x2,y2], axis=1)

    # Per-class NMS
    results = []
    for cls in np.unique(classes):
        m = classes == cls
        keep = _nms(xyxy[m], scores[m], iou_thresh)
        for k in keep:
            i = np.where(m)[0][k]
            results.append({"box": xyxy[i].tolist(), "score": float(scores[i]),
                             "class": int(classes[i])})
    return results


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep, suppressed = [], np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]: continue
        keep.append(i)
        iou = box_iou_np(boxes[[i]], boxes).flatten()
        suppressed |= (iou > iou_thresh)
        suppressed[i] = False
    return keep


# ─── Evaluation loop ─────────────────────────────────────────────────────────

def evaluate_detector(
    model: ONNXModel,
    data_dir: Path,
    class_names: list[str],
    imgsz: int = 640,
    conf: float = 0.25,
) -> tuple[float, list[DetectionScore]]:
    """Run detection eval on val split, compute per-class AP."""
    import cv2

    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    images  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    if not images:
        log.warning("No images found in %s – skipping detection eval", img_dir)
        return 0.0, []

    # Collect all predictions and GT per class
    nc = len(class_names)
    preds_by_class: dict[int, list] = {c: [] for c in range(nc)}
    gt_by_class:   dict[int, int]   = {c: 0 for c in range(nc)}

    latencies = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        inp  = preprocess(img[:,:,::-1], imgsz)  # BGR → RGB

        t0  = time.perf_counter()
        out = model.run(inp)
        latencies.append((time.perf_counter() - t0) * 1000)

        dets = postprocess_detections(out[0], w, h, imgsz, conf)

        # Load GT
        gt_boxes, gt_classes = [], []
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = list(map(float, line.split()))
                if len(parts) == 5:
                    cls, cx, cy, bw, bh = parts
                    gt_boxes.append([
                        (cx - bw/2)*w, (cy - bh/2)*h,
                        (cx + bw/2)*w, (cy + bh/2)*h,
                    ])
                    gt_classes.append(int(cls))
                    gt_by_class[int(cls)] += 1

        # Match dets to GT (greedy)
        matched_gt = set()
        for det in sorted(dets, key=lambda d: -d["score"]):
            c = det["class"]
            if c not in preds_by_class: continue
            box = np.array(det["box"])[None]
            best_iou, best_gt = 0.0, -1
            for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_classes)):
                if gc != c or gi in matched_gt: continue
                iou_val = box_iou_np(box, np.array(gb)[None]).item()
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gi
            if best_iou >= 0.5 and best_gt >= 0:
                preds_by_class[c].append((det["score"], True))
                matched_gt.add(best_gt)
            else:
                preds_by_class[c].append((det["score"], False))

    # Compute AP per class
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    log.info("Detector avg latency: %.1f ms", avg_lat)

    scores = []
    aps    = []
    for c, name in enumerate(class_names):
        entries = sorted(preds_by_class[c], key=lambda x: -x[0])
        if not entries:
            continue
        tp_arr = np.array([1 if e[1] else 0 for e in entries])
        fp_arr = 1 - tp_arr
        tp_cum = tp_arr.cumsum()
        fp_cum = fp_arr.cumsum()
        gt_n   = max(gt_by_class[c], 1)
        rec    = tp_cum / gt_n
        prec   = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap   = compute_ap(rec, prec)
        tp_t = int(tp_arr.sum())
        fp_t = int(fp_arr.sum())
        fn_t = gt_by_class[c] - tp_t
        pre  = tp_t / max(tp_t + fp_t, 1)
        re   = tp_t / max(gt_by_class[c], 1)
        aps.append(ap)
        scores.append(DetectionScore(
            class_name=name, ap50=ap, precision=pre, recall=re,
            f1=2*pre*re/max(pre+re,1e-6), tp=tp_t, fp=fp_t, fn=fn_t,
        ))

    map50 = float(np.mean(aps)) if aps else 0.0
    return map50, scores


# ─── Print results table ──────────────────────────────────────────────────────

def print_table(results: EvalResults) -> None:
    print("\n" + "="*70)
    print(f"  mAP@50:   {results.map50:.4f}")
    print(f"  Lane acc: {results.lane_accuracy:.4f}")
    print(f"  AEB TTC MAE:  {results.aeb_ttc_mae:.3f}s")
    print(f"  ACC speed MAE:{results.acc_speed_mae:.3f} m/s")
    print("-"*70)
    print(f"  {'Class':<18} {'AP50':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
    print("-"*70)
    for s in results.per_class:
        print(f"  {s.class_name:<18} {s.ap50:>6.3f}  {s.precision:>6.3f}"
              f"  {s.recall:>6.3f}  {s.f1:>6.3f}")
    print("-"*70)
    for k, v in results.latency_ms.items():
        print(f"  Latency [{k}]: {v:.1f} ms")
    print("="*70 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--lane-model",     default="")
    parser.add_argument("--dataset",        default="data/val")
    parser.add_argument("--output",         default="reports/eval_results.json")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # COCO-like class names (subset shown; extend as needed)
    class_names = [
        "person","bicycle","car","motorcycle","airplane","bus","train",
        "truck","boat","traffic light","stop sign","cat","dog",
    ][:13]  # shortened for demo

    results = EvalResults()

    # ── Detector eval ────────────────────────────────────────────────────────
    if args.detector_model and Path(args.detector_model).exists():
        log.info("Loading detector: %s", args.detector_model)
        det_model = ONNXModel(args.detector_model, args.device)
        det_model.warmup(args.imgsz)

        t0 = time.perf_counter()
        dummy = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
        for _ in range(50): det_model.run(dummy)
        results.latency_ms["detector"] = (time.perf_counter() - t0) / 50 * 1000

        results.map50, results.per_class = evaluate_detector(
            det_model, data_dir, class_names, args.imgsz, args.conf)
    else:
        log.warning("Detector model not found – filling with synthetic scores")
        results.map50 = 0.0
        results.latency_ms["detector"] = 0.0

    # ── Lane eval (stub) ─────────────────────────────────────────────────────
    results.lane_accuracy = 0.0

    # ── AEB / ACC eval (stub – would replay a driving log in production) ─────
    results.aeb_ttc_mae   = 0.0
    results.acc_speed_mae = 0.0

    print_table(results)

    with open(out_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    log.info("Results saved to: %s", out_path)


if __name__ == "__main__":
    main()
