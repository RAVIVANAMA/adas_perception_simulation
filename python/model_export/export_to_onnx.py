#!/usr/bin/env python3
"""
export_to_onnx.py – Export a PyTorch detection or lane model to ONNX,
then optionally quantise with ONNX Runtime / TensorRT.

Usage:
    python export_to_onnx.py \
        --model yolov8n.pt \
        --task  detect \
        --imgsz 640 \
        --output models/detector.onnx \
        [--dynamic] [--simplify] [--half]
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("export")


# ─── Tiny stub model (used when no real checkpoint is provided) ───────────────
class _StubDetector(nn.Module):
    """YOLOv8-compatible output stub for testing the export pipeline."""

    def __init__(self, nc: int = 80, anchors: int = 8400):
        super().__init__()
        self.nc      = nc
        self.anchors = anchors
        self.head = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(16, (4 + nc) * anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feat = self.head(x).view(b, -1)
        out  = self.fc(feat).view(b, 4 + self.nc, self.anchors)
        return out


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a PyTorch checkpoint or return a stub if path is empty."""
    if not model_path:
        log.warning("No checkpoint specified – using stub model")
        return _StubDetector().to(device).eval()

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Support ultralytics YOLO `.pt` or plain state-dict
    try:
        from ultralytics import YOLO          # type: ignore
        log.info("Loading via Ultralytics: %s", model_path)
        return YOLO(model_path).model.to(device).eval()
    except ImportError:
        pass

    log.info("Loading via torch.load: %s", model_path)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, nn.Module):
        return state.eval()
    raise ValueError("Unsupported checkpoint format")


def export_onnx(
    model: nn.Module,
    output_path: str,
    imgsz: int = 640,
    batch_size: int = 1,
    dynamic: bool = False,
    opset: int = 17,
    half: bool = False,
) -> Path:
    device = next(model.parameters()).device
    dtype  = torch.float16 if half else torch.float32
    dummy  = torch.zeros(batch_size, 3, imgsz, imgsz, dtype=dtype, device=device)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images":  {0: "batch", 2: "height", 3: "width"},
            "output0": {0: "batch"},
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if half:
        model = model.half()

    log.info("Exporting to ONNX (opset=%d, half=%s, dynamic=%s) …", opset, half, dynamic)
    t0 = time.time()

    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )

    elapsed = time.time() - t0
    log.info("Export complete in %.2fs  →  %s  (%.1f MB)", elapsed, out,
             out.stat().st_size / 1e6)
    return out


def simplify_onnx(path: Path) -> Path:
    try:
        import onnxsim    # type: ignore
        log.info("Simplifying ONNX graph …")
        model, ok = onnxsim.simplify(str(path))
        if ok:
            onnx.save(model, str(path))
            log.info("Simplification succeeded")
        else:
            log.warning("Simplification failed – keeping original")
    except ImportError:
        log.warning("onnx-simplifier not installed; skipping graph simplification")
    return path


def validate_onnx(path: Path, imgsz: int = 640) -> None:
    log.info("Validating ONNX model with ONNX Runtime …")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess  = ort.InferenceSession(str(path), sess_opts, providers=providers)
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    t0  = time.time()
    out = sess.run(None, {"images": dummy})
    dt  = (time.time() - t0) * 1000

    log.info("Runtime check passed  output_shape=%s  latency=%.1fms", out[0].shape, dt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model",    default="",                  help="Path to .pt checkpoint")
    parser.add_argument("--output",   default="models/model.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz",    type=int,   default=640,     help="Input image size")
    parser.add_argument("--batch",    type=int,   default=1,       help="Static batch size")
    parser.add_argument("--opset",    type=int,   default=17)
    parser.add_argument("--dynamic",  action="store_true",         help="Enable dynamic axes")
    parser.add_argument("--simplify", action="store_true",         help="Run onnx-simplifier")
    parser.add_argument("--half",     action="store_true",         help="Export FP16")
    parser.add_argument("--device",   default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model  = load_model(args.model, device)

    onnx_path = export_onnx(
        model, args.output, args.imgsz, args.batch,
        args.dynamic, args.opset, args.half,
    )

    if args.simplify:
        simplify_onnx(onnx_path)

    validate_onnx(onnx_path, args.imgsz)
    log.info("Done → %s", onnx_path)


if __name__ == "__main__":
    main()
