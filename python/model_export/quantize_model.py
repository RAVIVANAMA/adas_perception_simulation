#!/usr/bin/env python3
"""
quantize_model.py – Post-training quantisation (INT8 / FP16) of an ONNX
model using ONNX Runtime's quantisation toolkit and/or TensorRT.

Supports:
  • ONNX Runtime dynamic INT8 quantisation  (no calibration data needed)
  • ONNX Runtime static INT8 quantisation   (uses a calibration dataset)
  • TensorRT engine build with INT8 / FP16  (requires GPU + TensorRT)

Usage:
    python quantize_model.py \
        --model   models/detector.onnx \
        --mode    dynamic_int8 \
        --output  models/detector_q.onnx

    python quantize_model.py \
        --model   models/detector.onnx \
        --mode    trt_fp16 \
        --output  models/detector.engine \
        --device  0
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

log = logging.getLogger("quantize")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ─── ONNX Runtime dynamic quantisation (CPU / GPU) ───────────────────────────
def quantize_dynamic_int8(input_path: Path, output_path: Path) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore

    log.info("Dynamic INT8 quantisation …")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    log.info("Saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


# ─── ONNX Runtime static quantisation (requires calibration data) ─────────────
def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calib_images: list[np.ndarray],
    imgsz: int = 640,
) -> None:
    from onnxruntime.quantization import (  # type: ignore
        quantize_static, QuantType, CalibrationDataReader,
    )

    class _CalibReader(CalibrationDataReader):
        def __init__(self, images: list[np.ndarray]):
            self._data = iter({"images": img[None]} for img in images)

        def get_next(self):
            return next(self._data, None)

    log.info("Static INT8 quantisation with %d calibration images …", len(calib_images))
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=_CalibReader(calib_images),
        quant_format=QuantType.QInt8,
    )
    log.info("Saved: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


# ─── TensorRT build ───────────────────────────────────────────────────────────
def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    device: int = 0,
    fp16: bool = True,
    int8: bool = False,
    workspace_gb: float = 4.0,
) -> None:
    try:
        import tensorrt as trt  # type: ignore
    except ImportError:
        log.error("TensorRT Python bindings not found. Install: pip install tensorrt")
        raise SystemExit(1)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(trt_logger)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser     = trt.OnnxParser(network, trt_logger)
    config     = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * 1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 enabled")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        log.info("INT8 enabled")

    log.info("Parsing ONNX model …")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    log.info("Building TRT engine (this may take minutes) …")
    t0     = time.time()
    engine = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine)

    log.info("Engine saved: %s  (%.1f MB, %.0fs)", engine_path,
             engine_path.stat().st_size / 1e6, elapsed)


# ─── Benchmark helper ─────────────────────────────────────────────────────────
def benchmark_onnx(model_path: Path, imgsz: int = 640, iterations: int = 100) -> None:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {"images": dummy})

    t0 = time.perf_counter()
    for _ in range(iterations):
        sess.run(None, {"images": dummy})
    avg_ms = (time.perf_counter() - t0) / iterations * 1000

    log.info("Benchmark (%s): avg latency = %.2fms  (%.1f fps)",
             model_path.name, avg_ms, 1000 / avg_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--mode",     default="dynamic_int8",
                        choices=["dynamic_int8", "static_int8",
                                 "trt_fp16",     "trt_int8"])
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--device",   type=int, default=0)
    parser.add_argument("--benchmark",action="store_true")
    args = parser.parse_args()

    inp = Path(args.model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dynamic_int8":
        quantize_dynamic_int8(inp, out)
    elif args.mode == "static_int8":
        # Generate dummy calibration images
        calib = [np.random.rand(3, args.imgsz, args.imgsz).astype(np.float32)
                 for _ in range(64)]
        quantize_static_int8(inp, out, calib, args.imgsz)
    elif args.mode == "trt_fp16":
        build_trt_engine(inp, out, device=args.device, fp16=True, int8=False)
    elif args.mode == "trt_int8":
        build_trt_engine(inp, out, device=args.device, fp16=False, int8=True)

    if args.benchmark and out.suffix == ".onnx":
        benchmark_onnx(out, args.imgsz)


if __name__ == "__main__":
    main()
