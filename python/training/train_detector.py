#!/usr/bin/env python3
"""
train_detector.py – Training script for a YOLOv8-style object detector.

Features:
  • Mixed-precision (FP16) training via torch.cuda.amp
  • Distributed training with DDP via torchrun
  • Experiment tracking with wandb (optional)
  • Cosine LR schedule with linear warm-up
  • Automatic checkpointing (best val mAP)

Usage (single GPU):
    python train_detector.py \
        --data data/coco \
        --epochs 50 \
        --batch  16 \
        --imgsz  640 \
        --device cuda:0

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 train_detector.py \
        --data data/coco --epochs 100 --batch 64
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import DetectionDataset, detection_collate

log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Simple stub model (replace with actual YOLO/DETR backbone) ──────────────
class _TinyDetector(nn.Module):
    """Lightweight stub detector head for demonstration."""

    def __init__(self, nc: int = 80, anchors: int = 8400):
        super().__init__()
        self.backbone = nn.Sequential(
            # MobileNet-style depthwise blocks (simplified)
            nn.Conv2d(3, 32, 3, 2, 1),   nn.BatchNorm2d(32),   nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  nn.BatchNorm2d(64),   nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128),  nn.SiLU(),
            nn.Conv2d(128,256, 3, 2, 1), nn.BatchNorm2d(256),  nn.SiLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 25, 1024), nn.SiLU(),
            nn.Linear(1024, (4 + nc) * anchors),
        )
        self._anchors = anchors
        self._nc      = nc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b   = x.shape[0]
        out = self.head(self.backbone(x))
        return out.view(b, 4 + self._nc, self._anchors)


# ─── Loss (simplified YOLO-style) ─────────────────────────────────────────────
class YOLOLoss(nn.Module):
    def __init__(self, nc: int = 80):
        super().__init__()
        self.nc   = nc
        self.bce  = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse  = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor,
                gt_boxes: list[torch.Tensor],
                gt_labels: list[torch.Tensor]) -> torch.Tensor:
        # Dummy loss for demonstration – returns a meaningful scalar
        loss_box = self.mse(pred[:, :4, :], torch.zeros_like(pred[:, :4, :]))
        loss_cls = self.bce(pred[:, 4:, :], torch.zeros_like(pred[:, 4:, :]))
        return loss_box + loss_cls


# ─── LR schedule: linear warm-up + cosine decay ───────────────────────────────
def build_scheduler(optimiser, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    return optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


# ─── Single-epoch training ────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        boxes  = [b.to(device) for b in batch["boxes"]]
        labels = [l.to(device) for l in batch["labels"]]

        optimiser.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(images)
            loss = criterion(pred, boxes, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()

        if step % 50 == 0:
            log.info("  epoch=%d  step=%d/%d  loss=%.4f  lr=%.6f",
                     epoch, step, len(loader), loss.item(),
                     optimiser.param_groups[0]["lr"])

    return total_loss / len(loader)


# ─── Validation (stub mAP calculation) ───────────────────────────────────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    # In production: use pycocotools to compute mAP@0.5:0.95
    return 0.42  # placeholder


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="data",      help="Dataset root")
    parser.add_argument("--epochs",  type=int,  default=50)
    parser.add_argument("--batch",   type=int,  default=16)
    parser.add_argument("--imgsz",   type=int,  default=640)
    parser.add_argument("--lr",      type=float,default=0.01)
    parser.add_argument("--nc",      type=int,  default=80,    help="Number of classes")
    parser.add_argument("--workers", type=int,  default=4)
    parser.add_argument("--device",            default="cuda:0")
    parser.add_argument("--save-dir",          default="runs/detect/train")
    parser.add_argument("--wandb",   action="store_true")
    parser.add_argument("--seed",    type=int,  default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp        = local_rank >= 0
    if ddp:
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)

    is_main = not ddp or local_rank == 0
    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = DetectionDataset(args.data, "train", args.imgsz, args.nc)
    val_ds   = DetectionDataset(args.data, "val",   args.imgsz, args.nc,
                                transform=None)  # no augmentation for val

    sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2,
        num_workers=args.workers, pin_memory=True,
        collate_fn=detection_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _TinyDetector(nc=args.nc).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model: %.2fM parameters", n_params)

    # ── Optimiser + schedule ─────────────────────────────────────────────────
    optimiser = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.937, weight_decay=5e-4, nesterov=True)
    scheduler = build_scheduler(optimiser, warmup_epochs=3, total_epochs=args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = YOLOLoss(nc=args.nc).to(device)

    # ── Optional W&B ─────────────────────────────────────────────────────────
    if args.wandb and is_main:
        try:
            import wandb  # type: ignore
            wandb.init(project="adas-detector", config=vars(args))
        except ImportError:
            log.warning("wandb not installed – run: pip install wandb")

    # ── Training loop ────────────────────────────────────────────────────────
    best_map = 0.0
    for epoch in range(args.epochs):
        if ddp:
            sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, scaler, device, epoch)
        val_map = validate(model, val_loader, device) if is_main else 0.0
        scheduler.step()

        if is_main:
            elapsed = time.time() - t0
            log.info("Epoch %3d/%d  loss=%.4f  mAP50=%.3f  t=%.0fs",
                     epoch, args.epochs, train_loss, val_map, elapsed)

            if args.wandb:
                try:
                    import wandb
                    wandb.log({"train/loss": train_loss,
                               "val/mAP50": val_map, "epoch": epoch})
                except Exception:
                    pass

            # Save best checkpoint
            if val_map > best_map:
                best_map = val_map
                ckpt = model.module if ddp else model
                torch.save(ckpt.state_dict(), save_dir / "best.pt")

            # Save last checkpoint
            ckpt = model.module if ddp else model
            torch.save(ckpt.state_dict(), save_dir / "last.pt")

    if is_main:
        log.info("Training finished. Best mAP50 = %.4f  Saved to: %s",
                 best_map, save_dir)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
