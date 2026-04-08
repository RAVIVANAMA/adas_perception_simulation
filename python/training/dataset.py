#!/usr/bin/env python3
"""
dataset.py – PyTorch Dataset classes for ADAS perception tasks.

Provides:
  • DetectionDataset   – COCO-format 2-D object detection
  • LaneDataset        – TuSimple / CULane lane segmentation
  • MultiTaskDataset   – Combines both for joint training

Data directories are expected to follow:
    data/
      images/  *.jpg or *.png
      labels/  *.txt  (YOLO format: class cx cy w h per line, normalised)

For lane datasets:
    data/
      images/  *.jpg
      lane_masks/ *.png  (single-channel binary masks)
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

log = logging.getLogger(__name__)


# ─── Augmentation pipeline ────────────────────────────────────────────────────
def build_augment(train: bool, imgsz: int = 640) -> Callable:
    """Build a standard augmentation pipeline."""
    base = [T.Resize((imgsz, imgsz))]
    if train:
        base += [
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomGrayscale(p=0.05),
        ]
    base += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    return T.Compose(base)


# ─── Collate function (variable-length box lists) ────────────────────────────
def detection_collate(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes  = [b["boxes"]  for b in batch]
    labels = [b["labels"] for b in batch]
    return {"image": images, "boxes": boxes, "labels": labels}


# ─── YOLO-format detection dataset ───────────────────────────────────────────
class DetectionDataset(Dataset):
    """
    Loads images and YOLO-format label files.

    Label format (one line per object):
        <class_id> <cx> <cy> <w> <h>   (all normalised to [0,1])
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 640,
        num_classes: int = 80,
        transform: Optional[Callable] = None,
    ):
        self.root     = Path(root)
        self.imgsz    = imgsz
        self.nc       = num_classes
        train         = split == "train"
        self.transform = transform or build_augment(train, imgsz)

        img_dir  = self.root / split / "images"
        self.images = sorted(img_dir.glob("*.jpg")) + \
                      sorted(img_dir.glob("*.png"))

        if not self.images:
            log.warning("No images found in %s", img_dir)

        log.info("DetectionDataset[%s]:  %d images", split, len(self.images))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")

        # Load image via PIL
        from PIL import Image  # lazy import
        image = Image.open(img_path).convert("RGB")

        # Load labels
        boxes, labels = [], []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = list(map(float, line.split()))
                if len(parts) == 5:
                    cls, cx, cy, w, h = parts
                    boxes.append([cx, cy, w, h])
                    labels.append(int(cls))

        if self.transform:
            image = self.transform(image)

        return {
            "image":    image,
            "boxes":    torch.tensor(boxes,  dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long),
            "img_path": str(img_path),
        }


# ─── Lane segmentation dataset ───────────────────────────────────────────────
class LaneDataset(Dataset):
    """
    Loads images and binary lane masks.

    Mask: single-channel PNG where pixel value ≥ 1 = lane.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 640,
        transform: Optional[Callable] = None,
    ):
        self.root    = Path(root)
        self.imgsz   = imgsz
        train        = split == "train"
        self.img_tf  = transform or build_augment(train, imgsz)
        self.mask_tf = T.Compose([T.Resize((imgsz, imgsz)),
                                   T.ToTensor()])

        img_dir = self.root / split / "images"
        msk_dir = self.root / split / "lane_masks"
        self.pairs = [
            (p, msk_dir / (p.stem + ".png"))
            for p in sorted(img_dir.glob("*.jpg"))
            if (msk_dir / (p.stem + ".png")).exists()
        ]
        log.info("LaneDataset[%s]:  %d image-mask pairs", split, len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image
        img_p, msk_p = self.pairs[idx]
        image = Image.open(img_p).convert("RGB")
        mask  = Image.open(msk_p).convert("L")

        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        image = self.img_tf(image)
        torch.manual_seed(seed)
        mask  = (self.mask_tf(mask) > 0.5).float().squeeze(0)

        return {"image": image, "mask": mask, "img_path": str(img_p)}


# ─── Multi-task dataset ───────────────────────────────────────────────────────
class MultiTaskDataset(Dataset):
    """Merge DetectionDataset and LaneDataset for joint training."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 640,
    ):
        self.det  = DetectionDataset(root, split, imgsz)
        self.lane = LaneDataset(root, split, imgsz)
        # Use the shorter dataset's length (can also zip / repeat as needed)
        self._len = min(len(self.det), len(self.lane))
        log.info("MultiTaskDataset[%s]:  %d samples", split, self._len)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        d = self.det[idx]
        l = self.lane[idx % len(self.lane)]
        return {**d, "mask": l["mask"]}


# ─── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os, shutil
    from PIL import Image

    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Create synthetic data
        for split in ("train", "val"):
            (tmpdir / split / "images").mkdir(parents=True)
            (tmpdir / split / "labels").mkdir(parents=True)
            (tmpdir / split / "lane_masks").mkdir(parents=True)
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (480, 640, 3), np.uint8))
                img.save(tmpdir / split / "images" / f"{i:04d}.jpg")
                msk = Image.fromarray(
                    np.random.randint(0, 2, (480, 640), np.uint8) * 255)
                msk.save(tmpdir / split / "lane_masks" / f"{i:04d}.png")
                with open(tmpdir / split / "labels" / f"{i:04d}.txt", "w") as f:
                    f.write("2 0.5 0.5 0.3 0.4\n")

        ds = DetectionDataset(tmpdir, "train", imgsz=320)
        sample = ds[0]
        assert sample["image"].shape == torch.Size([3, 320, 320])
        print("[DetectionDataset] OK – shape:", sample["image"].shape)

        lane_ds = LaneDataset(tmpdir, "train", imgsz=320)
        s2 = lane_ds[0]
        assert s2["mask"].shape == torch.Size([320, 320])
        print("[LaneDataset] OK – mask shape:", s2["mask"].shape)

        mt_ds = MultiTaskDataset(tmpdir, "train", imgsz=320)
        s3 = mt_ds[0]
        print("[MultiTaskDataset] OK – keys:", list(s3.keys()))

    finally:
        shutil.rmtree(tmpdir)
