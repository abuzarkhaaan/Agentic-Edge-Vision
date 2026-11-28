from __future__ import annotations
import os, math, csv
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25),
    ])


def eval_transform():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_windows(image_paths: Sequence[str], labels: Sequence[int],
                  L: int = 16, stride: int = 8) -> list[dict]:
    N = len(image_paths)
    n_windows = math.ceil(N / stride)
    windows = []
    for w in range(n_windows):
        indices = [(w * stride + j) % N for j in range(L)]
        win_labels = [labels[i] for i in indices]
        fall_count = sum(win_labels)
        maj = 1 if fall_count >= L / 2 else 0
        windows.append({
            "window_id": w,
            "image_indices": indices,
            "label": maj,
        })
    return windows


def build_recording_windows(frame_paths: list[str], frame_labels: list[int],
                             recording_ids: list[str],
                             L: int = 16, stride: int = 8) -> list[dict]:
    from itertools import groupby
    combined = sorted(zip(recording_ids, frame_paths, frame_labels), key=lambda x: x[0])
    windows = []
    wid = 0
    for rec_id, group in groupby(combined, key=lambda x: x[0]):
        items = list(group)
        paths = [x[1] for x in items]
        lbls  = [x[2] for x in items]
        N = len(paths)
        for start in range(0, N - L + 1, stride):
            seg = lbls[start:start + L]
            maj = 1 if sum(seg) >= L / 2 else 0
            windows.append({
                "window_id": wid,
                "recording_id": rec_id,
                "frame_paths": paths[start:start + L],
                "label": maj,
            })
            wid += 1
    return windows


def stratified_split(image_paths: list[str], labels: list[int],
                     val_frac: float = 0.20, test_frac: float = 0.10,
                     seed: int = 42) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        idx = [i for i, l in enumerate(labels) if l == cls]
        idx = rng.permutation(idx).tolist()
        n_test = round(len(idx) * test_frac)
        n_val  = round(len(idx) * val_frac)
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    return train_idx, val_idx, test_idx


def load_split_file(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


CORRUPTION_PARAMS = {
    0: (0.0,  100, 1.00, 0.000),
    1: (0.5,   90, 0.85, 0.001),
    2: (1.0,   75, 0.70, 0.003),
    3: (1.5,   60, 0.55, 0.006),
    4: (2.0,   40, 0.40, 0.010),
    5: (2.5,   20, 0.25, 0.020),
}


def apply_corruption(img: torch.Tensor, corruption: str, severity: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    sigma, jpeg_q, beta, sigma_n = CORRUPTION_PARAMS[severity]
    if severity == 0:
        return img
    if corruption == "blur":
        k = 2 * int(sigma * 2) + 1
        return TF.gaussian_blur(img.unsqueeze(0), [k, k], [sigma, sigma]).squeeze(0)
    if corruption == "jpeg":
        import io
        pil = TF.to_pil_image(img)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=jpeg_q)
        buf.seek(0)
        return TF.to_tensor(Image.open(buf))
    if corruption == "lowlight":
        return (img * beta).clamp(0, 1)
    if corruption == "noise":
        return (img + torch.randn_like(img) * math.sqrt(sigma_n)).clamp(0, 1)
    return img


class WindowDataset(Dataset):
    def __init__(self, windows: list[dict], image_paths: list[str],
                 transform=None, L: int = 16,
                 corruption: str | None = None, severity: int = 0):
        self.windows   = windows
        self.paths     = image_paths
        self.transform = transform or eval_transform()
        self.L         = L
        self.corruption = corruption
        self.severity   = severity

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        frames = []
        for img_idx in w["image_indices"]:
            img = Image.open(self.paths[img_idx]).convert("RGB")
            t   = self.transform(img)
            if self.corruption:
                t = apply_corruption(t, self.corruption, self.severity)
            frames.append(t)
        return torch.stack(frames), w["label"]


class Le2iWindowDataset(Dataset):
    def __init__(self, windows: list[dict], transform=None,
                 corruption: str | None = None, severity: int = 0):
        self.windows    = windows
        self.transform  = transform or eval_transform()
        self.corruption = corruption
        self.severity   = severity

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        frames = []
        for p in w["frame_paths"]:
            img = Image.open(p).convert("RGB")
            t   = self.transform(img)
            if self.corruption:
                t = apply_corruption(t, self.corruption, self.severity)
            frames.append(t)
        return torch.stack(frames), w["label"]
