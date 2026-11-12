"""Plate reader: a small CRNN trained with CTC.

The whole plate crop goes in, the character sequence comes out. No character
segmentation is needed at inference time, and the model learns to skip the blue
"EGYPT / مصر" header band on its own.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .data import BLANK, CLASSES

H, W = 48, 144  # network input size (plates are roughly 2:1; width is stretched to give CTC more steps)


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PlateReader(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 48, hidden: int = 96, dropout: float = 0.2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv(1, w), nn.MaxPool2d(2),                 # 24 x 72
            conv(w, 2 * w), nn.MaxPool2d(2),             # 12 x 36
            conv(2 * w, 4 * w), conv(4 * w, 4 * w), nn.MaxPool2d((2, 1)),   # 6 x 36
            conv(4 * w, 4 * w), nn.MaxPool2d((2, 1)),    # 3 x 36
        )
        self.squeeze = nn.Sequential(nn.Conv2d(4 * w, 4 * w, (3, 1), bias=False), nn.BatchNorm2d(4 * w), nn.ReLU(inplace=True))  # 1 x 36
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(4 * w, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.head = nn.Linear(2 * hidden, n_classes + 1)  # +1 for the CTC blank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N,1,48,144] in 0..1  ->  logits [N, T=36, 28]"""
        f = self.squeeze(self.features(x)).squeeze(2).transpose(1, 2)  # [N, T, C]
        return self.head(self.rnn(self.drop(f))[0])


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL plate crop -> [1,48,144] float tensor (grayscale, per-image contrast stretch)."""
    a = np.asarray(img.convert("L").resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1)
    return torch.from_numpy(a)[None]


def greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """Best-path CTC decoding: collapse repeats, drop blanks."""
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = [], BLANK
        for k in seq:
            if k != prev and k != BLANK:
                s.append(k)
            prev = k
        out.append(s)
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    """Batch augmentation for plate crops: perspective-like affine jitter, blur, exposure, noise.

    No horizontal flip: character order matters and several glyphs are not symmetric.
    """
    n = x.size(0)
    r = lambda s: (torch.rand(n) - 0.5) * 2 * s
    ang, shear, scale = r(np.deg2rad(6)), r(0.25), 1 + r(0.12)
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin + shear, r(0.08)], 1), torch.stack([sin * 3, cos, r(0.1)], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    blur = torch.rand(n) < 0.35
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, 1, 1)
    low = torch.rand(n) < 0.25  # simulate small, far-away plates
    if low.any():
        s = np.random.uniform(0.4, 0.7)
        x[low] = F.interpolate(F.interpolate(x[low], scale_factor=s, mode="bilinear"), size=(H, W), mode="bilinear")
    gain, bias = 1 + r(0.25).view(-1, 1, 1, 1), r(0.12).view(-1, 1, 1, 1)
    x = x * gain + bias + torch.randn_like(x) * 0.03
    inv = torch.rand(n) < 0.03
    x[inv] = 1 - x[inv]
    return x.clamp(0, 1)
