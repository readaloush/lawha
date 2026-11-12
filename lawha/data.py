"""EALPR dataset access: file discovery, labels, and a leakage-free split.

EALPR (Youssef, Sayed & Ali, 2022) has three linked parts:
  * vehicle photos with YOLO-format plate boxes,
  * the plate crops cut from those photos,
  * per-plate YOLO-format character boxes (27 classes: 17 Arabic letters, 10 digits).
Files share a 4-digit vehicle id, so every split here is made by vehicle id:
a plate crop can never be in training while its source photo is in the test set.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

REPO = "https://github.com/ahmedramadan96/EALPR.git"
VEHICLES = "EALPR Vechicles dataset/Vehicles"            # (sic) folder names as published
VEHICLE_LABELS = "EALPR Vechicles dataset/Vehicles Labeling"
PLATES = "EALPR- Plates dataset"
CHAR_LABELS = "EALPR- LP characters dataset/Characters Labeling"

# Class order from the dataset's classes.txt. Index 14 is written with U+06BE in the
# dataset; plates show the letter heh, so it is displayed as U+0647 here.
LETTERS = list("أبجدرسصطعفقلمنهوى")
DIGITS = list("٠١٢٣٤٥٦٧٨٩")
CLASSES = LETTERS + DIGITS
LATIN_DIGITS = dict(zip(DIGITS, "0123456789"))
BLANK = len(CLASSES)  # CTC blank index


@dataclass
class Plate:
    vid: str                 # vehicle id, e.g. "0412"
    index: int               # plate number within the photo, 1-based
    image: Path              # plate crop
    labels: list[int]        # class ids in visual order, left to right
    boxes: list[tuple[float, float, float, float]]  # matching (xc, yc, w, h), normalised

    @property
    def text(self) -> str:
        return to_text(self.labels)


def to_text(labels: list[int]) -> str:
    """Human-readable plate: digits as printed (left block), then letters in reading order.

    On the plate the letters sit in the right block and are read right to left, so the
    visual left-to-right order is reversed for the letter group.
    """
    digits = [CLASSES[i] for i in labels if i >= len(LETTERS)]
    letters = [CLASSES[i] for i in labels if i < len(LETTERS)]
    return " ".join(letters[::-1]) + " | " + "".join(digits)


def load_plates(root: str | Path) -> list[Plate]:
    root = Path(root)
    plates = []
    for f in sorted((root / CHAR_LABELS).glob("*_license_plate_*.txt")):
        m = re.match(r"(\d+)_license_plate_(\d+)", f.stem)
        rows = [r.split() for r in f.read_text().splitlines() if r.strip()]
        rows = sorted(((int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in rows), key=lambda r: r[1])
        img = root / PLATES / f"{f.stem}.png"
        if rows and img.exists():
            plates.append(Plate(m.group(1), int(m.group(2)), img, [r[0] for r in rows], [r[1:] for r in rows]))
    return plates


def vehicle_ids(root: str | Path) -> list[str]:
    return sorted(p.stem for p in (Path(root) / VEHICLES).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def vehicle_image(root: str | Path, vid: str) -> Path:
    return next(p for p in (Path(root) / VEHICLES).glob(f"{vid}.*"))


def vehicle_boxes(root: str | Path, vid: str) -> list[tuple[float, float, float, float]]:
    f = Path(root) / VEHICLE_LABELS / f"{vid}.txt"
    if not f.exists():
        return []
    return [tuple(map(float, r.split()[1:5])) for r in f.read_text().splitlines() if r.strip()]


def split_ids(root: str | Path, seed: int = 0, val: float = 0.15, test: float = 0.15) -> dict[str, list[str]]:
    """70/15/15 split of vehicle ids, fixed by seed and shared by both models."""
    ids = vehicle_ids(root)
    random.Random(seed).shuffle(ids)
    n_test, n_val = round(len(ids) * test), round(len(ids) * val)
    return {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test:n_test + n_val]), "train": sorted(ids[n_test + n_val:])}
