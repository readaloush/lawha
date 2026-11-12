"""Figures for the README: training curves, pipeline summary and privacy-masked examples.

    python -m lawha.report --data data/EALPR

Plates in the example figure are real, so the last two digits of every plate are
pixelated in the image and replaced by dots in the printed text.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .data import LETTERS, CLASSES, load_plates, split_ids
from .pipeline import load_reader, read

INK, MUTED, ACCENT, WARN = "#1b1f24", "#6b7280", "#0f766e", "#b45309"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def masked_text(labels: list[int]) -> str:
    """Latin-digit, transliteration-free rendering with the last two digits hidden."""
    digits = [CLASSES[i] for i in labels if i >= len(LETTERS)]
    letters = [CLASSES[i] for i in labels if i < len(LETTERS)]
    shown = digits[:-2] + ["•"] * min(2, len(digits))
    return f"{len(letters)} letters + " + "".join(shown)


def mask_plate(img: Image.Image, plate) -> Image.Image:
    """Pixelates the last two digit boxes (digits are the left block; 'last' = rightmost of them)."""
    img = img.convert("RGB").copy()
    digit_boxes = [b for l, b in zip(plate.labels, plate.boxes) if l >= len(LETTERS)]
    for xc, yc, w, h in digit_boxes[-2:]:
        box = (int((xc - w / 2 - 0.01) * img.width), int((yc - h / 2 - 0.03) * img.height), int((xc + w / 2 + 0.01) * img.width) + 1, int((yc + h / 2 + 0.03) * img.height) + 1)
        patch = img.crop(box)
        img.paste(patch.resize((3, 3), Image.BILINEAR).resize(patch.size, Image.NEAREST), box)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/EALPR")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    style()

    hist = json.loads((out / "reader_history.json").read_text())
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3), dpi=160)
    a1.plot(ep, [h["train_loss"] for h in hist], color=INK, lw=1.8)
    a1.set_yscale("log")
    a1.set_title("Reader: CTC training loss", loc="left", fontsize=10, color=INK)
    a1.set_xlabel("Epoch")
    a2.plot(ep, [h["val_plate_accuracy"] for h in hist], color=ACCENT, lw=2, label="whole plate correct")
    a2.plot(ep, [1 - h["val_cer"] for h in hist], color=MUTED, lw=1.5, label="1 − character error rate")
    a2.set_ylim(0, 1.02)
    a2.set_title("Reader: validation", loc="left", fontsize=10, color=INK)
    a2.set_xlabel("Epoch")
    a2.legend(frameon=False, loc="lower right")
    for ax in (a1, a2):
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "reader_curves.png")

    pm = out / "pipeline_metrics.json"
    if pm.exists():
        m = json.loads(pm.read_text())
        rows = [("Detector recall (IoU 0.5)", m["detector"]["recall"]), ("Detector precision", m["detector"]["precision"]),
                ("Reader, on dataset crops", m["reader_on_dataset_crops"]["plate_accuracy"]), ("End to end: found and fully read", m["end_to_end"]["plate_accuracy"])]
        fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
        ys = range(len(rows))[::-1]
        ax.barh(list(ys), [v for _, v in rows], color=[MUTED, MUTED, MUTED, ACCENT], height=0.62)
        for y, (name, v) in zip(ys, rows):
            ax.text(v - 0.01, y, f"{v:.1%}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
        ax.set_yticks(list(ys), [n for n, _ in rows])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"Test split: {m['vehicles']} vehicles the models never saw", loc="left", fontsize=10, color=INK)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout()
        fig.savefig(out / "pipeline_summary.png")

    # Examples: reader on test crops, 8 correct + 4 wrong, privacy-masked.
    test = set(split_ids(a.data)["test"])
    plates = [p for p in load_plates(a.data) if p.vid in test and sum(l >= len(LETTERS) for l in p.labels) >= 2]
    reader = load_reader(out / "reader.pt")
    preds = [s for i in range(0, len(plates), 128) for s in read(reader, [Image.open(p.image) for p in plates[i:i + 128]])]
    right = [(p, s) for p, s in zip(plates, preds) if s == p.labels][:8]
    wrong = [(p, s) for p, s in zip(plates, preds) if s != p.labels][:4]
    fig, axes = plt.subplots(3, 4, figsize=(9, 5.2), dpi=160)
    for ax, (p, s) in zip(axes.flat, right + wrong):
        ok = s == p.labels
        ax.imshow(mask_plate(Image.open(p.image), p).resize((288, 144)))
        ax.set_title(("read correctly: " if ok else "misread; truth: ") + masked_text(p.labels) + ("" if ok else f"\nmodel: {masked_text(s)}"),
                     fontsize=7.5, color=ACCENT if ok else WARN, loc="left")
        ax.axis("off")
    fig.suptitle("Reader on test plates (last two digits pixelated for privacy)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "examples.png")


if __name__ == "__main__":
    main()
