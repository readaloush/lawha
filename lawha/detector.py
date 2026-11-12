"""Plate detector: fine-tune a pretrained YOLO nano model on EALPR vehicle photos.

    python -m lawha.detector prepare --data data/EALPR --out data/yolo
    python -m lawha.detector train --yolo data/yolo --epochs 60

Uses the Ultralytics package (AGPL-3.0) as a dependency. The dataset is written
with the same vehicle-id split the reader uses.
"""
import argparse
import json
import shutil
from pathlib import Path

from .data import split_ids, vehicle_boxes, vehicle_image


def prepare(data: str, out: str, seed: int = 0) -> Path:
    out = Path(out)
    for name, ids in split_ids(data, seed).items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for vid in ids:
            src = vehicle_image(data, vid)
            dst = out / "images" / name / f"{vid}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy(src, dst)
            boxes = vehicle_boxes(data, vid)
            (out / "labels" / name / f"{vid}.txt").write_text("".join(f"0 {x} {y} {w} {h}\n" for x, y, w, h in boxes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n")
    return out / "data.yaml"


def train(yolo: str, out: str, epochs: int, imgsz: int, weights: str, batch: int, device: str | None):
    from ultralytics import YOLO

    out = Path(out)
    model = YOLO(weights)
    # fliplr off: mirrored plates are not something the reader will ever see downstream.
    model.train(data=str(Path(yolo) / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                project=str(out.resolve() / "runs"), name="detector", exist_ok=True, fliplr=0.0, patience=20, seed=0, plots=True)
    best = out / "runs" / "detector" / "weights" / "best.pt"
    shutil.copy(best, out / "detector.pt")
    m = YOLO(str(best)).val(data=str(Path(yolo) / "data.yaml"), split="test", imgsz=imgsz, device=device, project=str(out.resolve() / "runs"), name="detector_test", exist_ok=True)
    metrics = {"split": "test", "precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "weights": weights, "imgsz": imgsz, "epochs_requested": epochs}
    (out / "detector_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", default="data/EALPR")
    p.add_argument("--out", default="data/yolo")
    t = sub.add_parser("train")
    t.add_argument("--yolo", default="data/yolo")
    t.add_argument("--out", default="results")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--weights", default="yolo11n.pt")
    t.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare":
        print(prepare(a.data, a.out))
    else:
        train(a.yolo, a.out, a.epochs, a.imgsz, a.weights, a.batch, a.device)


if __name__ == "__main__":
    main()
