"""End to end: photo -> plate boxes -> plate text, and its evaluation on the test split.

    python -m lawha.pipeline --data data/EALPR

Reports three things on the held-out vehicles:
  * reader alone, on the dataset's own plate crops,
  * detector alone, at IoU 0.5,
  * the full pipeline: a plate counts as correct only if it was found AND every
    character was read correctly.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import load_plates, split_ids, to_text, vehicle_boxes, vehicle_image
from .metrics import edit_distance, score
from .reader import PlateReader, greedy_decode, preprocess

PAD = 0.10  # the dataset's plate crops include roughly this much margin around the labelled box


def load_reader(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = PlateReader().to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def read(reader, crops: list[Image.Image]) -> list[list[int]]:
    if not crops:
        return []
    dev = next(reader.parameters()).device
    return greedy_decode(reader(torch.stack([preprocess(c) for c in crops]).to(dev)))


def crop_padded(img: Image.Image, box, pad: float = PAD) -> Image.Image:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad / 2, (y1 - y0) * pad / 2
    return img.crop((max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(img.width, int(x1 + dx)), min(img.height, int(y1 + dy))))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Pipeline:
    def __init__(self, detector_path: str, reader_path: str, conf: float = 0.25, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        self.detector, self.reader = YOLO(detector_path), load_reader(reader_path, device)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, img: Image.Image) -> list[dict]:
        img = img.convert("RGB")
        r = self.detector.predict(img, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        found = [(tuple(b), c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()) if b[2] - b[0] >= 8 and b[3] - b[1] >= 4]
        seqs = read(self.reader, [crop_padded(img, b) for b, _ in found])
        return [{"box": b, "confidence": c, "labels": s, "text": to_text(s)} for (b, c), s in zip(found, seqs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/EALPR")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    ids = split_ids(a.data)[a.split]
    truth = {(p.vid, p.index): p for p in load_plates(a.data) if p.vid in set(ids)}

    # 1. reader alone, on the published crops
    reader = load_reader(out / "reader.pt", a.device)
    plates = list(truth.values())
    preds = [s for i in range(0, len(plates), 128) for s in read(reader, [Image.open(p.image) for p in plates[i:i + 128]])]
    reader_m = score(preds, [p.labels for p in plates])
    by_len = {}
    for p, t in zip(preds, plates):
        d = by_len.setdefault(len(t.labels), [0, 0])
        d[0] += p == t.labels
        d[1] += 1
    reader_m["accuracy_by_length"] = {str(k): {"plates": n, "accuracy": c / n} for k, (c, n) in sorted(by_len.items())}

    # 2 + 3. detector, then the whole chain
    pipe = Pipeline(str(out / "detector.pt"), str(out / "reader.pt"), device=a.device)
    tp = fp = n_gt = e2e_ok = e2e_total = 0
    dists, chars, errors = 0, 0, []
    for vid in ids:
        img = Image.open(vehicle_image(a.data, vid)).convert("RGB")
        gts = [((x - w / 2) * img.width, (y - h / 2) * img.height, (x + w / 2) * img.width, (y + h / 2) * img.height) for x, y, w, h in vehicle_boxes(a.data, vid)]
        dets = sorted(pipe(img), key=lambda d: -d["confidence"])
        n_gt += len(gts)
        used = set()
        match = {}
        for d in dets:
            best = max(((iou(d["box"], g), k) for k, g in enumerate(gts) if k not in used), default=(0, None))
            if best[0] >= 0.5:
                used.add(best[1])
                match[best[1]] = d
                tp += 1
            else:
                fp += 1
        for k in range(len(gts)):
            t = truth.get((vid, k + 1))
            if t is None:      # plate has a box but no character labels: detection is scored, reading is not
                continue
            e2e_total += 1
            d = match.get(k)
            got = d["labels"] if d else []
            dists += edit_distance(got, t.labels)
            chars += len(t.labels)
            if got == t.labels:
                e2e_ok += 1
            else:
                errors.append({"vehicle": vid, "found": d is not None, "truth": t.text, "read": to_text(got) if d else None})

    result = {
        "split": a.split, "vehicles": len(ids),
        "reader_on_dataset_crops": reader_m,
        "detector": {"ground_truth_plates": n_gt, "true_positives": tp, "false_positives": fp,
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1), "iou_threshold": 0.5, "confidence_threshold": pipe.conf},
        "end_to_end": {"plates": e2e_total, "plate_accuracy": e2e_ok / max(e2e_total, 1), "cer": dists / max(chars, 1),
                       "missed_by_detector": sum(not e["found"] for e in errors), "misread": sum(e["found"] for e in errors)},
    }
    (out / "pipeline_metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out / "pipeline_errors.json").write_text(json.dumps(errors, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
