"""Export both models to ONNX and check the reader's ONNX output against PyTorch.

    python -m lawha.export
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from .pipeline import load_reader
from .reader import H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)

    reader = load_reader(out / "reader.pt")
    x = torch.rand(1, 1, H, W)  # batch 1: ONNX GRU export is only reliable at the traced batch size
    torch.onnx.export(reader, x, out / "reader.onnx", input_names=["plate"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    got = ort.InferenceSession(str(out / "reader.onnx"), providers=["CPUExecutionProvider"]).run(None, {"plate": x.numpy()})[0]
    with torch.no_grad():
        want = reader(x).numpy()
    print(f"reader.onnx: {(out / 'reader.onnx').stat().st_size / 1e6:.2f} MB, max |torch - onnx| = {np.abs(got - want).max():.2e}")

    if (out / "detector.pt").exists():
        from ultralytics import YOLO
        path = YOLO(str(out / "detector.pt")).export(format="onnx", imgsz=640, opset=17, simplify=False)
        shutil.move(path, out / "detector.onnx")
        print(f"detector.onnx: {(out / 'detector.onnx').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
