"""Train the CRNN plate reader on EALPR plate crops.

    python -m lawha.train_reader --data data/EALPR --epochs 60
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .data import BLANK, load_plates, split_ids
from .metrics import score
from .reader import PlateReader, augment, greedy_decode, preprocess


def tensors(plates):
    x = torch.stack([preprocess(Image.open(p.image)) for p in plates])
    return x, [p.labels for p in plates]


@torch.no_grad()
def evaluate(model, x, targets):
    model.eval()
    preds = [s for b in x.split(256) for s in greedy_decode(model(b))]
    return score(preds, targets), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/EALPR")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    split = {k: set(v) for k, v in split_ids(a.data, a.seed).items()}
    plates = load_plates(a.data)
    xt, yt = tensors([p for p in plates if p.vid in split["train"]])
    xv, yv = tensors([p for p in plates if p.vid in split["val"]])
    print(f"train {len(yt)} plates, val {len(yv)} plates, device {dev}")

    model = PlateReader().to(dev)
    print(f"PlateReader: {sum(p.numel() for p in model.parameters()):,} parameters")
    steps = a.epochs * (len(yt) // a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    history, best = [], -1.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            xb = augment(xt[idx].clone()).to(dev)
            tgt = [yt[j] for j in idx.tolist()]
            logp = model(xb).log_softmax(-1).transpose(0, 1)  # [T, N, C]
            loss = ctc(logp, torch.tensor([c for t in tgt for c in t]), torch.full((len(tgt),), logp.size(0)), torch.tensor([len(t) for t in tgt]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
        m, _ = evaluate(model, xv.to(dev), yv)
        history.append({"epoch": epoch, "train_loss": total / (len(perm) // a.batch), "val_plate_accuracy": m["plate_accuracy"], "val_cer": m["cer"]})
        flag = ""
        if m["plate_accuracy"] - m["cer"] > best:
            best = m["plate_accuracy"] - m["cer"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": m}, out / "reader.pt")
            flag = " *"
        print(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.3f}  val plate acc {m['plate_accuracy']:.4f}  CER {m['cer']:.4f}  {time.time()-t0:.0f}s{flag}", flush=True)
    (out / "reader_history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
