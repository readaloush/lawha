"""End-to-end check of the web demo in headless Chrome.

    python -m http.server 8000 -d docs &
    python tests/browser_check.py [N]

Runs N held-out vehicle photos through the in-browser pipeline (ONNX detector +
ONNX reader, JavaScript pre/post-processing) and scores them exactly like
lawha.pipeline does for the PyTorch models.
"""
import base64
import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lawha.data import load_plates, split_ids, vehicle_boxes, vehicle_image
from lawha.pipeline import iou

DATA, N = "data/EALPR", int(sys.argv[1]) if len(sys.argv) > 1 else 80
ids = split_ids(DATA)["test"][:N]
truth = {(p.vid, p.index): p.labels for p in load_plates(DATA)}

RUN = """async url => { const img = new Image(); img.src = url; await img.decode();
  const r = await lawha.analyse(await createImageBitmap(img));
  const v = document.getElementById('view');
  return { w: v.width, h: v.height, plates: r.plates.map(p => ({ box: p.box, labels: p.labels })) }; }"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_context(viewport={"width": 1100, "height": 900}, locale="en-US").new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://localhost:8000/")
    page.wait_for_function("document.documentElement.dataset.ready === '1'", timeout=120000)
    print(page.inner_text("#status"))

    total = ok = found = 0
    for vid in ids:
        img = Image.open(vehicle_image(DATA, vid)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        r = page.evaluate(RUN, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        for k, (x, y, w, h) in enumerate(vehicle_boxes(DATA, vid)):
            t = truth.get((vid, k + 1))
            if t is None:
                continue
            gt = ((x - w / 2) * r["w"], (y - h / 2) * r["h"], (x + w / 2) * r["w"], (y + h / 2) * r["h"])
            match = max(r["plates"], key=lambda q: iou(q["box"], gt), default=None)
            hit = match is not None and iou(match["box"], gt) >= 0.5
            total += 1
            found += hit
            ok += hit and match["labels"] == t
    print(f"real test photos: {total} plates, found {found}, found and fully read {ok} ({ok / total:.1%})")

    page.screenshot(path="/tmp/lawha_demo.png")
    print("console errors:", errors or "none")
    browser.close()
