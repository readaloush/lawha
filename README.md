# Lawha لوحة

Arabic licence plate recognition in two stages: a fine-tuned YOLO detector finds the plate, and a small CRNN trained with CTC reads its Arabic letters and digits.

Work in progress: results, figures and a demo will be added once training on Kaggle finishes.

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/ahmedramadan96/EALPR.git data/EALPR
python -m lawha.detector prepare
python -m lawha.train_reader --epochs 80
python -m lawha.detector train --epochs 60
python -m lawha.pipeline
```

Data: EALPR, Youssef, Sayed & Ali, *A New Benchmark Dataset for Egyptian License Plate Detection and Recognition*, ACIRS 2022. The dataset is not redistributed here.
