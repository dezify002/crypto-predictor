import json
import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS


# ==========================================
# SETTINGS
# ==========================================

ASSET = "BTC"

TIMEFRAME = 60

MODEL = f"models/xgb_{TIMEFRAME}m.joblib"

THRESHOLD_FILE = (
    f"models/xgb_{TIMEFRAME}m_threshold.json"
)

DATASET = (
    f"datasets/{ASSET.lower()}_questions_{TIMEFRAME}m.parquet"
)


# ==========================================
# LOAD MODEL + TEST SET (same split as trainer.py)
# ==========================================

print()
print("=" * 60)
print("RECALL / PRECISION BY MOVE SIZE")
print("=" * 60)

model = joblib.load(MODEL)

with open(THRESHOLD_FILE) as f:
    threshold = json.load(f)["threshold"]

df = pd.read_parquet(DATASET)
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
valid_end = int(n * 0.85)
df = df.iloc[valid_end:]  # held-out test only

X = df[FEATURE_COLUMNS]
y = df["label"]

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

df = df.copy()
df["label"] = y.values
df["prediction"] = predictions
df["probability"] = probabilities

return_bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 1.0]
return_labels = ["0-0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5%+"]

df["return_bucket"] = pd.cut(
    df["required_return"].abs(),
    bins=return_bins,
    labels=return_labels,
    right=False,
)

print()
print(f"{'Bucket':>8} | {'n':>8} | {'YES rate':>9} | {'Precision':>9} | {'Recall':>8} | {'Predicted YES%':>15}")
print("-" * 75)

for bucket in return_labels:

    sub = df[df["return_bucket"] == bucket]

    if len(sub) == 0:
        continue

    actual_yes_rate = sub["label"].mean()

    predicted_yes = sub["prediction"] == 1
    actual_yes = sub["label"] == 1

    tp = (predicted_yes & actual_yes).sum()
    fp = (predicted_yes & ~actual_yes).sum()
    fn = (~predicted_yes & actual_yes).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    predicted_yes_rate = predicted_yes.mean()

    print(
        f"{bucket:>8} | {len(sub):>8,} | {actual_yes_rate:>8.2%} | "
        f"{precision:>8.2%} | {recall:>7.2%} | {predicted_yes_rate:>14.2%}"
    )

print()
print("=" * 60)
print("DONE")
print("=" * 60)