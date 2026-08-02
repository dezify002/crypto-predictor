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
print("DIFFICULTY BREAKDOWN")
print("=" * 60)

model = joblib.load(MODEL)

with open(THRESHOLD_FILE) as f:
    threshold = json.load(f)["threshold"]

df = pd.read_parquet(DATASET)
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
valid_end = int(n * 0.85)
df = df.iloc[valid_end:]  # held-out test only

print(f"Test Rows : {len(df):,}")

X = df[FEATURE_COLUMNS]
y = df["label"]

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)
correct = (predictions == y)

df = df.copy()
df["correct"] = correct
df["probability"] = probabilities


# ==========================================
# BREAKDOWN BY REQUIRED RETURN
# ==========================================

return_bins = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 1.0]
return_labels = ["0-0.5%", "0.5-1%", "1-2%", "2-3%", "3-5%", "5%+"]

df["return_bucket"] = pd.cut(
    df["required_return"].abs(),
    bins=return_bins,
    labels=return_labels,
    right=False,
)

print()
print("=" * 60)
print("ACCURACY BY REQUIRED RETURN (move size)")
print("=" * 60)

by_return = (
    df.groupby("return_bucket", observed=False)
    .agg(Predictions=("correct", "count"), Accuracy=("correct", "mean"))
)

for bucket, row in by_return.iterrows():
    if row["Predictions"] > 0:
        print(f"{bucket:>8} | n={int(row['Predictions']):>7,} | acc={row['Accuracy']:.2%}")


# ==========================================
# BREAKDOWN BY TIME WINDOW
# ==========================================

print()
print("=" * 60)
print("ACCURACY BY TIME WINDOW (minutes_remaining)")
print("=" * 60)

by_window = (
    df.groupby("minutes_remaining", observed=False)
    .agg(Predictions=("correct", "count"), Accuracy=("correct", "mean"))
)

for window, row in by_window.iterrows():
    if row["Predictions"] > 0:
        print(f"{window:>5} min | n={int(row['Predictions']):>7,} | acc={row['Accuracy']:.2%}")


# ==========================================
# HARDEST SLICE: large move, short window
# ==========================================

print()
print("=" * 60)
print("HARDEST SLICE (>=2% move, <=30 min window)")
print("=" * 60)

hard = df[
    (df["required_return"].abs() >= 0.02) &
    (df["minutes_remaining"] <= 30)
]

if len(hard) > 0:
    print(f"n={len(hard):,} | accuracy={hard['correct'].mean():.2%}")
else:
    print("No rows in this slice for this horizon/timeframe file.")

print()
print("=" * 60)
print("DONE")
print("=" * 60)