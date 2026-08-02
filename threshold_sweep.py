import json
import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS


# ==========================================
# SETTINGS
# ==========================================

ASSET = "BTC"

TIMEFRAME = 15

MODEL = f"models/xgb_{TIMEFRAME}m.joblib"

DATASET = (
    f"datasets/{ASSET.lower()}_questions_{TIMEFRAME}m.parquet"
)


# ==========================================
# LOAD MODEL + TEST SET (same split as trainer.py)
# ==========================================

print()
print("=" * 60)
print(f"THRESHOLD SWEEP — {TIMEFRAME} MIN MODEL")
print("=" * 60)

model = joblib.load(MODEL)

df = pd.read_parquet(DATASET)
df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
valid_end = int(n * 0.85)
df = df.iloc[valid_end:]  # held-out test only

print(f"Test Rows : {len(df):,}")

X = df[FEATURE_COLUMNS]
y = df["label"].values

probabilities = model.predict_proba(X)[:, 1]

base_rate = y.mean()

print(f"Base YES Rate (real-world frequency) : {base_rate:.2%}")


# ==========================================
# SWEEP THRESHOLDS
# ==========================================

print()
print(f"{'Threshold':>10} | {'Signals':>8} | {'Precision':>10} | {'Recall':>8} | {'Edge vs Base':>13}")
print("-" * 65)

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:

    predicted_yes = probabilities >= threshold
    actual_yes = y == 1

    tp = (predicted_yes & actual_yes).sum()
    fp = (predicted_yes & ~actual_yes).sum()
    fn = (~predicted_yes & actual_yes).sum()

    n_signals = predicted_yes.sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    edge = precision - base_rate

    print(
        f"{threshold:>10.2f} | {n_signals:>8,} | {precision:>9.2%} | "
        f"{recall:>7.2%} | {edge:>+12.2%}"
    )

print()
print("=" * 60)
print("HOW TO READ THIS")
print("=" * 60)
print("Precision = of the times it said YES at this threshold, % that were actually right")
print("Recall    = of all real YES moves, % that this threshold catches")
print("Edge      = precision minus the real-world base rate (higher = more real signal)")
print("Pick the LOWEST threshold where Edge is still meaningfully positive")
print("and Signals is still a large enough number to trust.")
print("=" * 60)