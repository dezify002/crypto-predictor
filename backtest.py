import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config import FEATURE_COLUMNS


# ==========================================
# SETTINGS
# ==========================================

ASSET = "BTC"
TIMEFRAME = 60

MODEL_FILE = Path(
    f"models/xgb_{TIMEFRAME}m.joblib"
)

THRESHOLD_FILE = Path(
    f"models/xgb_{TIMEFRAME}m_threshold.json"
)

DATASET = Path(
    f"datasets/{ASSET.lower()}_questions_{TIMEFRAME}m.parquet"
)

OUTPUT_FILE = Path(
    "logs/backtest_predictions.csv"
)

OUTPUT_FILE.parent.mkdir(
    exist_ok=True,
)


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = joblib.load(MODEL_FILE)

print("Model loaded.")


# ==========================================
# LOAD THRESHOLD
# ==========================================

with open(THRESHOLD_FILE) as f:

    threshold_data = json.load(f)

best_threshold = threshold_data["threshold"]

print(
    f"Decision Threshold: {best_threshold:.2f}"
)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_parquet(DATASET)

print(
    f"Rows: {len(df):,}"
)


# ==========================================
# FEATURES
# ==========================================

X = df[FEATURE_COLUMNS]

y = df["label"]


# ==========================================
# PREDICT
# ==========================================

print("\nRunning predictions...")

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= best_threshold
).astype(int)


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y,
    predictions,
)

precision = precision_score(
    y,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0,
)

roc = roc_auc_score(
    y,
    probabilities,
)

cm = confusion_matrix(
    y,
    predictions,
)

tn, fp, fn, tp = cm.ravel()


# ==========================================
# CONFIDENCE
# ==========================================

confidence = np.where(
    probabilities >= 0.5,
    probabilities,
    1 - probabilities,
)

average_confidence = confidence.mean()


# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = df.copy()

results["probability"] = probabilities

results["prediction"] = predictions

results["correct"] = (
    predictions == y
)

results["threshold"] = best_threshold

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ==========================================
# REPORT
# ==========================================

print()

print("=" * 60)
print("MODEL HEALTH REPORT")
print("=" * 60)

print(f"Asset               : {ASSET}")
print(f"Timeframe           : {TIMEFRAME} minutes")

print()

print(f"Rows Tested         : {len(df):,}")

print(
    f"YES Predictions     : {predictions.sum():,}"
)

print(
    f"NO Predictions      : {(predictions==0).sum():,}"
)

print()

print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"ROC AUC             : {roc:.4f}")

print()

print(
    f"Decision Threshold  : {best_threshold:.2f}"
)

print(
    f"Average Confidence  : {average_confidence:.2%}"
)

print()

print(f"True Positives      : {tp:,}")
print(f"False Positives     : {fp:,}")
print(f"True Negatives      : {tn:,}")
print(f"False Negatives     : {fn:,}")

print()

if roc >= 0.95:

    rating = "OUTSTANDING"

elif roc >= 0.90:

    rating = "EXCELLENT"

elif roc >= 0.85:

    rating = "VERY GOOD"

elif roc >= 0.80:

    rating = "GOOD"

else:

    rating = "NEEDS IMPROVEMENT"

print(
    f"Overall Rating      : {rating}"
)

print()

print(
    f"Predictions Saved   : {OUTPUT_FILE}"
)

print("=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)