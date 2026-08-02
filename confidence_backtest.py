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
# LOAD MODEL
# ==========================================

print()

print("=" * 60)
print("CONFIDENCE ANALYSIS")
print("=" * 60)

print("Loading model...")

model = joblib.load(MODEL)

with open(THRESHOLD_FILE) as f:

    threshold = json.load(f)["threshold"]

print(
    f"Decision Threshold : {threshold:.2f}"
)

print()

print("Loading dataset...")

df = pd.read_parquet(DATASET)

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
valid_end = int(n * 0.85)

df = df.iloc[valid_end:]   # only the held-out test 15% (matches trainer.py split)

print(f"Rows : {len(df):,}")

X = df[FEATURE_COLUMNS]

y = df["label"]

print()

print("Running predictions...")

probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

correct = (
    predictions == y
)

confidence = np.maximum(
    probabilities,
    1 - probabilities,
)
# ==========================================
# CONFIDENCE BUCKETS
# ==========================================

bins = [

    0.50,

    0.60,

    0.70,

    0.80,

    0.90,

    1.01,

]

labels = [

    "50-60%",

    "60-70%",

    "70-80%",

    "80-90%",

    "90-100%",

]

analysis = pd.DataFrame({

    "confidence": confidence,

    "correct": correct,

    "probability": probabilities,

    "prediction": predictions,

})

analysis["bucket"] = pd.cut(

    analysis["confidence"],

    bins=bins,

    labels=labels,

    right=False,

)

summary = (

    analysis

    .groupby("bucket", observed=False)

    .agg(

        Predictions=("correct", "count"),

        Correct=("correct", "sum"),

        AvgConfidence=("confidence", "mean"),

    )

)

summary["Accuracy"] = (

    summary["Correct"]

    /

    summary["Predictions"]

)

summary["Accuracy"] = (

    summary["Accuracy"]

    .fillna(0)

)

summary["AvgConfidence"] = (

    summary["AvgConfidence"]

    .fillna(0)

)

summary = summary.reset_index()

print()

print("=" * 60)

print("CONFIDENCE BREAKDOWN")

print("=" * 60)

for _, row in summary.iterrows():

    print()

    print(f"Confidence : {row['bucket']}")

    print(f"Predictions: {int(row['Predictions']):,}")

    print(f"Correct    : {int(row['Correct']):,}")

    print(

        f"Accuracy   : {row['Accuracy']:.2%}"

    )

    print(

        f"Avg Conf   : {row['AvgConfidence']:.2%}"

    )
    # ==========================================
# CALIBRATION SCORE
# ==========================================

summary["Difference"] = (
    summary["AvgConfidence"]
    - summary["Accuracy"]
).abs()

calibration_error = (
    summary["Difference"]
    .mean()
)

print()

print("=" * 60)
print("CALIBRATION")
print("=" * 60)

print(
    f"Average Calibration Error : {calibration_error:.2%}"
)

# ==========================================
# MODEL RATING
# ==========================================

if calibration_error <= 0.03:

    rating = "EXCELLENT"

elif calibration_error <= 0.05:

    rating = "VERY GOOD"

elif calibration_error <= 0.08:

    rating = "GOOD"

elif calibration_error <= 0.12:

    rating = "FAIR"

else:

    rating = "POOR"

print(
    f"Confidence Rating         : {rating}"
)

print()

# ==========================================
# SAVE RESULTS
# ==========================================

summary["AvgConfidence"] = (
    summary["AvgConfidence"] * 100
)

summary["Accuracy"] = (
    summary["Accuracy"] * 100
)

summary["Difference"] = (
    summary["Difference"] * 100
)

output_file = "logs/confidence_analysis.csv"

summary.to_csv(
    output_file,
    index=False,
)

print("=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(f"Saved : {output_file}")

print()

print("=" * 60)
print("CONFIDENCE ANALYSIS COMPLETE")
print("=" * 60)