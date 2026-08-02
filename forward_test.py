from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib
import pandas as pd

from data.features import FeatureEngineer
from data.live_market import LiveMarket
from config import FEATURE_COLUMNS


# ==========================================================
# SETTINGS
# ==========================================================

ASSET = "BTC"

TARGET_PRICE = 68000

MINUTES = 60

LOG_FILE = Path(
    "logs/pending_predictions.csv"
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

if MINUTES <= 15:

    horizon = 15

elif MINUTES <= 30:

    horizon = 30

elif MINUTES <= 60:

    horizon = 60

elif MINUTES <= 120:

    horizon = 120

else:

    horizon = 240

print()

print("=" * 60)
print("FORWARD TEST")
print("=" * 60)

print(f"Using {horizon} Minute Model")

model = joblib.load(
    f"models/xgb_{horizon}m.joblib"
)

with open(
    f"models/xgb_{horizon}m_threshold.json"
) as f:

    threshold = json.load(f)["threshold"]


# ==========================================================
# LIVE DATA
# ==========================================================

market = LiveMarket()

features = FeatureEngineer()

print("Loading market...")

df, source = market.get_latest(ASSET)

print(f"Market Source : {source}")

df = features.add_features(df)

latest = df.iloc[-1].copy()

current_price = float(
    latest["close"]
)


# ==========================================================
# BUILD QUESTION
# ==========================================================

required_return = (

    TARGET_PRICE -
    current_price

) / current_price

latest["required_return"] = required_return

latest["minutes_remaining"] = MINUTES

latest["direction"] = 1

X = pd.DataFrame(
    [latest[FEATURE_COLUMNS]]
)


# ==========================================================
# PREDICT
# ==========================================================

probability = float(

    model.predict_proba(X)[0][1]

)

prediction = (

    probability >= threshold

)

confidence = max(

    probability,

    1 - probability,

)


# ==========================================================
# SAVE
# ==========================================================

now = datetime.utcnow()

expiry = now + timedelta(
    minutes=MINUTES
)

row = {

    "timestamp": now,

    "expiry": expiry,

    "asset": ASSET,

    "current_price": current_price,

    "target_price": TARGET_PRICE,

    "minutes": MINUTES,

    "probability": probability,

    "threshold": threshold,

    "confidence": confidence,

    "prediction": prediction,

    "actual_price": None,

    "correct": None,

}

if LOG_FILE.exists():

    old = pd.read_csv(
        LOG_FILE
    )

else:

    old = pd.DataFrame()

new = pd.concat(

    [

        old,

        pd.DataFrame([row])

    ],

    ignore_index=True,

)

new.to_csv(

    LOG_FILE,

    index=False,

)


# ==========================================================
# REPORT
# ==========================================================

print()

print("=" * 60)
print("NEW PREDICTION")
print("=" * 60)

print(f"Asset         : {ASSET}")

print(f"Current Price : ${current_price:,.2f}")

print(f"Target Price  : ${TARGET_PRICE:,.2f}")

print(f"Window        : {MINUTES} minutes")

print()

print(f"Probability   : {probability:.2%}")

print(f"Threshold     : {threshold:.2f}")

print(f"Confidence    : {confidence:.2%}")

print()

print(

    "Prediction    :",

    "YES ✅" if prediction else "NO ❌"

)

print()

print(

    "Saved To      :",

    LOG_FILE

)

print("=" * 60)