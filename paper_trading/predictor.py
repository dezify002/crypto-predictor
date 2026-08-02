import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib
import pandas as pd

from data.features import FeatureEngineer
from data.live_market import LiveMarket
from config import FEATURE_COLUMNS

class LivePredictor:

    def __init__(self):

        self.market = LiveMarket()
        self.features = FeatureEngineer()

        self.log_file = Path(
            "logs/live_predictions.csv"
        )

        self.log_file.parent.mkdir(
            exist_ok=True
        )

    def predict(
        self,
        asset,
        target_price,
        minutes,
    ):

        # ------------------------
        # Choose model
        # ------------------------

        if minutes <= 15:
            horizon = 15
        elif minutes <= 30:
            horizon = 30
        elif minutes <= 60:
            horizon = 60
        elif minutes <= 120:
            horizon = 120
        else:
            horizon = 240

        model = joblib.load(
            f"models/xgb_{horizon}m.joblib"
        )

        with open(
            f"models/xgb_{horizon}m_threshold.json"
        ) as f:

            threshold = json.load(f)["threshold"]

        # ------------------------
        # Market Data
        # ------------------------

        df = self.market.get_latest(asset)

        df = self.features.add_features(df)

        latest = df.iloc[-1].copy()

        current_price = float(
            latest["close"]
        )

        latest["required_return"] = (
            target_price - current_price
        ) / current_price

        latest["minutes_remaining"] = minutes

        latest["direction"] = 1

        X = pd.DataFrame(
            [latest[FEATURE_COLUMNS]]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        prediction = (
            probability >= threshold
        )

        self.save_prediction(

            asset=asset,

            current_price=current_price,

            target_price=target_price,

            minutes=minutes,

            probability=probability,

            threshold=threshold,

            prediction=prediction,

        )

        print()

        print("=" * 50)

        print("LIVE PREDICTION")

        print("=" * 50)

        print(f"Asset       : {asset}")

        print(f"Price       : {current_price:.2f}")

        print(f"Target      : {target_price:.2f}")

        print(f"Probability : {probability:.2%}")

        print(f"Prediction  : {prediction}")

        print("=" * 50)

    def save_prediction(

        self,

        asset,

        current_price,

        target_price,

        minutes,

        probability,

        threshold,

        prediction,

    ):

        now = datetime.utcnow()

        expiry = now + timedelta(
            minutes=minutes
        )

        row = {

            "timestamp": now,

            "expiry": expiry,

            "asset": asset,

            "current_price": current_price,

            "target_price": target_price,

            "minutes": minutes,

            "probability": probability,

            "threshold": threshold,

            "prediction": prediction,

            "actual_price": None,

            "correct": None,

        }

        if self.log_file.exists():

            df = pd.read_csv(
                self.log_file
            )

        else:

            df = pd.DataFrame()

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

        df.to_csv(
            self.log_file,
            index=False,
        )


if __name__ == "__main__":

    bot = LivePredictor()

    bot.predict(

        asset="BTC",

        target_price=68000,

        minutes=30,

    )