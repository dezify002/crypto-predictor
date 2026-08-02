from pathlib import Path
from datetime import datetime
import pandas as pd


LOG_FILE = Path("logs/predictions.csv")


class PredictionLogger:

    def log(
        self,
        asset,
        current_price,
        target_price,
        minutes,
        model_name,
        probability,
        prediction,
        confidence,
        volatility,
    ):

        now = datetime.utcnow()

        expiry = now + pd.Timedelta(minutes=minutes)

        row = {

            "timestamp": now,

            "asset": asset,

            "current_price": current_price,

            "target_price": target_price,

            "minutes": minutes,

            "model": model_name,

            "probability": probability,

            "prediction": prediction,

            "confidence": confidence,

            "volatility": volatility,

            "status": "Pending",

            "expiry_time": expiry,

            "actual_price": None,

            "correct": None,

        }

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:

            try:
                df = pd.read_csv(LOG_FILE)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()

        else:

            df = pd.DataFrame()

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

        df.to_csv(LOG_FILE, index=False)

        print("\nPrediction saved.")