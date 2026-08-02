from datetime import datetime
from pathlib import Path

import pandas as pd

from data.live_market import LiveMarket


LOG_FILE = Path("logs/predictions.csv")

market = LiveMarket()


def get_current_price(asset: str):
    """
    Returns the latest market price.
    Falls back to local historical data if live data fails.
    """

    try:
        df = market.get_latest(asset)
        return float(df.iloc[-1]["close"])

    except Exception:

        file = Path(f"historical_data/{asset}/15m.csv")

        df = pd.read_csv(file)

        return float(df.iloc[-1]["close"])


def evaluate():

    if not LOG_FILE.exists():
        print("No prediction log found.")
        return

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        print("Prediction log is empty.")
        return

    now = datetime.now()

    updated = 0

    for i, row in df.iterrows():

        # Skip predictions already checked
        if pd.notna(row["correct"]):
            continue

        expiry = pd.to_datetime(row["expiry_time"])

        # Not expired yet
        if expiry > now:
            continue

        asset = row["asset"]

        current_price = get_current_price(asset)

        df.at[i, "actual_price"] = current_price

        target = row["target_price"]

        if current_price >= target:
            df.at[i, "correct"] = 1
        else:
            df.at[i, "correct"] = 0

        updated += 1

    df.to_csv(LOG_FILE, index=False)

    print(f"\nUpdated {updated} prediction(s).")


if __name__ == "__main__":
    evaluate()