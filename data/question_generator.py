import pandas as pd

from config import FEATURE_COLUMNS


class QuestionGenerator:
    """
    Generates prediction questions.

    direction:
        1  = ABOVE target
       -1  = BELOW target
    """

    def generate(
        self,
        df: pd.DataFrame,
        asset: int,
        horizon_minutes: int,
        target_return: float,
        direction: int = 1,
    ):

        df = df.copy()

        candles_ahead = max(1, horizon_minutes // 15)

        # =========================
        # Current / Target Price
        # =========================

        current_price = df["close"]

        if direction == 1:
            target_price = current_price * (1 + target_return)
        else:
            target_price = current_price * (1 - target_return)

        # =========================
        # Future Window
        # (strictly future candles only — excludes the
        # current/decision candle to avoid look-ahead leakage)
        # =========================

        future_high = (
            df["high"]
            .rolling(window=candles_ahead, min_periods=candles_ahead)
            .max()
            .shift(-candles_ahead)
        )

        future_low = (
            df["low"]
            .rolling(window=candles_ahead, min_periods=candles_ahead)
            .min()
            .shift(-candles_ahead)
        )

        # =========================
        # Labels
        # =========================

        if direction == 1:
            label = (future_high >= target_price).astype(int)
        else:
            label = (future_low <= target_price).astype(int)

        # =========================
        # Question Features
        # =========================

        df["asset"] = asset
        df["target_price"] = target_price
        df["required_return"] = abs(target_return)
        df["minutes_remaining"] = horizon_minutes
        df["direction"] = direction
        df["label"] = label

        # Remove rows without enough future candles
        df = df.iloc[:-candles_ahead].copy()

        # =========================
        # Keep All Training Features
        # =========================

        columns = (
            ["asset", "timestamp"]
            + FEATURE_COLUMNS
            + [
                "target_price",
                "label",
            ]
        )

        return df[columns].reset_index(drop=True)