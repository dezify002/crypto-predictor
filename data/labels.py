import pandas as pd


class LabelGenerator:
    """
    Generates labels and prediction-specific features.
    """

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 4,
        target_pct: float = 0.01,
    ) -> pd.DataFrame:

        df = df.copy()

        # Highest price within the prediction window
        future_high = (
            df["high"]
            .rolling(window=horizon, min_periods=horizon)
            .max()
            .shift(-horizon + 1)
        )

        # Target price
        df["target_price"] = df["close"] * (1 + target_pct)

        # New Feature 1
        df["required_return"] = (
            df["target_price"] - df["close"]
        ) / df["close"]

        # New Feature 2
        df["minutes_remaining"] = horizon * 15

        # Label
        df["label"] = (
            future_high >= df["target_price"]
        ).astype(int)

        df = df.iloc[:-horizon].reset_index(drop=True)

        return df