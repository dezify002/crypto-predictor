from pathlib import Path
import pandas as pd

from features import FeatureEngineer
from labels import LabelGenerator


class DatasetBuilder:

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.labeler = LabelGenerator()

    def build(
        self,
        csv_path,
        horizon,
        target_pct,
    ):

        df = pd.read_csv(csv_path)

        df = self.engineer.add_features(df)

        df = self.labeler.create_labels(
            df,
            horizon=horizon,
            target_pct=target_pct,
        )

        return df


if __name__ == "__main__":

    builder = DatasetBuilder()

    datasets = [
        (4, 0.005),   # +0.5% in 1h
        (4, 0.01),    # +1%
        (8, 0.015),   # +1.5% in 2h
        (16, 0.02),   # +2% in 4h
    ]

    Path("datasets").mkdir(exist_ok=True)

    for horizon, target in datasets:

        df = builder.build(
            "../historical_data/BTC/15m.csv",
            horizon,
            target,
        )

        filename = (
            f"datasets/"
            f"btc_h{horizon}_t{int(target*1000)}.csv"
        )

        df.to_csv(filename, index=False)

        print(
            filename,
            df.shape,
            df["label"].mean(),
        )