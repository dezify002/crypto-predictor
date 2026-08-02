import os
import pandas as pd

from config import (
    ASSETS,
    TARGET_RETURNS,
    HORIZONS,
    DATA_PATH,
    DATASET_PATH,
)

from data.features import FeatureEngineer
from data.question_generator import QuestionGenerator


ASSET_IDS = {
    "BTC": 0,
    "ETH": 1,
    "SOL": 2,
}


feature_engineer = FeatureEngineer()
generator = QuestionGenerator()


for asset in ASSETS:

    print(f"\n==============================")
    print(f"Building datasets for {asset}")
    print("==============================")

    file = DATA_PATH / asset / "15m.csv"

    df = pd.read_csv(file)

    df = feature_engineer.add_features(df)

    os.makedirs(DATASET_PATH, exist_ok=True)

    # Build one dataset PER horizon
    for horizon in HORIZONS:

        print(f"\n----- Horizon: {horizon} minutes -----")

        datasets = []

        for target in TARGET_RETURNS:

            print(f"Target Return: {target:.2%}")

            datasets.append(
                generator.generate(
                    df=df,
                    asset=ASSET_IDS[asset],
                    horizon_minutes=horizon,
                    target_return=target,
                    direction=1,   # ONLY ABOVE QUESTIONS
                )
            )

        dataset = pd.concat(
            datasets,
            ignore_index=True,
        )

        output = (
            DATASET_PATH
            / f"{asset.lower()}_questions_{horizon}m.parquet"
        )

        dataset.to_parquet(output)

        print()
        print("Dataset shape:", dataset.shape)
        print("Positive labels:", dataset["label"].mean())
        print("Saved:", output)

print("\n===================================")
print("All datasets generated successfully.")
print("===================================")