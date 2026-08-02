import pandas as pd

from data.features import FeatureEngineer
from data.labels import LabelGenerator

df = pd.read_csv("historical_data/BTC/15m.csv")

engineer = FeatureEngineer()
df = engineer.add_features(df)

generator = LabelGenerator()

df = generator.create_labels(
    df,
    horizon=4,      # 4 x 15 minutes = 1 hour
    target_pct=0.01 # +1%
)

print(df[["close", "target_price", "label"]].head())

print("\nLabel Distribution:")
print(df["label"].value_counts(normalize=True))