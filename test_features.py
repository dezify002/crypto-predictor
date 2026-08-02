import pandas as pd

from data.features import FeatureEngineer

df = pd.read_csv("historical_data/BTC/15m.csv")

engineer = FeatureEngineer()

df = engineer.add_features(df)

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())