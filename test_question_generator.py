import pandas as pd

from data.question_generator import QuestionGenerator
from data.features import FeatureEngineer

df = pd.read_csv("historical_data/BTC/15m.csv")

df = FeatureEngineer().add_features(df)

generator = QuestionGenerator()

dataset = generator.generate(
    df=df,
    asset=0,
    horizon_minutes=60,
    target_return=0.01,
    direction=-1
)
print(dataset.head())

print()

print(dataset.shape)

print()

print(dataset["label"].value_counts(normalize=True))