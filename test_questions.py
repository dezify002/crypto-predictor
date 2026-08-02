import pandas as pd

from data.features import FeatureEngineer
from data.question_generator import QuestionGenerator

df = pd.read_csv("historical_data/BTC/15m.csv")

engineer = FeatureEngineer()

df = engineer.add_features(df)

generator = QuestionGenerator()

questions = generator.generate(df, "BTC")

print(questions.head())

print()

print("Rows:", len(questions))

print()

print(questions["label"].value_counts(normalize=True))