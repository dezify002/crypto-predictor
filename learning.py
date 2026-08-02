import pandas as pd


class LearningEngine:

    def __init__(self, log_file="logs/predictions.csv"):
        self.log_file = log_file

    def load(self):

        try:
            return pd.read_csv(self.log_file)

        except FileNotFoundError:
            return pd.DataFrame()

    def statistics(self):

        df = self.load()

        if df.empty:
            print("No predictions yet.")
            return

        evaluated = df.dropna(subset=["correct"])

        if evaluated.empty:
            print("No completed predictions.")
            return

        accuracy = evaluated["correct"].mean()

        print("\n==========================")
        print("AI PERFORMANCE")
        print("==========================")

        print(f"Completed Predictions : {len(evaluated)}")
        print(f"Accuracy              : {accuracy:.2%}")

        print()

        high = evaluated[evaluated["probability"] >= 0.80]

        if len(high):

            print(
                f"High Confidence Accuracy : {high['correct'].mean():.2%}"
            )

        low = evaluated[evaluated["probability"] <= 0.20]

        if len(low):

            print(
                f"Strong NO Accuracy       : {low['correct'].mean():.2%}"
            )