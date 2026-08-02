from learning import LearningEngine
import subprocess

engine = LearningEngine()

df = engine.load()

completed = df.dropna(subset=["correct"])

if len(completed) >= 500:

    print("Retraining model...")

    subprocess.run(["python", "train.py"])

else:

    print(
        f"Only {len(completed)} completed predictions."
    )

    print("Need at least 500 before retraining.")