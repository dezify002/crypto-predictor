import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from config import (
    DATASET_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
)

HORIZONS = [
    15,
    30,
    60,
    120,
    240,
]


class ModelTrainer:

    def load_dataset(self, horizon, direction="above"):

        frames = []

        suffix = "" if direction == "above" else "_below"

        for asset in ["btc", "eth", "sol"]:

            frames.append(
                pd.read_parquet(
                    DATASET_PATH /
                    f"{asset}_questions_{horizon}m{suffix}.parquet"
                )
            )

        df = pd.concat(
            frames,
            ignore_index=True,
        )

        return (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def split(self, df):

        n = len(df)

        train_end = int(n * 0.70)
        valid_end = int(n * 0.85)

        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        test = df.iloc[valid_end:]

        return train, valid, test

    def train_one(self, horizon, direction="above"):

        print(f"\n=== Training {horizon}m model ({direction}) ===")

        df = self.load_dataset(horizon, direction)

        train, valid, test = self.split(df)

        X_train = train[FEATURE_COLUMNS]
        y_train = train["label"]

        X_valid = valid[FEATURE_COLUMNS]
        y_valid = valid["label"]

        X_test = test[FEATURE_COLUMNS]
        y_test = test["label"]

        negatives = (y_train == 0).sum()
        positives = (y_train == 1).sum()

        scale_pos_weight = negatives / max(
            positives,
            1,
        )

        print(
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )

        model = XGBClassifier(

            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,

            random_state=RANDOM_STATE,

            tree_method="hist",

            eval_metric="logloss",

            n_jobs=-1,

        )

        print("Training XGBoost...")

        model.fit(

            X_train,

            y_train,

        )

        print("Searching best threshold...")

        valid_probs = model.predict_proba(

            X_valid

        )[:, 1]

        best_threshold = 0.50
        best_f1 = 0.0

        for threshold in np.arange(

            0.20,
            0.96,
            0.01,

        ):

            predictions = (

                valid_probs >= threshold

            ).astype(int)

            score = f1_score(

                y_valid,

                predictions,

                zero_division=0,

            )

            if score > best_f1:

                best_f1 = score
                best_threshold = float(threshold)

        print()

        print(
            f"Best Threshold : {best_threshold:.2f}"
        )

        print(
            f"Validation F1  : {best_f1:.4f}"
        )

        # ==========================================
        # Test Evaluation
        # ==========================================

        probs = model.predict_proba(

            X_test

        )[:, 1]

        preds = (

            probs >= best_threshold

        ).astype(int)

        print()

        print(
            "Accuracy :",
            accuracy_score(
                y_test,
                preds,
            ),
        )

        print(
            "Precision:",
            precision_score(
                y_test,
                preds,
                zero_division=0,
            ),
        )

        print(
            "Recall   :",
            recall_score(
                y_test,
                preds,
                zero_division=0,
            ),
        )

        print(
            "F1       :",
            f1_score(
                y_test,
                preds,
                zero_division=0,
            ),
        )

        print(
            "ROC AUC  :",
            roc_auc_score(
                y_test,
                probs,
            ),
        )

        save_dir = MODEL_PATH

        save_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        suffix = "" if direction == "above" else "_below"

        model_file = (

            save_dir /

            f"xgb_{horizon}m{suffix}.joblib"

        )

        joblib.dump(

            model,

            model_file,

        )

        threshold_file = (

            save_dir /

            f"xgb_{horizon}m{suffix}_threshold.json"

        )

        with open(

            threshold_file,

            "w",

        ) as f:

            json.dump(

                {

                    "threshold": best_threshold,

                    "validation_f1": best_f1,

                },

                f,

                indent=4,

            )

        print()

        print(

            "Saved Model     :",

            model_file,

        )

        print(

            "Saved Threshold :",

            threshold_file,

        )

    def train(self):

        print()
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        for horizon in HORIZONS:
            self.train_one(horizon, direction="above")
            self.train_one(horizon, direction="below")

        print()
        print("=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 60)


if __name__ == "__main__":

    trainer = ModelTrainer()
    trainer.train()