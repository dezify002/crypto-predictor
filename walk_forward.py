from pathlib import Path
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
    FEATURE_COLUMNS,
    RANDOM_STATE,
)


# ==========================================
# SETTINGS
# ==========================================

ASSET = "BTC"

TIMEFRAME = 60

DATASET = Path(
    f"datasets/{ASSET.lower()}_questions_{TIMEFRAME}m.parquet"
)

RESULTS_FILE = Path(
    "logs/walk_forward_results.csv"
)

RESULTS_FILE.parent.mkdir(
    exist_ok=True
)


INITIAL_TRAIN_SIZE = 0.50

TEST_SIZE = 0.10


# ==========================================
# LOAD DATA
# ==========================================

print()

print("=" * 60)
print("WALK FORWARD TEST")
print("=" * 60)

df = pd.read_parquet(DATASET)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(f"Rows Loaded : {len(df):,}")

results = []
# ==========================================
# WALK FORWARD LOOP
# ==========================================

n = len(df)

train_end = int(
    n * INITIAL_TRAIN_SIZE
)

test_size = int(
    n * TEST_SIZE
)

fold = 1

while train_end + test_size <= n:

    print()
    print("=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    train = df.iloc[:train_end]

    test = df.iloc[
        train_end:
        train_end + test_size
    ]

    print(
        f"Train Rows : {len(train):,}"
    )

    print(
        f"Test Rows  : {len(test):,}"
    )

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train["label"]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test["label"]

    negatives = (
        y_train == 0
    ).sum()

    positives = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negatives /
        max(positives, 1)
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

    print(
        "Training model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]
        # ==========================================
    # FIND BEST THRESHOLD
    # ==========================================

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in np.arange(
        0.20,
        0.96,
        0.01,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(threshold)

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC AUC   : {roc:.4f}"
    )

    results.append({

        "fold": fold,

        "train_rows": len(train),

        "test_rows": len(test),

        "threshold": best_threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc,

    })

    fold += 1

    train_end += test_size
    # ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame(results)

results.to_csv(
    RESULTS_FILE,
    index=False,
)

print()

print("=" * 60)
print("WALK FORWARD SUMMARY")
print("=" * 60)

print(
    f"Total Folds : {len(results)}"
)

print()

print(
    f"Average Accuracy : {results['accuracy'].mean():.4f}"
)

print(
    f"Average Precision: {results['precision'].mean():.4f}"
)

print(
    f"Average Recall   : {results['recall'].mean():.4f}"
)

print(
    f"Average F1 Score : {results['f1'].mean():.4f}"
)

print(
    f"Average ROC AUC  : {results['roc_auc'].mean():.4f}"
)

print(
    f"Average Threshold: {results['threshold'].mean():.2f}"
)

print()

print("=" * 60)
print("BEST FOLD")
print("=" * 60)

best = results.loc[
    results["f1"].idxmax()
]

print(
    f"Fold       : {int(best['fold'])}"
)

print(
    f"Accuracy   : {best['accuracy']:.4f}"
)

print(
    f"Precision  : {best['precision']:.4f}"
)

print(
    f"Recall     : {best['recall']:.4f}"
)

print(
    f"F1 Score   : {best['f1']:.4f}"
)

print(
    f"ROC AUC    : {best['roc_auc']:.4f}"
)

print(
    f"Threshold  : {best['threshold']:.2f}"
)

print()

print(
    f"Results saved to: {RESULTS_FILE}"
)

print("=" * 60)
print("WALK FORWARD TEST COMPLETE")
print("=" * 60)
