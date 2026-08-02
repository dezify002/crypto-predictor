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


# ============================================================
# SETTINGS
# ============================================================

ASSET = "BTC"

TIMEFRAME = 60

DATASET = Path(
    f"datasets/{ASSET.lower()}_questions_{TIMEFRAME}m.parquet"
)

RESULTS_FILE = Path(
    "logs/walk_forward_v2.csv"
)

RESULTS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# First fold uses first 50% for training
INITIAL_TRAIN = 0.50

# 10% validation
VALID_SIZE = 0.10

# 10% testing
TEST_SIZE = 0.10


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("WALK FORWARD TEST V2")
print("=" * 70)

df = pd.read_parquet(DATASET)

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)

print(f"Rows Loaded : {len(df):,}")

results = []

n = len(df)

train_end = int(
    n * INITIAL_TRAIN
)

valid_size = int(
    n * VALID_SIZE
)

test_size = int(
    n * TEST_SIZE
)

fold = 1
# ============================================================
# WALK FORWARD LOOP
# ============================================================

while (

    train_end +
    valid_size +
    test_size

    <= n

):

    print()

    print("=" * 70)

    print(f"FOLD {fold}")

    print("=" * 70)

    # -------------------------
    # SPLITS
    # -------------------------

    train = df.iloc[
        :train_end
    ]

    valid = df.iloc[
        train_end:
        train_end + valid_size
    ]

    test = df.iloc[
        train_end + valid_size:
        train_end + valid_size + test_size
    ]

    print(
        f"Train Rows : {len(train):,}"
    )

    print(
        f"Validation : {len(valid):,}"
    )

    print(
        f"Test Rows  : {len(test):,}"
    )

    # -------------------------
    # FEATURES
    # -------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train["label"]

    X_valid = valid[
        FEATURE_COLUMNS
    ]

    y_valid = valid["label"]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test["label"]

    # -------------------------
    # CLASS BALANCING
    # -------------------------

    negatives = (
        y_train == 0
    ).sum()

    positives = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negatives /
        max(
            positives,
            1,
        )
    )

    print(
        f"scale_pos_weight = {scale_pos_weight:.2f}"
    )

    # -------------------------
    # MODEL
    # -------------------------

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
        "Training..."
    )

    model.fit(

        X_train,

        y_train,

    )

    print(
        "Training Complete."
    )
        # ============================================================
    # THRESHOLD OPTIMIZATION
    # (Validation Set Only)
    # ============================================================

    print(
        "Optimizing Threshold..."
    )

    valid_probs = model.predict_proba(
        X_valid
    )[:, 1]

    best_threshold = 0.50

    best_f1 = -1.0

    threshold_scores = []

    for threshold in np.arange(
        0.20,
        0.96,
        0.01,
    ):

        valid_predictions = (
            valid_probs >= threshold
        ).astype(int)

        score = f1_score(

            y_valid,

            valid_predictions,

            zero_division=0,

        )

        threshold_scores.append(
            (
                threshold,
                score,
            )
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                threshold
            )

    print()

    print(
        f"Best Threshold : {best_threshold:.2f}"
    )

    print(
        f"Validation F1  : {best_f1:.4f}"
    )

    # ============================================================
    # TEST SET
    # (Never used during optimization)
    # ============================================================

    print(
        "Testing..."
    )

    test_probs = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        test_probs >= best_threshold
    ).astype(int)

    confidence = np.maximum(
        test_probs,
        1 - test_probs,
    )

    average_confidence = float(
        confidence.mean()
    )

    print(
        f"Average Confidence : {average_confidence:.2%}"
    )
        # ============================================================
    # METRICS
    # ============================================================

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
        test_probs,
    )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    tp = (
        (predictions == 1) &
        (y_test.values == 1)
    ).sum()

    tn = (
        (predictions == 0) &
        (y_test.values == 0)
    ).sum()

    fp = (
        (predictions == 1) &
        (y_test.values == 0)
    ).sum()

    fn = (
        (predictions == 0) &
        (y_test.values == 1)
    ).sum()

    print()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc:.4f}")

    print()

    print(f"TP : {tp:,}")
    print(f"FP : {fp:,}")
    print(f"TN : {tn:,}")
    print(f"FN : {fn:,}")

    # ============================================================
    # SAVE FOLD
    # ============================================================

    results.append({

        "fold": fold,

        "train_rows": len(train),

        "validation_rows": len(valid),

        "test_rows": len(test),

        "threshold": best_threshold,

        "validation_f1": best_f1,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc,

        "avg_confidence": average_confidence,

        "tp": tp,

        "fp": fp,

        "tn": tn,

        "fn": fn,

    })

    # ============================================================
    # MOVE TO NEXT WINDOW
    # ============================================================

    fold += 1

    train_end += test_size
    # ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(results)

results.to_csv(
    RESULTS_FILE,
    index=False,
)

print()
print("=" * 70)
print("WALK FORWARD SUMMARY")
print("=" * 70)

print(f"Total Folds          : {len(results)}")
print()

print(f"Average Accuracy     : {results['accuracy'].mean():.4f}")
print(f"Average Precision    : {results['precision'].mean():.4f}")
print(f"Average Recall       : {results['recall'].mean():.4f}")
print(f"Average F1 Score     : {results['f1'].mean():.4f}")
print(f"Average ROC AUC      : {results['roc_auc'].mean():.4f}")
print(f"Average Threshold    : {results['threshold'].mean():.2f}")
print(f"Average Confidence   : {results['avg_confidence'].mean():.2%}")

print()

print(f"Accuracy Std Dev     : {results['accuracy'].std():.4f}")
print(f"F1 Std Dev           : {results['f1'].std():.4f}")
print(f"ROC AUC Std Dev      : {results['roc_auc'].std():.4f}")

print()
# ============================================================
# BEST FOLD
# ============================================================

best = results.loc[
    results["f1"].idxmax()
]

print("=" * 70)
print("BEST FOLD")
print("=" * 70)

print(f"Fold                 : {int(best['fold'])}")
print(f"Accuracy             : {best['accuracy']:.4f}")
print(f"Precision            : {best['precision']:.4f}")
print(f"Recall               : {best['recall']:.4f}")
print(f"F1 Score             : {best['f1']:.4f}")
print(f"ROC AUC              : {best['roc_auc']:.4f}")
print(f"Threshold            : {best['threshold']:.2f}")
print(f"Average Confidence   : {best['avg_confidence']:.2%}")

print()

# ============================================================
# WORST FOLD
# ============================================================

worst = results.loc[
    results["f1"].idxmin()
]

print("=" * 70)
print("WORST FOLD")
print("=" * 70)

print(f"Fold                 : {int(worst['fold'])}")
print(f"Accuracy             : {worst['accuracy']:.4f}")
print(f"Precision            : {worst['precision']:.4f}")
print(f"Recall               : {worst['recall']:.4f}")
print(f"F1 Score             : {worst['f1']:.4f}")
print(f"ROC AUC              : {worst['roc_auc']:.4f}")

print()

# ============================================================
# MODEL RATING
# ============================================================

avg_auc = results["roc_auc"].mean()
avg_f1 = results["f1"].mean()

if avg_auc >= 0.95 and avg_f1 >= 0.70:

    rating = "OUTSTANDING"

elif avg_auc >= 0.90 and avg_f1 >= 0.65:

    rating = "EXCELLENT"

elif avg_auc >= 0.85:

    rating = "VERY GOOD"

elif avg_auc >= 0.80:

    rating = "GOOD"

else:

    rating = "NEEDS IMPROVEMENT"

print("=" * 70)
print("FINAL MODEL RATING")
print("=" * 70)

print(f"Overall Rating       : {rating}")

print()

print(f"Results Saved        : {RESULTS_FILE}")

print()

print("=" * 70)
print("WALK FORWARD TEST V2 COMPLETE")
print("=" * 70)