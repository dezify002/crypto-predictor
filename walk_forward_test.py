import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

from config import (
    ASSETS,
    HORIZONS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
)


# ==========================================================
# SETTINGS
# ==========================================================

N_FOLDS = 5          # number of expanding walk-forward steps
N_ESTIMATORS = 150   # lower than trainer.py's 300 for reasonable runtime
FIXED_THRESHOLD = 0.70  # simple fixed threshold for comparability across folds/horizons


def load_full_dataset(horizon):

    frames = []

    for asset in ASSETS:

        path = DATASET_PATH / f"{asset.lower()}_questions_{horizon}m.parquet"

        frames.append(pd.read_parquet(path))

    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def run_fold(train_df, test_df):

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    scale_pos_weight = negatives / max(positives, 1)

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
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

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= FIXED_THRESHOLD).astype(int)

    base_rate = y_test.mean()

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    n_signals = int(preds.sum())

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "base_rate": base_rate,
        "signals": n_signals,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "edge": precision - base_rate,
    }


def walk_forward_for_horizon(horizon):

    print()
    print("=" * 60)
    print(f"WALK-FORWARD TEST — {horizon} MIN MODEL")
    print("=" * 60)

    t0 = time.time()

    df = load_full_dataset(horizon)

    print(f"Total rows: {len(df):,}  (loaded in {time.time()-t0:.1f}s)")

    n = len(df)

    # Reserve the earliest chunk purely as the initial training base,
    # then split the remainder into N_FOLDS equal test chunks.
    initial_train_end = int(n * 0.40)

    remaining = n - initial_train_end
    fold_size = remaining // N_FOLDS

    results = []

    train_end = initial_train_end

    for fold in range(N_FOLDS):

        test_start = train_end
        test_end = test_start + fold_size

        if fold == N_FOLDS - 1:
            test_end = n  # last fold takes any remainder

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        print()
        print(f"--- Fold {fold + 1}/{N_FOLDS} ---")
        print(f"Train: rows 0 to {train_end:,}")
        print(f"Test : rows {test_start:,} to {test_end:,}")

        t1 = time.time()

        result = run_fold(train_df, test_df)

        print(f"Base Rate : {result['base_rate']:.2%}")
        print(f"Signals   : {result['signals']:,}")
        print(f"Accuracy  : {result['accuracy']:.2%}")
        print(f"Precision : {result['precision']:.2%}")
        print(f"Recall    : {result['recall']:.2%}")
        print(f"Edge      : {result['edge']:+.2%}")
        print(f"(fold took {time.time()-t1:.1f}s)")

        results.append(result)

        train_end = test_end  # expand training window for next fold

    print()
    print(f"--- {horizon}m SUMMARY (avg across {N_FOLDS} folds) ---")

    avg_precision = np.mean([r["precision"] for r in results])
    avg_recall = np.mean([r["recall"] for r in results])
    avg_edge = np.mean([r["edge"] for r in results])
    avg_base = np.mean([r["base_rate"] for r in results])

    print(f"Avg Base Rate : {avg_base:.2%}")
    print(f"Avg Precision : {avg_precision:.2%}")
    print(f"Avg Recall    : {avg_recall:.2%}")
    print(f"Avg Edge      : {avg_edge:+.2%}")

    return {
        "horizon": horizon,
        "avg_base_rate": avg_base,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_edge": avg_edge,
    }


def main():

    print()
    print("=" * 60)
    print("WALK-FORWARD TEST — ALL HORIZONS")
    print("=" * 60)
    print(f"Folds per horizon : {N_FOLDS}")
    print(f"Fixed threshold   : {FIXED_THRESHOLD}")
    print(f"Estimators/fold   : {N_ESTIMATORS}")

    overall_start = time.time()

    summaries = []

    for horizon in HORIZONS:

        summary = walk_forward_for_horizon(horizon)
        summaries.append(summary)

    print()
    print("=" * 60)
    print("FINAL COMPARISON ACROSS HORIZONS")
    print("=" * 60)
    print(f"{'Horizon':>8} | {'Base Rate':>10} | {'Precision':>10} | {'Recall':>8} | {'Edge':>8}")
    print("-" * 55)

    for s in summaries:
        print(
            f"{s['horizon']:>6}m | {s['avg_base_rate']:>9.2%} | "
            f"{s['avg_precision']:>9.2%} | {s['avg_recall']:>7.2%} | {s['avg_edge']:>+7.2%}"
        )

    print()
    print(f"Total runtime: {(time.time()-overall_start)/60:.1f} minutes")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()