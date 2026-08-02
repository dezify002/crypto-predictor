import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from predict_engine import run_prediction
from data.features import FeatureEngineer
from data.live_market import LiveMarket
from utils.logger import PredictionLogger


class Dashboard:

    def __init__(self):

        self.logs = Path("logs")
        self.models = Path("models")

        self.feature_engineer = FeatureEngineer()
        self.market = LiveMarket()
        self.logger = PredictionLogger()

    # =====================================================
    # Safe CSV Loader
    # =====================================================

    def load_csv(self, file):

        path = self.logs / file

        if path.exists():
            try:
                return pd.read_csv(path)
            except:
                return pd.DataFrame()

        return pd.DataFrame()

    # =====================================================
    # Safe JSON Loader
    # =====================================================

    def load_json(self, file):

        path = self.models / file

        if path.exists():
            with open(path) as f:
                return json.load(f)

        return {}

    # =====================================================
    # Dashboard Data
    # =====================================================

    def load(self):

        walk = self.load_csv("walk_forward_v2.csv")
        confidence = self.load_csv("confidence_analysis.csv")
        pending = self.load_csv("pending_predictions.csv")

        if len(walk):
            accuracy = float(walk["accuracy"].mean())
            precision = float(walk["precision"].mean())
            recall = float(walk["recall"].mean())
            f1 = float(walk["f1"].mean())
            roc = float(walk["roc_auc"].mean())
            folds = int(len(walk))
        else:
            accuracy = precision = recall = f1 = roc = folds = 0

        if len(confidence):
            calibration = "GOOD"
            avg_confidence = float(confidence["AvgConfidence"].mean())
        else:
            calibration = "UNKNOWN"
            avg_confidence = 0

        total_predictions = len(pending)

        recent_predictions = []

        if total_predictions:
            recent_predictions = (
                pending.tail(10).fillna("").to_dict(orient="records")
            )

        thresholds = {}

        for horizon in [15, 30, 60, 120, 240]:
            data = self.load_json(f"xgb_{horizon}m_threshold.json")
            thresholds[f"{horizon}m"] = data.get("threshold", None)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc,
            "folds": folds,
            "avg_confidence": avg_confidence,
            "calibration": calibration,
            "total_predictions": total_predictions,
            "recent_predictions": recent_predictions,
            "thresholds": thresholds,
        }

    # =====================================================
    # Live Prediction — reuses predict_engine.py, no duplicate logic
    # =====================================================

    def predict(self, asset, target_price, minutes):

        return run_prediction(
            asset,
            target_price,
            minutes,
            feature_engineer=self.feature_engineer,
            market=self.market,
            logger=self.logger,
        )