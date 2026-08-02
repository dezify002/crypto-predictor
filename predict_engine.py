import joblib
import pandas as pd
import json

from data.features import FeatureEngineer
from data.live_market import LiveMarket
from config import FEATURE_COLUMNS
from utils.logger import PredictionLogger


def run_prediction(asset, target_price, minutes, direction=1, feature_engineer=None, market=None, logger=None):

    feature_engineer = feature_engineer or FeatureEngineer()
    market = market or LiveMarket()
    logger = logger or PredictionLogger()

    asset = asset.upper().strip()

    if asset not in ["BTC", "ETH", "SOL"]:
        raise ValueError("Asset must be BTC, ETH or SOL.")

    target_price = float(target_price)
    minutes = int(minutes)
    direction = int(direction)

    if minutes <= 0:
        raise ValueError("Minutes must be greater than zero.")

    if direction not in [1, -1]:
        raise ValueError("Direction must be 1 (ABOVE) or -1 (BELOW).")

    # ====================================
    # SELECT MODEL
    # ====================================

    suffix = "" if direction == 1 else "_below"

    if minutes <= 15:
        model_file = f"models/xgb_15m{suffix}.joblib"
        model_name = "15 Minute"
    elif minutes <= 30:
        model_file = f"models/xgb_30m{suffix}.joblib"
        model_name = "30 Minute"
    elif minutes <= 60:
        model_file = f"models/xgb_60m{suffix}.joblib"
        model_name = "60 Minute"
    elif minutes <= 120:
        model_file = f"models/xgb_120m{suffix}.joblib"
        model_name = "120 Minute"
    else:
        model_file = f"models/xgb_240m{suffix}.joblib"
        model_name = "240 Minute"

    model = joblib.load(model_file)

    threshold_file = model_file.replace(".joblib", "_threshold.json")

    try:
        with open(threshold_file, "r") as f:
            threshold_data = json.load(f)
        decision_threshold = threshold_data["threshold"]
        threshold_found = True
    except Exception:
        decision_threshold = 0.50
        threshold_found = False

    # ====================================
    # LOAD MARKET DATA
    # ====================================

    try:
        df, source = market.get_latest(asset)
        data_source = f"LIVE ({source})"
    except Exception:
        file = f"historical_data/{asset}/15m.csv"
        df = pd.read_csv(file)
        data_source = "HISTORICAL (live unavailable)"

    # ====================================
    # FEATURES
    # ====================================

    df = feature_engineer.add_features(df)

    latest = df.iloc[-1].copy()

    current_price = float(latest["close"])

    if direction == 1 and target_price <= current_price:
        raise ValueError(
            f"For an ABOVE prediction, target price must be above the current price "
            f"(current: ${current_price:,.2f})."
        )

    if direction == -1 and target_price >= current_price:
        raise ValueError(
            f"For a BELOW prediction, target price must be below the current price "
            f"(current: ${current_price:,.2f})."
        )

    required_return = abs(target_price - current_price) / current_price

    latest["required_return"] = required_return
    latest["minutes_remaining"] = minutes
    latest["direction"] = direction

    X = pd.DataFrame([latest[FEATURE_COLUMNS]])

    outside_training = required_return > 0.03

    # ====================================
    # PREDICT
    # ====================================

    probability = float(model.predict_proba(X)[0][1])
    prediction_yes = probability >= decision_threshold

    # ====================================
    # VOLATILITY
    # ====================================

    volatility = df["return1"].rolling(20).std().iloc[-1]

    if pd.isna(volatility):
        volatility = 0.0

    if volatility < 0.002:
        volatility_status = "LOW"
    elif volatility < 0.005:
        volatility_status = "MEDIUM"
    else:
        volatility_status = "HIGH"

    # ====================================
    # TREND
    # ====================================

    if latest["ema10"] > latest["ema50"] > latest["ema200"]:
        trend = "STRONG BULLISH"
    elif latest["ema10"] > latest["ema50"]:
        trend = "BULLISH"
    elif latest["ema10"] < latest["ema50"] < latest["ema200"]:
        trend = "STRONG BEARISH"
    elif latest["ema10"] < latest["ema50"]:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    # ====================================
    # RSI
    # ====================================

    rsi = float(latest["rsi"])

    if rsi >= 70:
        rsi_text = "Overbought"
    elif rsi <= 30:
        rsi_text = "Oversold"
    else:
        rsi_text = "Neutral"

    # ====================================
    # CONFIDENCE
    # ====================================

    distance = abs(probability - decision_threshold)

    if distance >= 0.30:
        confidence = "EXTREMELY HIGH"
    elif distance >= 0.20:
        confidence = "VERY HIGH"
    elif distance >= 0.10:
        confidence = "HIGH"
    elif distance >= 0.05:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ====================================
    # RISK
    # ====================================

    risk_score = 0

    if volatility_status == "HIGH":
        risk_score += 2

    if outside_training:
        risk_score += 2

    if abs(required_return) > 0.02:
        risk_score += 1

    if confidence == "LOW":
        risk_score += 2

    if risk_score <= 1:
        risk = "LOW"
    elif risk_score <= 3:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # ====================================
    # VERDICT
    # ====================================

    if prediction_yes:
        margin = probability - decision_threshold
        if margin >= 0.30:
            verdict = "VERY STRONG YES"
        elif margin >= 0.15:
            verdict = "STRONG YES"
        else:
            verdict = "YES"
    else:
        margin = decision_threshold - probability
        if margin >= 0.30:
            verdict = "VERY STRONG NO"
        elif margin >= 0.15:
            verdict = "STRONG NO"
        else:
            verdict = "NO"

    # ====================================
    # LOG
    # ====================================

    logger.log(
        asset=asset,
        current_price=current_price,
        target_price=target_price,
        minutes=minutes,
        model_name=model_name,
        probability=float(probability),
        prediction="YES" if prediction_yes else "NO",
        confidence=confidence,
        volatility=volatility_status,
    )

    return {
        "asset": asset,
        "direction": "ABOVE" if direction == 1 else "BELOW",
        "model_name": model_name,
        "decision_threshold": decision_threshold,
        "threshold_found": threshold_found,
        "data_source": data_source,
        "current_price": current_price,
        "target_price": target_price,
        "required_return": required_return,
        "minutes": minutes,
        "trend": trend,
        "rsi": rsi,
        "rsi_text": rsi_text,
        "volatility": float(volatility),
        "volatility_status": volatility_status,
        "probability": probability,
        "prediction_yes": prediction_yes,
        "verdict": verdict,
        "confidence": confidence,
        "risk": risk,
        "outside_training": outside_training,
    }