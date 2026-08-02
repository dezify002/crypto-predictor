import numpy as np
import pandas as pd
import ta


class FeatureEngineer:

    def add_features(self, df):

        df = df.copy()

        # =========================
        # EMA
        # =========================

        df["ema10"] = ta.trend.ema_indicator(df["close"], window=10)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
        df["ema100"] = ta.trend.ema_indicator(df["close"], window=100)
        df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)

        df["ema10_dist"] = (df["close"] - df["ema10"]) / df["ema10"]
        df["ema20_dist"] = (df["close"] - df["ema20"]) / df["ema20"]
        df["ema50_dist"] = (df["close"] - df["ema50"]) / df["ema50"]
        df["ema100_dist"] = (df["close"] - df["ema100"]) / df["ema100"]
        df["ema200_dist"] = (df["close"] - df["ema200"]) / df["ema200"]

        df["ema10_50"] = (df["ema10"] - df["ema50"]) / df["ema50"]
        df["ema20_200"] = (df["ema20"] - df["ema200"]) / df["ema200"]

        # =========================
        # RSI
        # =========================

        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        df["rsi_change"] = df["rsi"].diff()
        df["rsi_ma"] = df["rsi"].rolling(5).mean()

        # =========================
        # MACD
        # =========================

        df["macd"] = ta.trend.macd(df["close"])
        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        df["macd_hist"] = ta.trend.macd_diff(df["close"])

        df["macd_cross"] = df["macd"] - df["macd_signal"]
        df["macd_hist_change"] = df["macd_hist"].diff()

        # =========================
        # ATR
        # =========================

        df["atr"] = ta.volatility.average_true_range(
            df["high"],
            df["low"],
            df["close"],
        )

        df["atr_percent"] = df["atr"] / df["close"]

        # =========================
        # ADX
        # =========================

        df["adx"] = ta.trend.adx(
            df["high"],
            df["low"],
            df["close"],
        )

        df["adx_pos"] = ta.trend.adx_pos(
            df["high"],
            df["low"],
            df["close"],
        )

        df["adx_neg"] = ta.trend.adx_neg(
            df["high"],
            df["low"],
            df["close"],
        )

        # =========================
        # Bollinger Bands
        # =========================

        bb = ta.volatility.BollingerBands(df["close"])

        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()

        band_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        df["bb_position"] = (
            (df["close"] - df["bb_lower"]) / band_width
        )

        # =========================
        # Returns
        # =========================

        df["return1"] = df["close"].pct_change(1)
        df["return4"] = df["close"].pct_change(4)
        df["return8"] = df["close"].pct_change(8)
        df["return16"] = df["close"].pct_change(16)
        df["return32"] = df["close"].pct_change(32)

        # =========================
        # Candle Structure
        # =========================

        candle_range = (
            df["high"] - df["low"]
        ).replace(0, np.nan)

        df["body"] = (
            df["close"] - df["open"]
        ) / candle_range

        df["upper_wick"] = (
            df["high"]
            - df[["open", "close"]].max(axis=1)
        ) / candle_range

        df["lower_wick"] = (
            df[["open", "close"]].min(axis=1)
            - df["low"]
        ) / candle_range

        # =========================
        # Volume
        # =========================

        df["volume_ma20"] = (
            df["volume"].rolling(20).mean()
        )

        df["relative_volume"] = (
            df["volume"] / df["volume_ma20"]
        )

        volume_std = (
            df["volume"].rolling(20).std()
        ).replace(0, np.nan)

        df["volume_z"] = (
            df["volume"] - df["volume_ma20"]
        ) / volume_std

        # =========================
        # Volatility
        # =========================

        df["volatility20"] = (
            df["return1"].rolling(20).std()
        )

        df["volatility50"] = (
            df["return1"].rolling(50).std()
        )

        # =========================
        # Time
        # =========================

        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"], format="mixed")
        else:
            dt = pd.to_datetime(df["timestamp"], unit="ms")

        df["hour"] = dt.dt.hour
        df["dayofweek"] = dt.dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # =========================
        # Cleanup
        # =========================

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df