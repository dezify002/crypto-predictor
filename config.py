from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "historical_data"
DATASET_PATH = ROOT / "datasets"
MODEL_PATH = ROOT / "models"

DATASET_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)

# =====================================================
# ASSETS
# =====================================================

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
]

EXCHANGE_SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
}

# =====================================================
# EXCHANGE
# =====================================================

DEFAULT_EXCHANGE = "bitget"

# =====================================================
# DATA
# =====================================================

TIMEFRAME = "15m"

HORIZONS = [
    15,
    30,
    60,
    120,
    240,
]

TARGET_RETURNS = [
    0.0025,
    0.0050,
    0.0075,
    0.0100,
    0.0150,
    0.0200,
    0.0300,
]

# =====================================================
# DATASET SPLITS
# =====================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# =====================================================
# RANDOMNESS
# =====================================================

RANDOM_STATE = 42

# =====================================================
# FEATURES USED FOR TRAINING
# =====================================================

FEATURE_COLUMNS = [

    # EMA
    "ema10",
    "ema20",
    "ema50",
    "ema100",
    "ema200",

    "ema10_dist",
    "ema20_dist",
    "ema50_dist",
    "ema100_dist",
    "ema200_dist",

    "ema10_50",
    "ema20_200",

    # RSI
    "rsi",
    "rsi_change",
    "rsi_ma",

    # MACD
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_cross",
    "macd_hist_change",

    # ATR
    "atr",
    "atr_percent",

    # ADX
    "adx",
    "adx_pos",
    "adx_neg",

    # Bollinger
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",

    # Returns
    "return1",
    "return4",
    "return8",
    "return16",
    "return32",

    # Candle structure
    "body",
    "upper_wick",
    "lower_wick",

    # Volume
    "volume_ma20",
    "relative_volume",
    "volume_z",

    # Volatility
    "volatility20",
    "volatility50",

    # Time
    "hour",
    "dayofweek",
    "is_weekend",

    # Question features
    "required_return",
    "minutes_remaining",
    "direction",
]