"""
Downloads multi-year 15m candle history from Binance's public data archive
and merges it into historical_data/<ASSET>/15m.csv, matching the schema
LiveMarket.get_latest() already produces (timestamp, open, high, low, close,
volume, datetime).

Run this locally (data.binance.vision is not reachable from Claude's sandbox).

    pip install requests pandas
    python download_historical_data.py
"""

import io
import os
import time
import zipfile
from datetime import datetime

import pandas as pd
import requests

# ============================================================
# CONFIG
# ============================================================

ASSETS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

INTERVAL = "15m"

START_YEAR = 2022
START_MONTH = 1

OUTPUT_DIR = "historical_data"
CACHE_DIR = "binance_archive_cache"

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Older archive files ship with no header row in this column order.
# Newer files (mid-2024 onward) include a header row instead - handled below.
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

REQUEST_TIMEOUT = 30
RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.3


def month_range(start_year, start_month):
    """Yield (year, month) from start up to, but excluding, the current month."""

    now = datetime.utcnow()

    current = datetime(start_year, start_month, 1)
    end = datetime(now.year, now.month, 1)

    while current < end:

        yield current.year, current.month

        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def download_month(symbol, year, month):
    """Download one monthly zip (cached locally). Returns raw CSV bytes or None."""

    filename = f"{symbol}-{INTERVAL}-{year}-{month:02d}"

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{filename}.zip")

    if os.path.exists(cache_path):

        with open(cache_path, "rb") as f:
            zip_bytes = f.read()

    else:

        url = f"{BASE_URL}/{symbol}/{INTERVAL}/{filename}.zip"

        zip_bytes = None

        for attempt in range(1, RETRIES + 1):

            try:

                resp = requests.get(url, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 404:
                    print(f"  [skip] {filename} not available (404)")
                    return None

                resp.raise_for_status()

                zip_bytes = resp.content
                break

            except Exception as e:

                print(f"  [retry {attempt}/{RETRIES}] {filename}: {e}")
                time.sleep(1.5 * attempt)

        if zip_bytes is None:
            print(f"  [FAILED] {filename} after {RETRIES} attempts, skipping")
            return None

        with open(cache_path, "wb") as f:
            f.write(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:

        inner_name = zf.namelist()[0]

        with zf.open(inner_name) as f:
            return f.read()


def parse_csv_bytes(csv_bytes):
    """Handles both the old (no header) and newer (header row) archive formats."""

    first_line = csv_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    first_field = first_line.strip().split(",")[0]

    has_header = not first_field.lstrip("-").isdigit()

    buf = io.BytesIO(csv_bytes)

    if has_header:

        df = pd.read_csv(buf)
        df.columns = [c.strip().lower() for c in df.columns]

        df = df.rename(columns={
            "opentime": "open_time",
            "closetime": "close_time",
        })

    else:

        df = pd.read_csv(buf, header=None, names=RAW_COLUMNS)

    return df[["open_time", "open", "high", "low", "close", "volume"]]


def normalize_timestamp_unit(series):
    """
    Binance's monthly archive has used both millisecond and microsecond
    open_time values depending on when the file was generated. Detect
    which one we've got by magnitude and normalize everything to
    milliseconds, regardless of file format/header era.
    """

    series = series.astype("int64")

    magnitude = int(series.abs().max())

    if magnitude > 10 ** 16:
        # nanoseconds -> milliseconds
        series = series // 10 ** 6
    elif magnitude > 10 ** 13:
        # microseconds -> milliseconds
        series = series // 10 ** 3

    return series


def build_asset_csv(asset, symbol):

    print(f"\n=== {asset} ({symbol}) ===")

    frames = []

    for year, month in month_range(START_YEAR, START_MONTH):

        print(f"Downloading {symbol} {year}-{month:02d}...")

        raw_bytes = download_month(symbol, year, month)

        if raw_bytes is None:
            continue

        df = parse_csv_bytes(raw_bytes)
        frames.append(df)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not frames:
        print(f"No data downloaded for {asset}, skipping.")
        return

    full = pd.concat(frames, ignore_index=True)

    full["open_time"] = normalize_timestamp_unit(full["open_time"])

    full = full.drop_duplicates(subset="open_time")
    full = full.sort_values("open_time").reset_index(drop=True)

    full = full.rename(columns={"open_time": "timestamp"})

    full["datetime"] = pd.to_datetime(
        full["timestamp"], unit="ms"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    for col in ["open", "high", "low", "close", "volume"]:
        full[col] = full[col].astype(float)

    out_dir = os.path.join(OUTPUT_DIR, asset)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "15m.csv")
    full.to_csv(out_path, index=False)

    print(f"Saved {len(full):,} candles -> {out_path}")
    print(f"Range: {full['datetime'].iloc[0]} to {full['datetime'].iloc[-1]}")


def main():

    print("=" * 60)
    print("BINANCE HISTORICAL DATA DOWNLOADER")
    print("=" * 60)
    print(f"Interval : {INTERVAL}")
    print(f"From     : {START_YEAR}-{START_MONTH:02d}")
    print(f"Assets   : {', '.join(ASSETS.keys())}")

    for asset, symbol in ASSETS.items():
        build_asset_csv(asset, symbol)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("Next steps:")
    print("  python build_dataset.py")
    print("  python train.py")


if __name__ == "__main__":
    main()