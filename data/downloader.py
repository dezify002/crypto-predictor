import time
from pathlib import Path

import ccxt
import pandas as pd


class BitgetDownloader:
    def __init__(self):
        self.exchange = ccxt.bitget({
            "enableRateLimit": True,
        })

        # Project root (crypto_predictor/)
        self.project_root = Path(__file__).resolve().parent.parent

    def download(
        self,
        symbol,
        timeframe="15m",
        limit=200,
        max_batches=100,
    ):
        print(f"\nDownloading {symbol} ({timeframe})...")

        all_candles = []

        since = self.exchange.parse8601("2024-01-01T00:00:00Z")

        for _ in range(max_batches):

            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit,
            )

            if not candles:
                break

            all_candles.extend(candles)

            since = candles[-1][0] + 1

            print(f"Downloaded {len(all_candles)} candles", end="\r")

            time.sleep(self.exchange.rateLimit / 1000)

        df = pd.DataFrame(
            all_candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        asset = symbol.split("/")[0]

        # Save to crypto_predictor/historical_data/
        folder = self.project_root / "historical_data" / asset
        folder.mkdir(parents=True, exist_ok=True)

        filename = folder / f"{timeframe}.csv"

        df.to_csv(filename, index=False)

        print(f"\nSaved {len(df)} rows to {filename}")


if __name__ == "__main__":

    downloader = BitgetDownloader()

    downloader.download("BTC/USDT")
    downloader.download("ETH/USDT")
    downloader.download("SOL/USDT")