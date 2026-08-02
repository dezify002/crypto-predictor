import ccxt
import pandas as pd
import traceback


class LiveMarket:

    def __init__(self):

        self.exchange = ccxt.bitget({
            "enableRateLimit": True,
            "timeout": 30000,
        })

    def get_latest(self, asset):

        symbol = f"{asset}/USDT"

        try:

            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe="15m",
                limit=250,
            )

            df = pd.DataFrame(
                candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )

            df["datetime"] = pd.to_datetime(
                df["timestamp"],
                unit="ms",
            )

            return df, "Bitget"

        except Exception as e:

            print("FULL ERROR TRACEBACK:")
            traceback.print_exc()

            raise RuntimeError(
                f"Bitget fetch failed: {repr(e)}"
            )