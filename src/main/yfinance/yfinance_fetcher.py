"""
Parallel YFinance -> CSV pipeline

Supports:
- 4h, 1d, 1wk, 1mo intervals
- Weekly = Friday close
- Monthly = last working day close
- Proper datetime/date normalization
"""

from dataclasses import dataclass
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Optional, List
import os
import argparse

# ---------------- Defaults ----------------
DEFAULT_INPUT_FILE = "files/stocks.csv"
DEFAULT_OUTPUT_PATH = "output/yfinance_output.csv"
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

NSE_SUFFIX = ".NS"
MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5


@dataclass
class CSVReader:
    csv_path: str

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]

        if "Symbol" not in df.columns:
            raise ValueError("Input CSV must contain a 'Symbol' column")

        return df


class YFinanceFetcher:
    def __init__(self, period: str, interval: str):
        self.period = period
        self.interval = interval

    def _symbol_for_fetch(self, symbol: str) -> str:
        return symbol.strip() + NSE_SUFFIX if not symbol.endswith(NSE_SUFFIX) else symbol

    def _resample_weekly_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily data to:
        - Weekly (Friday close)
        - Monthly (Business month end)
        """
        df = df.set_index("date")

        if self.interval == "1wk":
            rule = "W-FRI"
        elif self.interval == "1mo":
            rule = "BME"
        else:
            return df.reset_index()

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        out = df.resample(rule).agg(agg).dropna()
        return out.reset_index()

    def fetch_with_retries(self, symbol: str) -> Optional[pd.DataFrame]:
        fetch_symbol = self._symbol_for_fetch(symbol)

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                # 🔑 Weekly/Monthly must be derived from DAILY
                yf_interval = "1d" if self.interval in {"1wk", "1mo"} else self.interval

                df = yf.Ticker(fetch_symbol).history(
                    period=self.period,
                    interval=yf_interval,
                    auto_adjust=False
                )

                if df is None or df.empty:
                    raise RuntimeError("Empty data returned")

                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})
                elif "date" not in df.columns:
                    raise RuntimeError("No date/datetime column found")

                df["date"] = pd.to_datetime(df["date"])

                # 🔑 Resample if weekly/monthly
                df = self._resample_weekly_monthly(df)

                df["date"] = df["date"].dt.date.astype(str)
                return df

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    time.sleep((RETRY_BACKOFF_BASE ** attempt) + 0.1)
                else:
                    print(f"[FAIL] {fetch_symbol}: {e}")
                    return None

        return None


def _process_symbol_result(hist: Optional[pd.DataFrame], meta_row: pd.Series):
    if hist is None or hist.empty:
        return None

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(hist.columns):
        return None

    hist["Symbol"] = meta_row["Symbol"]

    hist[["open", "high", "low", "close"]] = (
        hist[["open", "high", "low", "close"]].astype(float).round(2)
    )
    hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce").astype("Int64")

    return hist[
        ["Symbol", "date", "open", "close", "high", "low", "volume"]
    ]


def build_historical_dataframe_parallel(csv_path, period, interval):
    reader = CSVReader(csv_path)
    base_df = reader.read()
    fetcher = YFinanceFetcher(period, interval)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetcher.fetch_with_retries, r["Symbol"]): r
            for _, r in base_df.iterrows()
        }

        for f in as_completed(futures):
            row = futures[f]
            hist = f.result()
            out = _process_symbol_result(hist, row)
            if out is not None:
                results.append(out)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    })

    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data (4H / Daily / Weekly / Monthly with proper closes)"
    )

    parser.add_argument("--input", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL,
                        choices=["4h", "1d", "1wk", "1mo"])
    parser.add_argument("--period", default=DEFAULT_PERIOD)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df_out = build_historical_dataframe_parallel(
        csv_path=args.input,
        period=args.period,
        interval=args.interval
    )

    if not df_out.empty:
        df_out.to_csv(args.output, index=False)
        print(f"[OK] Saved to {args.output} ({len(df_out)} rows)")
    else:
        print("[WARN] No historical data produced.")
