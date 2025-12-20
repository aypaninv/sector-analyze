"""
Parallel YFinance -> CSV pipeline

Supports:
- 4h, 1d, 1wk intervals
- Proper datetime/date normalization
- CLI arguments
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
DEFAULT_PERIOD = "3mo"
DEFAULT_INTERVAL = "1d"

NSE_SUFFIX = ".NS"
MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5


# ---------------- CSV Reader ----------------
@dataclass
class CSVReader:
    csv_path: str

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]

        if "Symbol" not in df.columns:
            raise ValueError("Input CSV must contain a 'Symbol' column")

        return df


# ---------------- YFinance Fetcher ----------------
class YFinanceFetcher:
    def __init__(self, period: str, interval: str):
        self.period = period
        self.interval = interval

    def _symbol_for_fetch(self, symbol: str) -> str:
        symbol = symbol.strip()
        return symbol if symbol.endswith(NSE_SUFFIX) else symbol + NSE_SUFFIX

    def fetch_with_retries(self, symbol: str) -> Optional[pd.DataFrame]:
        fetch_symbol = self._symbol_for_fetch(symbol)

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                df = yf.Ticker(fetch_symbol).history(
                    period=self.period,
                    interval=self.interval,
                    auto_adjust=False
                )

                if df is None or df.empty:
                    raise RuntimeError("Empty data returned")

                # ---- Normalize index column ----
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]

                if "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})
                elif "date" not in df.columns:
                    raise RuntimeError("No date/datetime column found")

                df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

                return df

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    time.sleep((RETRY_BACKOFF_BASE ** attempt) + 0.1)
                else:
                    print(f"[FAIL] {fetch_symbol}: {e}")
                    return None

        return None


# ---------------- Processing ----------------
def _process_symbol_result(
    hist: Optional[pd.DataFrame],
    meta_row: pd.Series
) -> Optional[pd.DataFrame]:

    if hist is None or hist.empty:
        return None

    required_cols = {"date", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(hist.columns):
        print(f"[SKIP] Missing columns for {meta_row['Symbol']}: {hist.columns}")
        return None

    hist = hist[list(required_cols)].dropna(
        subset=["open", "high", "low", "close"]
    )

    hist[["open", "high", "low", "close"]] = (
        hist[["open", "high", "low", "close"]]
        .astype(float)
        .round(2)
    )

    hist["volume"] = pd.to_numeric(
        hist["volume"], errors="coerce"
    ).astype("Int64")

    # ---- Attach metadata ----
    hist["Symbol"] = meta_row["Symbol"]
    hist["Sector"] = meta_row.get("Sector", "")
    hist["MarketCap"] = meta_row.get(
        "Market Cap",
        meta_row.get("MarketCap", "")
    )

    return hist[
        ["Symbol", "Sector", "MarketCap",
         "date", "open", "close", "high", "low", "volume"]
    ]


# ---------------- Parallel Builder ----------------
def build_historical_dataframe_parallel(
    csv_path: str,
    period: str,
    interval: str,
    max_workers: int = MAX_WORKERS
) -> pd.DataFrame:

    reader = CSVReader(csv_path)
    base_df = reader.read()
    fetcher = YFinanceFetcher(period=period, interval=interval)

    results: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetcher.fetch_with_retries, row["Symbol"]): row
            for _, row in base_df.iterrows()
        }

        for future in as_completed(futures):
            meta_row = futures[future]
            hist = future.result()

            processed = _process_symbol_result(hist, meta_row)
            if processed is not None and not processed.empty:
                results.append(processed)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    out = out.rename(columns={
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    })

    return out.sort_values(["Symbol", "Date"]).reset_index(drop=True)


# ---------------- Main (CLI) ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from Yahoo Finance (4h / 1d / 1wk)"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Input CSV file"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV file"
    )

    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=["4h", "1h", "1d", "1wk"],
        help="YFinance interval"
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="YFinance period (e.g. 1mo, 60d, 6mo)"
    )

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
