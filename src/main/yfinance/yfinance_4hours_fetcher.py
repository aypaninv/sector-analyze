"""
Parallel YFinance -> CSV pipeline (ADX removed)

Features:
- Reads symbols from files/stocks.csv (expects a "Symbol" column; optional "Folio", "Sector", "Market Cap")
- Appends ".NS" for NSE tickers if missing
- Fetches 6 months of daily OHLCV data from yfinance in parallel (ThreadPool)
- Retries failed fetches with exponential backoff
- Normalizes columns, formats Date as YYYY-MM-DD, rounds prices to 2 decimals, volume as Int64
- Saves output to output/yfinance_daily_output.csv
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Optional, List

# --- Configuration ---
DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "4h"
NSE_SUFFIX = ".NS"
MAX_WORKERS = 4
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5
OUTPUT_PATH = "output/yfinance_4hours_output.csv"


@dataclass
class CSVReader:
    csv_path: str = "files/stocks.csv"

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Symbol" not in df.columns:
            raise ValueError("CSV must contain a 'Symbol' column")
        return df


class YFinanceFetcher:
    def __init__(self, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL, nse_suffix: str = NSE_SUFFIX):
        self.period = period
        self.interval = interval
        self.nse_suffix = nse_suffix

    def _symbol_for_fetch(self, symbol: str) -> str:
        s = symbol.strip()
        return s if s.endswith(self.nse_suffix) else s + self.nse_suffix

    def _try_fetch(self, fetch_sym: str) -> Optional[pd.DataFrame]:
        t = yf.Ticker(fetch_sym)
        return t.history(period=self.period, interval=self.interval, auto_adjust=False)

    def fetch_with_retries(
        self,
        symbol: str,
        attempts: int = RETRY_ATTEMPTS,
        backoff_base: float = RETRY_BACKOFF_BASE
    ) -> Optional[pd.DataFrame]:

        fetch_sym = self._symbol_for_fetch(symbol)
        for attempt in range(1, attempts + 1):
            try:
                df = self._try_fetch(fetch_sym)
                if df is None or df.empty:
                    raise RuntimeError(f"empty data for {fetch_sym}")

                df = df.copy()
                try:
                    df.index = pd.to_datetime(df.index).normalize()
                except Exception:
                    pass

                df = df.reset_index()

                date_col = None
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c
                        break

                if date_col and date_col != "date":
                    df = df.rename(columns={date_col: "date"})
                elif "index" in df.columns and "date" not in df.columns:
                    df = df.rename(columns={"index": "date"})

                df.columns = [str(c).strip().lower() for c in df.columns]
                return df

            except Exception:
                if attempt < attempts:
                    delay = (backoff_base ** attempt) + (0.1 * attempt)
                    time.sleep(delay)
                else:
                    return None
        return None


def _process_symbol_result(symbol: str, hist: Optional[pd.DataFrame], meta_row: pd.Series) -> Optional[pd.DataFrame]:
    if hist is None or hist.empty:
        return None

    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(hist.columns)):
        return None

    hist = hist[["date", "open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )

    hist[["open", "high", "low", "close"]] = (
        hist[["open", "high", "low", "close"]]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )

    hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce")

    for c in ("open", "high", "low", "close"):
        hist[c] = hist[c].round(2)

    hist["volume"] = hist["volume"].round(0).astype("Int64")
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date.astype(str)

    # ---- Attach metadata (NEW: Folio added) ----
    hist["symbol"] = str(meta_row["Symbol"]).strip()
    hist["folio"] = meta_row.get("Folio", "")
    hist["sector"] = meta_row.get("Sector", "")
    hist["marketcap"] = meta_row.get("Market Cap", meta_row.get("MarketCap", ""))

    cols = [
        "symbol",
        "folio",
        "sector",
        "marketcap",
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
    ]

    hist = hist.reindex(columns=cols)
    return hist


def build_historical_dataframe_parallel(
    csv_path: str = "files/stocks.csv",
    max_workers: int = MAX_WORKERS
) -> pd.DataFrame:

    reader = CSVReader(csv_path=csv_path)
    base_df = reader.read()
    fetcher = YFinanceFetcher()

    results: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_meta = {}

        for _, row in base_df.iterrows():
            symbol = str(row["Symbol"]).strip()
            fut = ex.submit(fetcher.fetch_with_retries, symbol)
            future_to_meta[fut] = (symbol, row)

        for fut in as_completed(future_to_meta):
            symbol, meta_row = future_to_meta[fut]
            try:
                hist = fut.result(timeout=60)
            except Exception:
                hist = None

            processed = _process_symbol_result(symbol, hist, meta_row)
            if processed is not None and not processed.empty:
                results.append(processed)

    if not results:
        cols = ["Symbol", "Folio", "Sector", "MarketCap", "Date", "Open", "Close", "High", "Low", "Volume"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(results, ignore_index=True)

    out = out.rename(columns={
        "symbol": "Symbol",
        "folio": "Folio",
        "sector": "Sector",
        "marketcap": "MarketCap",
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    })

    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import os

    os.makedirs("output", exist_ok=True)
    df_out = build_historical_dataframe_parallel("files/stocks.csv", max_workers=MAX_WORKERS)

    if not df_out.empty:
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH} ({len(df_out)} rows)")
    else:
        print("No historical data produced.")
