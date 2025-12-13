"""
Fetch 10 years of MONTHLY close prices for a list of stocks.

- Input:  stocks.csv  (must contain at least a "Symbol" column; optional "Sector" and "MarketCap")
- Output: output/yf_monthly_close_10y.csv

Columns in output:
    Symbol, Sector, MarketCap, Date, Close

Requirements:
    pip install pandas yfinance numpy
"""

from dataclasses import dataclass
from typing import Optional, List

import os
import time
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Configuration ---
DEFAULT_PERIOD = "5y"    # how far back to fetch
DEFAULT_INTERVAL = "1mo"  # monthly candles
NSE_SUFFIX = ".NS"        # change/remove if not using NSE symbols
MAX_WORKERS = 4           # number of parallel threads for fetching
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5
INPUT_CSV_PATH = "files/stocks.csv"
OUTPUT_PATH = "output/stocks_monthly.csv"


# --- CSV Reader --------------------------------------------------------------

@dataclass
class CSVReader:
    csv_path: str = INPUT_CSV_PATH

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = [c.strip() for c in df.columns]

        if "Symbol" not in df.columns:
            raise ValueError("Input CSV must contain a 'Symbol' column")

        return df


# --- YFinance Fetcher --------------------------------------------------------

class YFinanceFetcher:
    def __init__(self,
                 period: str = DEFAULT_PERIOD,
                 interval: str = DEFAULT_INTERVAL,
                 nse_suffix: str = NSE_SUFFIX):
        self.period = period
        self.interval = interval
        self.nse_suffix = nse_suffix

    def _symbol_for_fetch(self, symbol: str) -> str:
        s = symbol.strip()
        # Append ".NS" if not already present (for NSE tickers)
        return s if (not self.nse_suffix or s.endswith(self.nse_suffix)) else s + self.nse_suffix

    def _try_fetch(self, fetch_sym: str) -> Optional[pd.DataFrame]:
        print(fetch_sym)
        ticker = yf.Ticker(fetch_sym)
        df = ticker.history(period=self.period,
                            interval=self.interval,
                            auto_adjust=False)
        return df

    def fetch_with_retries(
        self,
        symbol: str,
        attempts: int = RETRY_ATTEMPTS,
        backoff_base: float = RETRY_BACKOFF_BASE,
    ) -> Optional[pd.DataFrame]:
        fetch_sym = self._symbol_for_fetch(symbol)
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                df = self._try_fetch(fetch_sym)
                if df is None or df.empty:
                    last_exc = RuntimeError(f"Empty data for {fetch_sym}")
                    raise last_exc

                df = df.copy()

                # Index is usually DatetimeIndex -> normalize & move to column
                try:
                    df.index = pd.to_datetime(df.index).normalize()
                except Exception:
                    pass

                df = df.reset_index()

                # Find the date column
                date_col = None
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c
                        break

                if date_col and date_col != "date":
                    df = df.rename(columns={date_col: "date"})
                elif "index" in df.columns and "date" not in df.columns:
                    df = df.rename(columns={"index": "date"})

                # Normalize column names to lowercase
                df.columns = [str(c).strip().lower() for c in df.columns]

                return df

            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    delay = (backoff_base ** attempt) + (0.1 * attempt)
                    time.sleep(delay)
                else:
                    print(f"[WARN] Failed to fetch {fetch_sym}: {last_exc}")
                    return None

        return None


# --- Processing --------------------------------------------------------------

def _process_symbol_result(
    symbol: str,
    hist: Optional[pd.DataFrame],
    meta_row: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    Processes yfinance monthly data and extracts ONLY:
      - symbol
      - sector
      - marketcap
      - date
      - close (monthly close)
    """
    if hist is None or hist.empty:
        return None

    required = {"date", "close"}
    if not required.issubset(set(hist.columns)):
        # Missing required columns, skip
        return None

    # Keep only date & close; drop rows where close is NaN
    hist = hist[["date", "close"]].dropna(subset=["close"])

    # Convert types
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce").astype(float).round(2)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date.astype(str)

    # Attach metadata from the input CSV
    hist["symbol"] = str(meta_row.get("Symbol", "")).strip()
    hist["sector"] = meta_row.get("Sector", "")
    hist["marketcap"] = meta_row.get("MarketCap", "")

    # Final column order (lowercase here; we'll rename at the end)
    cols = ["symbol", "sector", "marketcap", "date", "close"]
    hist = hist.reindex(columns=cols)

    return hist


def build_historical_dataframe_parallel(
    csv_path: str = INPUT_CSV_PATH,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    reader = CSVReader(csv_path=csv_path)
    base_df = reader.read()

    fetcher = YFinanceFetcher(
        period=DEFAULT_PERIOD,
        interval=DEFAULT_INTERVAL,
        nse_suffix=NSE_SUFFIX,
    )

    results: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {}

        # Schedule all fetches
        for _, row in base_df.iterrows():
            symbol = str(row["Symbol"]).strip()
            fut = executor.submit(fetcher.fetch_with_retries, symbol)
            future_to_meta[fut] = (symbol, row)

        # Collect results
        for fut in as_completed(future_to_meta):
            symbol, meta_row = future_to_meta[fut]
            try:
                hist = fut.result(timeout=60)
            except Exception as exc:
                print(f"[WARN] Exception fetching {symbol}: {exc}")
                hist = None

            processed = _process_symbol_result(symbol, hist, meta_row)
            if processed is not None and not processed.empty:
                results.append(processed)

    if not results:
        # Empty result, but return proper columns
        cols = ["Symbol", "Sector", "MarketCap", "Date", "Close"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(results, ignore_index=True)

    # Rename to final column names
    out = out.rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
            "marketcap": "MarketCap",
            "date": "Date",
            "close": "Close",
        }
    )

    # Sort for nicer output
    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    return out


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    df_out = build_historical_dataframe_parallel(
        csv_path=INPUT_CSV_PATH,
        max_workers=MAX_WORKERS,
    )

    if not df_out.empty:
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH} ({len(df_out)} rows)")
    else:
        print("No historical data produced.")
