"""
ADX + DI+ pipeline with parallel fetching

Features:
- Reads symbols from files/stocks.csv
- Appends ".NS" for NSE tickers
- Fetches 6 months of daily OHLCV data from yfinance in parallel (ThreadPool)
- Retries failed fetches with exponential backoff
- Calculates ADX and DI+ (period=10)
- Keeps Date as YYYY-MM-DD, prices & volume as integers, DI+ and ADX with 2 decimals
- Saves output to files/stocks_with_adx_historical.csv

Usage:
    pip install pandas yfinance numpy
    python adx_pipeline_parallel.py
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Optional, List

# --- Configuration ---
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"
NSE_SUFFIX = ".NS"
MAX_WORKERS = 4         # tune as needed
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.5


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
        df = t.history(period=self.period, interval=self.interval, auto_adjust=False)
        return df

    def fetch_with_retries(self, symbol: str, attempts: int = RETRY_ATTEMPTS, backoff_base: float = RETRY_BACKOFF_BASE) -> Optional[pd.DataFrame]:
        fetch_sym = self._symbol_for_fetch(symbol)
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                df = self._try_fetch(fetch_sym)
                if df is None or df.empty:
                    last_exc = RuntimeError(f"empty data for {fetch_sym}")
                    raise last_exc
                df = df.copy()
                # normalize index -> date
                try:
                    df.index = pd.to_datetime(df.index).normalize()
                except Exception:
                    pass
                df = df.reset_index()
                # find date-like column and rename to 'date'
                date_col = None
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c
                        break
                if date_col and date_col != "date":
                    df = df.rename(columns={date_col: "date"})
                elif "index" in df.columns and "date" not in df.columns:
                    df = df.rename(columns={"index": "date"})
                # normalize column names
                df.columns = [str(c).strip().lower() for c in df.columns]
                return df
            except Exception:
                last_exc = last_exc or RuntimeError(f"fetch failed for {fetch_sym}")
                if attempt < attempts:
                    delay = (backoff_base ** attempt) + (0.1 * attempt)
                    time.sleep(delay)
                else:
                    return None
        return None


class ADXCalculator:
    def __init__(self, period: int = 10):
        self.period = period

    def _rma(self, series: pd.Series) -> pd.Series:
        n = self.period
        out = pd.Series(index=series.index, dtype=float)
        if len(series) < n:
            return out
        out.iloc[n - 1] = series.iloc[:n].mean()
        for i in range(n, len(series)):
            out.iloc[i] = (out.iloc[i - 1] * (n - 1) + series.iloc[i]) / n
        return out

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy().reset_index(drop=True)

        # ensure floats for calc
        for c in ("high", "low", "close"):
            if c not in df.columns:
                return df
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)

        up_move = high.diff().fillna(0.0)
        down_move = -low.diff().fillna(0.0)

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

        atr = self._rma(tr)
        plus_dm_sm = self._rma(plus_dm)
        minus_dm_sm = self._rma(minus_dm)

        atr_safe = atr.replace(0, np.nan)

        di_plus = 100 * (plus_dm_sm / atr_safe)
        di_minus = 100 * (minus_dm_sm / atr_safe)

        denom = (di_plus + di_minus).replace(0, np.nan)
        dx = 100 * (di_plus - di_minus).abs() / denom
        dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        adx = self._rma(dx)

        df[f"di_plus_{self.period}"] = di_plus.round(2)
        df[f"adx_{self.period}"] = adx.round(2)
        return df


def _process_symbol_result(symbol: str, hist: Optional[pd.DataFrame], meta_row: pd.Series, period_days: int) -> Optional[pd.DataFrame]:
    if hist is None or hist.empty:
        return None
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(hist.columns)):
        return None

    hist = hist[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    hist[["open", "high", "low", "close"]] = hist[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").astype(float)
    hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce")

    adx_calc = ADXCalculator(period_days)
    hist = adx_calc.calculate(hist)

    # convert price & volume to nullable integer after ADX calc
    for c in ("open", "close", "high", "low"):
        if c in hist.columns:
            hist[c] = hist[c].round(0).astype("Int64")
    if "volume" in hist.columns:
        hist["volume"] = hist["volume"].round(0).astype("Int64")

    di_col = f"di_plus_{period_days}"
    adx_col = f"adx_{period_days}"
    if di_col in hist.columns:
        hist[di_col] = pd.to_numeric(hist[di_col], errors="coerce").round(2)
    if adx_col in hist.columns:
        hist[adx_col] = pd.to_numeric(hist[adx_col], errors="coerce").round(2)

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date.astype(str)

    hist["symbol"] = str(meta_row["Symbol"]).strip()
    hist["sector"] = meta_row.get("Sector", "")
    hist["market cap"] = meta_row.get("Market Cap", "")

    cols = ["symbol", "sector", "market cap", "date", "open", "close", "high", "low", "volume", di_col, adx_col]
    available = [c for c in cols if c in hist.columns]
    hist = hist.reindex(columns=available)
    return hist


def build_historical_enriched_dataframe_parallel(csv_path: str = "files/stocks.csv", period_days: int = 10,
                                                 max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    reader = CSVReader(csv_path=csv_path)
    base_df = reader.read()

    fetcher = YFinanceFetcher(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL, nse_suffix=NSE_SUFFIX)
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
            processed = _process_symbol_result(symbol, hist, meta_row, period_days)
            if processed is not None and not processed.empty:
                results.append(processed)

    if not results:
        cols = ["symbol", "sector", "market cap", "date", "open", "close", "high", "low", "volume", f"di_plus_{period_days}", f"adx_{period_days}"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(results, ignore_index=True)
    out = out.rename(columns={"symbol": "Symbol", "sector": "Sector", "market cap": "Market Cap", "date": "Date"})
    out = out.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df_out = build_historical_enriched_dataframe_parallel("files/stocks.csv", period_days=10, max_workers=MAX_WORKERS)
    if not df_out.empty:
        df_out.to_csv("output/adx_output.csv", index=False)
        print("Saved to output/adx_output.csv")
    else:
        print("No historical ADX data produced.")
