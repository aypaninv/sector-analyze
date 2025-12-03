# stock_data_fetcher.py
"""
StockDataFetcher - compatible replacement that mirrors original behavior.

Features:
 - Prefer yf.Ticker(symbol).history(...) (this tends to match the original attached code behavior)
 - Fallback to yf.download(...) if necessary
 - Normalize 'Adj Close' -> 'Close' when needed
 - Works for single-row input (used by your parallel fetcher) or multi-row DataFrame
 - Returns DataFrame with columns: Symbol, Date, Open, High, Low, Close, Volume (plus optional Sector/Market Cap/Fetched_Ticker)
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf


class StockDataFetcher:
    def __init__(self, stock_df: pd.DataFrame, lookback_days: int = 365, verbose: bool = True):
        if stock_df is None or stock_df.empty:
            raise ValueError("Stock list is empty.")
        if "Symbol" not in stock_df.columns:
            raise ValueError("stock_df must include a 'Symbol' column")
        self.stock_df = stock_df.reset_index(drop=True)
        self.lookback_days = int(lookback_days)
        self.verbose = bool(verbose)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize common column names and map 'Adj Close' to 'Close' if needed."""
        if df is None or df.empty:
            return df

        # Flatten MultiIndex if yfinance returns it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

        # Standardize names
        rename_map = {}
        for c in df.columns:
            lower = str(c).strip().lower()
            if lower in ("adj close", "adjclose", "adjusted_close", "adjustedclose"):
                rename_map[c] = "Adj Close"
            elif lower == "close":
                rename_map[c] = "Close"
            elif lower == "open":
                rename_map[c] = "Open"
            elif lower == "high":
                rename_map[c] = "High"
            elif lower == "low":
                rename_map[c] = "Low"
            elif lower == "volume":
                rename_map[c] = "Volume"
            elif lower in ("date", "index"):
                rename_map[c] = "Date"
        if rename_map:
            df = df.rename(columns=rename_map)
        # If only Adj Close present, create Close
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        return df

    def _history_fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """Primary fetch: use Ticker.history (preferred, matches many existing codebases)."""
        try:
            tk = yf.Ticker(ticker)
            period = f"{min(max(self.lookback_days, 7), 3650)}d"
            df = tk.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                return None
            # reset index to get Date
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            df = self._normalize_columns(df)
            # require Close data
            if "Close" not in df.columns or df["Close"].isna().all():
                return None
            return df
        except Exception:
            return None

    def _download_fetch(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fallback fetch: use yf.download with explicit start/end."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return None
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            df = self._normalize_columns(df)
            if "Close" not in df.columns or df["Close"].isna().all():
                return None
            return df
        except Exception:
            return None

    def _try_variants(self, symbol: str):
        """Return tuple (success_df, used_ticker) or (None, None) if no candidate worked."""
        # build candidate forms: try as-is, then common Indian suffixes, then uppercase plain
        candidates = []
        raw = symbol.strip()
        candidates.append(raw)
        upper = raw.upper()
        # only add suffixes for short simple tickers (avoid adding if already contains dot)
        if "." not in upper and len(upper) <= 10:
            candidates.extend([f"{upper}.NS", f"{upper}.BO", f"{upper}.NSE"])
        if upper not in candidates:
            candidates.append(upper)

        for cand in candidates:
            # 1) try history first (matching original)
            df = self._history_fetch(cand)
            if df is not None:
                return df, cand
            # 2) try download fallback
            df2 = self._download_fetch(cand)
            if df2 is not None:
                return df2, cand
        return None, None

    def fetch_all(self) -> pd.DataFrame:
        """Fetch OHLCV for each symbol in self.stock_df and return concatenated DataFrame."""
        rows = []
        fetched_count = 0
        for _, row in self.stock_df.iterrows():
            symbol = str(row.get("Symbol", "")).strip()
            sector = row.get("Sector") if "Sector" in row.index else None
            mcap = row.get("Market Cap") if "Market Cap" in row.index else None

            if not symbol:
                if self.verbose:
                    print("⚠️ Skipping empty symbol row")
                continue

            df, used_ticker = self._try_variants(symbol)
            if df is None:
                # print concise message but continue
                print(f"❌ Error fetching {symbol}: no usable Close found for tried ticker variants.")
                continue

            # Build symbol-specific DataFrame
            out = pd.DataFrame({
                "Symbol": symbol,
                "Date": df["Date"]
            })
            # Ensure we include OHLCV columns (fill with NaN if missing)
            for col in ("Open", "High", "Low", "Close", "Volume"):
                out[col] = df[col].values if col in df.columns else pd.NA

            out["Sector"] = sector
            out["Market Cap"] = mcap
            out["Fetched_Ticker"] = used_ticker

            rows.append(out)
            fetched_count += 1
            if self.verbose:
                print(f"✅ {symbol}: fetched via {used_ticker}")

        if not rows:
            # return an empty DataFrame with expected columns
            cols = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Sector", "Market Cap", "Fetched_Ticker"]
            return pd.DataFrame(columns=cols)

        result = pd.concat(rows, ignore_index=True)
        # ensure Date is datetime
        result["Date"] = pd.to_datetime(result["Date"])
        if self.verbose:
            print(f"✅ Fetched data for {fetched_count} symbols.")
        return result


if __name__ == "__main__":
    # quick self test (manual)
    print("StockDataFetcher module loaded. Use StockDataFetcher(stock_df).fetch_all()")
