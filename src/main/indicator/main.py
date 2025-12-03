#!/usr/bin/env python3
"""
main.py

Flow:
 - Load stock list (from stock_loader.get_stock_list() if available OR from files/stocks.csv)
 - Fetch OHLCV (High, Low, Close, Open, Volume) via StockDataFetcher
 - Compute RSI (daily & weekly) using rsi_calculator.RSI_Calculator
 - Compute ADX/DI+ (daily & weekly) using adx_calculator.ADX_Calculator
 - Save outputs to output/ folder

Usage:
    python main.py --source files/stocks.csv --period 14 --last-days 180 --parallel
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Attempt imports (fall back gracefully if parallel fetcher exists)
try:
    from stock_data_fetcher import StockDataFetcher
except Exception:
    StockDataFetcher = None

try:
    from stock_data_fetcher_parallel import StockDataFetcherParallel as StockDataFetcherParallel
except Exception:
    StockDataFetcherParallel = None

from rsi_calculator import RSI_Calculator
from adx_calculator import ADX_Calculator

# Optional user-provided loader
try:
    import stock_loader
except Exception:
    stock_loader = None


def ensure_output_dir(path: str = "output"):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_stock_list(source_csv: str = "files/stocks.csv") -> pd.DataFrame:
    """
    Try to load a list of stocks:
     1) If stock_loader.get_stock_list exists and returns a DataFrame/list -> use it
     2) Else read CSV at source_csv (expects a header including 'Symbol' column)
    Returns a DataFrame with at least a 'Symbol' column. May also have Sector / Market Cap.
    """
    if stock_loader is not None and hasattr(stock_loader, "get_stock_list"):
        try:
            loader_result = stock_loader.get_stock_list()
            if isinstance(loader_result, pd.DataFrame):
                df = loader_result
            elif isinstance(loader_result, (list, tuple)):
                df = pd.DataFrame(loader_result, columns=["Symbol"])
            else:
                df = pd.DataFrame(loader_result)
            if "Symbol" not in df.columns:
                raise ValueError("stock_loader.get_stock_list didn't return a 'Symbol' column")
            print("Loaded stock list from stock_loader.get_stock_list()")
            return df
        except Exception as e:
            print(f"⚠️ stock_loader.get_stock_list() failed: {e}. Falling back to CSV.")

    # Fallback CSV
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Stock source file not found: {source_csv}")
    df = pd.read_csv(source_csv, dtype=str)
    if "Symbol" not in df.columns:
        # try first column if header different
        df.columns = [c.strip() for c in df.columns]
        if "Symbol" not in df.columns:
            # assume first column is symbol
            df = df.rename(columns={df.columns[0]: "Symbol"})
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    print(f"Loaded {len(df)} symbols from {source_csv}")
    return df


def fetch_data(stock_df: pd.DataFrame, use_parallel: bool = False, start_days: int = 365):
    """
    Use available StockDataFetcher implementation to pull OHLCV.
    """
    if use_parallel and StockDataFetcherParallel is not None:
        print("Using StockDataFetcherParallel for fetching...")
        fetcher = StockDataFetcherParallel(stock_df)
    elif StockDataFetcher is not None:
        print("Using StockDataFetcher for fetching...")
        fetcher = StockDataFetcher(stock_df)
    else:
        raise RuntimeError("No StockDataFetcher implementation found. Provide stock_data_fetcher.py or stock_data_fetcher_parallel.py")
    data = fetcher.fetch_all()
    # Ensure Date is datetime
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    return data


def main(args):
    ensure_output_dir(args.output_dir)

    # 1) Load stock list
    stock_df = load_stock_list(args.source)

    # 2) Fetch OHLCV (High/Low/Close required)
    try:
        raw_data = fetch_data(stock_df, use_parallel=args.parallel)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        sys.exit(2)

    if raw_data.empty:
        print("❌ No data fetched. Exiting.")
        sys.exit(3)

    # Save raw fetch for debugging
    raw_out = os.path.join(args.output_dir, "raw_fetch.csv")
    raw_data.to_csv(raw_out, index=False)
    print(f"Saved raw fetch to {raw_out}")

    # 3) Compute RSI
    rsi_calc = RSI_Calculator(period=args.period)
    # rsi_calculator.add_rsi_columns expects DataFrame with Symbol, Date, Close etc.
    print("Calculating RSI (daily & weekly)...")
    try:
        rsi_df = rsi_calc.add_rsi_columns(raw_data, period=args.period, last_days=args.last_days)
    except Exception as e:
        print(f"❌ RSI calculation failed: {e}")
        sys.exit(4)

    rsi_out = os.path.join(args.output_dir, "with_rsi.csv")
    rsi_df.to_csv(rsi_out, index=False)
    print(f"Saved RSI output to {rsi_out}")

    # 4) Compute ADX and DI+
    adx_calc = ADX_Calculator(period=args.period)
    print("Calculating ADX and DI+ (daily & weekly)...")
    try:
        final_df = adx_calc.add_adx_columns(rsi_df, period=args.period, last_days=args.last_days)
    except Exception as e:
        print(f"❌ ADX calculation failed: {e}")
        sys.exit(5)

    final_out = os.path.join(args.output_dir, "final_with_rsi_adx.csv")
    final_df.to_csv(final_out, index=False)
    print(f"Saved final output to {final_out}")

    # Print a short summary
    symbols_done = final_df["Symbol"].nunique() if "Symbol" in final_df.columns else 0
    rows = len(final_df)
    print(f"✅ Completed. Symbols processed: {symbols_done}. Rows in final output: {rows}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV, compute RSI and ADX/DI+ and save CSVs")
    parser.add_argument("--source", "-s", default="files/stocks.csv", help="CSV file with stock list (must contain 'Symbol' header) or will use stock_loader.get_stock_list() if available")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--period", "-p", type=int, default=14, help="Lookback period for RSI/ADX (default 14)")
    parser.add_argument("--last-days", type=int, default=90, help="Number of last daily rows to keep per symbol in the final output")
    parser.add_argument("--parallel", action="store_true", help="Use stock_data_fetcher_parallel.StockDataFetcherParallel if available")
    args = parser.parse_args()
    main(args)
