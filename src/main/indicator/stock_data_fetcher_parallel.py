import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from stock_data_fetcher import StockDataFetcher  # import your original class

def fetch_symbol_parallel(stock_row, retries=3, delay=1):
    """
    Fetch one symbol safely using the original StockDataFetcher.
    Retries if it fails, returns a DataFrame or None.
    """
    symbol = stock_row["Symbol"].strip().upper()
    for attempt in range(1, retries + 1):
        try:
            fetcher = StockDataFetcher(pd.DataFrame([stock_row]))
            df = fetcher.fetch_all()
            return df
        except Exception as e:
            print(f"⚠️ Error fetching {symbol} on attempt {attempt}: {e}")
            time.sleep(delay)
    print(f"❌ Failed to fetch {symbol} after {retries} attempts.")
    return None

def fetch_all_stocks_parallel(stock_df, max_workers=5, retries=3, delay=1):
    """
    Fetch all symbols in parallel with retry and sequential fallback.
    Returns concatenated DataFrame.
    """
    all_data = []
    failed_symbols = []

    # --- Parallel fetching ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_symbol_parallel, row, retries, delay): row["Symbol"]
            for _, row in stock_df.iterrows()
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if df is not None:
                    all_data.append(df)
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                print(f"❌ {symbol} generated an exception: {e}")
                failed_symbols.append(symbol)

    # --- Sequential retry for failed symbols ---
    if failed_symbols:
        print(f"\nRetrying failed symbols sequentially: {failed_symbols}")
        for symbol in failed_symbols:
            row = stock_df[stock_df["Symbol"] == symbol].iloc[0]
            df = fetch_symbol_parallel(row, retries=retries, delay=delay)
            if df is not None:
                all_data.append(df)
            else:
                print(f"❌ Could not fetch data for {symbol} at all.")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Successfully fetched data for {final_df['Symbol'].nunique()} symbols.")
        return final_df
    else:
        raise RuntimeError("No stock data fetched.")
