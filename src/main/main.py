from stock_loader import StockList
from stock_data_fetcher_parallel import fetch_all_stocks_parallel
from rsi_calculator import RSI_Calculator

# Load stock list
loader = StockList("data/sector-ridewinner.csv").load()

# Fetch historical data in parallel (max_workers controls concurrency)
price_data = fetch_all_stocks_parallel(loader, max_workers=5, retries=3, delay=1)

# Compute RSI and cumulative sums
rsi_calc = RSI_Calculator()
final_df = rsi_calc.add_rsi_columns(price_data, period=14, last_days=90)

# Save final output
final_df.to_csv("output.csv", index=False)
