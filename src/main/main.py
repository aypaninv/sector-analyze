from stock_loader import StockList
from stock_data_fetcher import StockDataFetcher
from rsi_calculator import RSI_Calculator
import os

if __name__ == "__main__":
    # Load stock list
    stocks = StockList("data/sector-ridewinner.csv").load()

    # Fetch price data
    fetcher = StockDataFetcher(stocks)
    price_data = fetcher.fetch_all()

    # Calculate RSI
    rsi_calc = RSI_Calculator()
    final_df = rsi_calc.add_rsi_columns(price_data)

    # Save final CSV
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "nse_stocks_with_rsi.csv")
    final_df.to_csv(output_file, index=False)

    print(f"✅ Final CSV saved: {output_file}")
