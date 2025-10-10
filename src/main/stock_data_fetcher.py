import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataFetcher:
    """Fetch 1-year daily close prices for NSE stocks."""

    def __init__(self, stock_df: pd.DataFrame):
        self.stock_df = stock_df
        self.data = pd.DataFrame()

    def fetch_all(self):
        if self.stock_df is None or self.stock_df.empty:
            raise ValueError("Stock list is empty.")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        all_data = []

        for _, row in self.stock_df.iterrows():
            symbol = row["Symbol"].strip().upper()
            sector = row["Sector"]
            market_cap = row["Market Cap"]
            ticker_symbol = f"{symbol}.NS"

            print(f"Fetching data for {ticker_symbol}...")

            try:
                data = yf.download(
                    ticker_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )

                if data.empty:
                    print(f"⚠️ No data for {symbol}")
                    continue

                data = data.reset_index()

                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ["_".join([str(i) for i in col if i]) for col in data.columns]

                # Determine close column
                if "Close" in data.columns:
                    close_col = "Close"
                elif "Adj Close" in data.columns:
                    close_col = "Adj Close"
                else:
                    numeric_cols = data.select_dtypes(include='number').columns
                    close_col = numeric_cols[0]

                df_symbol = pd.DataFrame({
                    "Symbol": symbol,
                    "Sector": sector,
                    "Market Cap": market_cap,
                    "Date": data["Date"],
                    "Close": data[close_col]
                })

                all_data.append(df_symbol)

            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")

        if not all_data:
            raise RuntimeError("No stock data fetched.")

        self.data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Fetched data for {self.data['Symbol'].nunique()} symbols.")
        return self.data
