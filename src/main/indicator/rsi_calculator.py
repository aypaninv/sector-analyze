import pandas as pd
import numpy as np

class RSI_Calculator:
    """Calculate daily and weekly RSI values (TradingView-style) with 2-decimal precision."""

    def __init__(self, period=14):
        self.period = period

    def calculate_rsi(self, series: pd.Series) -> pd.Series:
        """Compute RSI using Wilder’s smoothing method (like TradingView)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily OHLC to weekly frequency (W-FRI). Keep High=max, Low=min, Close=last."""
        df = df.set_index("Date")
        weekly = df.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna().reset_index()
        return weekly

    def add_rsi_columns(self, df: pd.DataFrame, period: int = 14, last_days: int = 90) -> pd.DataFrame:
        """Add RSI columns (daily & weekly) and keep only the latest 90 days."""
        all_data = []

        for symbol, sym_df in df.groupby("Symbol"):
            sym_df = sym_df.sort_values("Date").reset_index(drop=True)
            close_series = sym_df["Close"]

            # --- Daily RSI ---
            sym_df["dailyRSI"] = self.calculate_rsi(close_series)

            # --- Weekly RSI ---
            weekly_df = self.resample_to_weekly(sym_df)
            weekly_df["weeklyRSI"] = self.calculate_rsi(weekly_df["Close"])

            # Merge weekly RSI back to daily
            sym_df = pd.merge_asof(
                sym_df.sort_values("Date"),
                weekly_df[["Date", "weeklyRSI"]].sort_values("Date"),
                on="Date",
                direction="backward"
            )

            # --- Keep only latest N days ---
            sym_df = sym_df.tail(last_days)

            # --- Round to 2 decimal places for numeric cols we care about ---
            for col in ["Close", "dailyRSI", "weeklyRSI"]:
                if col in sym_df.columns:
                    sym_df[col] = sym_df[col].round(2)

            all_data.append(sym_df)

        final_df = pd.concat(all_data, ignore_index=True)

        # Preserve OHLC columns if available
        cols = [c for c in ["Symbol", "Sector", "Market Cap", "Date", "Open", "High", "Low", "Close", "Volume", "dailyRSI", "weeklyRSI"] if c in final_df.columns]
        return final_df[cols]