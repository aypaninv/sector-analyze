"""
adx_calculator.py

Calculate ADX and +DI (DI+) for daily and weekly frequencies.

Usage:
    from adx_calculator import ADX_Calculator
    adx = ADX_Calculator(period=14)
    df_with_adx = adx.add_adx_columns(ohlc_df, period=14, last_days=90)

Expectations:
 - Input df must contain columns: ['Symbol', 'Date', 'High', 'Low', 'Close'] at minimum.
 - 'Date' must be a datetime-like column (not necessarily index).
 - The function preserves other columns (Open, Volume, dailyRSI, weeklyRSI) if present.
"""

import pandas as pd
import numpy as np
from typing import Optional


class ADX_Calculator:
    def __init__(self, period: int = 14):
        """
        Initialize ADX calculator.

        :param period: Lookback period for DI and ADX (default 14).
        """
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = int(period)

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
        """
        Compute True Range (TR) for each row.
        TR = max(High-Low, abs(High - prevClose), abs(Low - prevClose))
        """
        a = high - low
        b = (high - close_prev).abs()
        c = (low - close_prev).abs()
        tr = pd.concat([a, b, c], axis=1).max(axis=1)
        return tr

    def _calculate_adx_for_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame with sorted Date and columns High, Low, Close, compute:
          - TR (unsmoothed)
          - +DM and -DM (unsmoothed)
          - smoothed TR, +DM, -DM (Wilder smoothing using ewm alpha=1/period)
          - +DI, -DI, DX, ADX (smoothed DX)
        Returns a DataFrame aligned with the input df (same row count).
        """
        # require columns
        if not set(["High", "Low", "Close"]).issubset(df.columns):
            raise ValueError("DataFrame must contain 'High', 'Low', 'Close' columns")

        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # Directional movements
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # True range
        tr = self._true_range(high, low, prev_close)

        # Wilder smoothing via EWM with alpha = 1/period and adjust=False
        # min_periods=self.period ensures NaNs for initial entries until enough data
        tr_smooth = tr.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        # Avoid division by zero by replacing zeros with NaN before division
        tr_safe = tr_smooth.replace(0, np.nan)

        plus_di = 100.0 * (plus_dm_smooth / tr_safe)
        minus_di = 100.0 * (minus_dm_smooth / tr_safe)

        # DX: absolute difference over sum
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0

        # ADX: smoothed DX
        adx = dx.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        out = pd.DataFrame({
            "TR": tr,
            "+DM": plus_dm,
            "-DM": minus_dm,
            "TR_smooth": tr_smooth,
            "+DM_smooth": plus_dm_smooth,
            "-DM_smooth": minus_dm_smooth,
            "DIPlus": plus_di,
            "DIMinus": minus_di,
            "DX": dx,
            "ADX": adx
        }, index=df.index)

        return out

    def resample_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily OHLC rows to weekly OHLC (W-FRI).
        Aggregation:
          - Open: first
          - High: max
          - Low: min
          - Close: last
          - Volume: sum (if present)
        Returns a new DataFrame with a 'Date' column (week-ending date) and OHLC columns.
        """
        df = df.copy()
        if "Date" not in df.columns:
            raise ValueError("DataFrame must contain a 'Date' column for resampling")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        agg_map = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last"
        }
        if "Volume" in df.columns:
            agg_map["Volume"] = "sum"

        weekly = df.resample("W-FRI").agg(agg_map).dropna().reset_index()
        return weekly

    def add_adx_columns(self, df: pd.DataFrame, period: Optional[int] = None, last_days: int = 90) -> pd.DataFrame:
        """
        Add daily and weekly ADX/DI+ columns to the input DataFrame.

        :param df: Input DataFrame with at least ['Symbol', 'Date', 'High', 'Low', 'Close'].
                   It may contain other columns (Open, Volume, dailyRSI, weeklyRSI) which are preserved.
        :param period: Optional override for lookback period (defaults to self.period).
        :param last_days: Number of last days to keep for each symbol in the returned DataFrame.
        :return: DataFrame containing OHLC, RSI (if present) and ADX/DI+ columns:
                 dailyDIPlus, dailyADX, weeklyDIPlus, weeklyADX (when computable).
        """
        if period is None:
            period = self.period
        else:
            if period < 1:
                raise ValueError("period must be >= 1")
            period = int(period)

        required_cols = {"Symbol", "Date", "High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        results = []

        # Ensure Date is datetime
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Group by symbol
        for symbol, sym_df in df.groupby("Symbol", sort=True):
            sym_df = sym_df.sort_values("Date").reset_index(drop=True)

            # Calculate daily ADX components
            daily_adx = self._calculate_adx_for_ohlc(sym_df)

            # Append the daily ADX columns to symbol df
            sym_df = pd.concat([sym_df.reset_index(drop=True), daily_adx.reset_index(drop=True)], axis=1)

            # Rename daily DIPlus/ADX columns to friendly names
            if "DIPlus" in sym_df.columns:
                sym_df = sym_df.rename(columns={"DIPlus": "dailyDIPlus"})
            if "ADX" in sym_df.columns:
                sym_df = sym_df.rename(columns={"ADX": "dailyADX"})

            # Weekly ADX: build weekly OHLC, compute ADX, then merge back using merge_asof
            try:
                weekly = self.resample_to_weekly(sym_df[["Date", "Open", "High", "Low", "Close", "Volume"]] if "Open" in sym_df.columns else sym_df[["Date", "High", "Low", "Close"]])
            except Exception:
                weekly = None

            if weekly is not None and not weekly.empty:
                weekly_adx = self._calculate_adx_for_ohlc(weekly)
                # attach Date so we can merge_asof
                weekly_adx = pd.concat([weekly[["Date"]].reset_index(drop=True), weekly_adx.reset_index(drop=True)], axis=1)

                # Rename weekly columns
                weekly_adx = weekly_adx.rename(columns={"DIPlus": "weeklyDIPlus", "ADX": "weeklyADX"})
                # Keep only Date, weeklyDIPlus, weeklyADX
                weekly_adx = weekly_adx[["Date", "weeklyDIPlus", "weeklyADX"]]

                # merge_asof to propagate latest weekly values to daily rows (backward)
                sym_df = pd.merge_asof(
                    sym_df.sort_values("Date"),
                    weekly_adx.sort_values("Date"),
                    on="Date",
                    direction="backward"
                )
            else:
                # If no weekly data, add empty weekly columns
                sym_df["weeklyDIPlus"] = np.nan
                sym_df["weeklyADX"] = np.nan

            # Keep only last N days
            sym_df = sym_df.tail(last_days).reset_index(drop=True)

            # Round useful numeric columns for readability (if present)
            for col in ["Close", "dailyDIPlus", "dailyADX", "weeklyDIPlus", "weeklyADX"]:
                if col in sym_df.columns:
                    sym_df[col] = pd.to_numeric(sym_df[col], errors="coerce").round(2)

            results.append(sym_df)

        if not results:
            raise RuntimeError("No symbols processed for ADX calculation.")

        final = pd.concat(results, ignore_index=True)

        # Choose preferred column order while preserving existing columns
        preferred = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
                     "dailyDIPlus", "dailyADX", "weeklyDIPlus", "weeklyADX"]
        # include any additional columns that user might have (like dailyRSI, weeklyRSI) after preferred
        remaining = [c for c in final.columns if c not in preferred]
        cols_to_return = [c for c in preferred if c in final.columns] + remaining

        return final[cols_to_return]


if __name__ == "__main__":
    # Quick self-check example (no real data fetch here). This demonstrates usage only.
    print("adx_calculator.py loaded. Use ADX_Calculator.add_adx_columns(...) with your OHLC DataFrame.")
