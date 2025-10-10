import pandas as pd

class RSI_Calculator:
    """Calculate Daily and Weekly RSI using Wilder's method (TradingView style)."""

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14):
        """
        Correct Wilder's RSI for pandas: daily and weekly.
        Aligns exactly with the original series index.
        """
        series = pd.to_numeric(series, errors="coerce")
        delta = series.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rsi = pd.Series(index=series.index, dtype=float)

        for i in range(len(series)):
            if i < period:
                rsi.iloc[i] = None  # Not enough data for RSI
            elif i == period:
                rs = avg_gain.iloc[i] / avg_loss.iloc[i] if avg_loss.iloc[i] != 0 else 0
                rsi.iloc[i] = 100 - (100 / (1 + rs))
            else:
                avg_gain.iloc[i] = (avg_gain.iloc[i-1]*(period-1) + gain.iloc[i])/period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1]*(period-1) + loss.iloc[i])/period
                rs = avg_gain.iloc[i] / avg_loss.iloc[i] if avg_loss.iloc[i] != 0 else 0
                rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi

    def add_rsi_columns(self, df: pd.DataFrame, period: int = 14):
        """
        Add daily and weekly RSI columns to the DataFrame.
        Returns a DataFrame with 'dailyRSI' and 'weeklyRSI' rounded to 2 decimals.
        """
        result = []

        # Flatten MultiIndex columns if exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(i) for i in col if i]) for col in df.columns]
        else:
            df.columns = [str(c).strip() for c in df.columns]

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        for symbol, sym_df in df.groupby("Symbol"):
            sym_df = sym_df.sort_values("Date").copy()

            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(sym_df["Date"]):
                sym_df["Date"] = pd.to_datetime(sym_df["Date"], errors="coerce")

            # Determine Close column
            available_cols = [c.strip() for c in sym_df.columns]
            if "Close" in available_cols:
                close_col = "Close"
            elif "Adj Close" in available_cols:
                close_col = "Adj Close"
            else:
                numeric_cols = sym_df.select_dtypes(include="number").columns
                if len(numeric_cols) == 0:
                    raise ValueError(f"No numeric column found for symbol {symbol}")
                close_col = numeric_cols[0]

            # --- Daily RSI ---
            close_series = sym_df[close_col]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            sym_df["dailyRSI"] = self.calculate_rsi(close_series, period=period)

            # --- Weekly RSI (Friday close) ---
            sym_df["Week"] = sym_df["Date"].dt.to_period("W-FRI")
            temp_df = sym_df.loc[:, ["Date", close_col, "Week"]].copy()
            weekly = temp_df.groupby("Week", as_index=False).last()

            weekly_close = weekly[close_col].reset_index(drop=True)
            weekly_rsi = self.calculate_rsi(weekly_close, period=period)
            weekly["weeklyRSI"] = weekly_rsi.values
            weekly = weekly[["Week", "weeklyRSI"]].copy()

            # Merge weekly RSI back
            sym_df = sym_df.merge(weekly, on="Week", how="left", validate="m:1")
            result.append(sym_df)

        # Concatenate all symbols
        final_df = pd.concat(result, ignore_index=True)
        final_df = final_df.drop(columns=["Week"], errors="ignore")

        # Round numeric columns to 2 decimals
        for col in ["Close", "dailyRSI", "weeklyRSI"]:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(
                    lambda x: float(f"{x:.2f}") if pd.notnull(x) else x
                )

        return final_df
