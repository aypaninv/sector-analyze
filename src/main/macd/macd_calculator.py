import pandas as pd
import os
import argparse

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_DAILY_INPUT = "output/yfinance_daily_output.csv"
DEFAULT_WEEKLY_INPUT = "output/yfinance_weekly_output.csv"
DEFAULT_FOURHOURS_INPUT = "output/yfinance_4hours_output.csv"

DEFAULT_DAILY_OUTPUT = "output/macd_daily_output.csv"
DEFAULT_WEEKLY_OUTPUT = "output/macd_weekly_output.csv"
DEFAULT_FOURHOURS_OUTPUT = "output/macd_4hours_output.csv"

FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
# -----------------------------------------------


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MACD (Fast & Slow) per symbol independently
    """

    df = df.sort_values(["Symbol", "Date"]).copy()

    def macd_for_symbol(g: pd.DataFrame) -> pd.DataFrame:
        close = g["Close"]

        ema_fast = close.ewm(span=FAST_PERIOD, adjust=False).mean()
        ema_slow = close.ewm(span=SLOW_PERIOD, adjust=False).mean()

        macd_fast = ema_fast - ema_slow
        macd_slow = macd_fast.ewm(span=SIGNAL_PERIOD, adjust=False).mean()

        g["MACD_Fast"] = macd_fast.round(2)
        g["MACD_Slow"] = macd_slow.round(2)

        return g

    df = (
        df.groupby("Symbol", group_keys=True)
          .apply(macd_for_symbol, include_groups=False)
          .reset_index(level=0)
    )

    return df


def process_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"[SKIP] Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # ✅ Folio removed
    required_cols = {
        "Symbol", "Sector", "MarketCap", "Date", "Close"
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_path}")

    df["Date"] = pd.to_datetime(df["Date"])

    df = calculate_macd(df)

    output_cols = [
        "Symbol",
        "Sector",
        "MarketCap",
        "Date",
        "Close",
        "MACD_Fast",
        "MACD_Slow",
    ]

    df = df[output_cols]

    df.to_csv(output_path, index=False)
    print(f"[OK] MACD file generated: {output_path} ({len(df)} rows)")


# ---------------- Main (CLI) ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calculate MACD for Daily, Weekly, and 4H OHLC data"
    )

    parser.add_argument("--daily-input", default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--daily-output", default=DEFAULT_DAILY_OUTPUT)

    parser.add_argument("--weekly-input", default=DEFAULT_WEEKLY_INPUT)
    parser.add_argument("--weekly-output", default=DEFAULT_WEEKLY_OUTPUT)

    parser.add_argument("--fourhours-input", default=DEFAULT_FOURHOURS_INPUT)
    parser.add_argument("--fourhours-output", default=DEFAULT_FOURHOURS_OUTPUT)

    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    # Daily
    process_file(args.daily_input, args.daily_output)

    # Weekly
    process_file(args.weekly_input, args.weekly_output)

    # 4 Hours
    process_file(args.fourhours_input, args.fourhours_output)
