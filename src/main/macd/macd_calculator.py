import pandas as pd
import os

# ---------------- CONFIG ----------------
DAILY_INPUT = "output/yfinance_daily_output.csv"
WEEKLY_INPUT = "output/yfinance_weekly_output.csv"
FOURHOURS_INPUT = "output/yfinance_4hours_output.csv"

DAILY_OUTPUT = "output/macd_daily_output.csv"
WEEKLY_OUTPUT = "output/macd_weekly_output.csv"
FOURHOURS_OUTPUT = "output/macd_4hours_output.csv"

FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
# ---------------------------------------


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
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = {"Symbol", "Folio", "Sector", "MarketCap", "Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_path}")

    df["Date"] = pd.to_datetime(df["Date"])

    df = calculate_macd(df)

    output_cols = [
        "Symbol",
        "Folio",
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


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Daily MACD
    process_file(DAILY_INPUT, DAILY_OUTPUT)

    # Weekly MACD
    process_file(WEEKLY_INPUT, WEEKLY_OUTPUT)

    # 4 Hours MACD
    process_file(FOURHOURS_INPUT, FOURHOURS_OUTPUT)
