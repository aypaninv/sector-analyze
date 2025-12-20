import pandas as pd
import os
import argparse

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_DAILY_INPUT = "output/yfinance_daily_output.csv"
DEFAULT_WEEKLY_INPUT = "output/yfinance_weekly_output.csv"
DEFAULT_FOURHOURS_INPUT = "output/yfinance_4hours_output.csv"
DEFAULT_MONTHLY_INPUT = "output/yfinance_monthly_output.csv"

DEFAULT_DAILY_OUTPUT = "output/technical_daily_output.csv"
DEFAULT_WEEKLY_OUTPUT = "output/technical_weekly_output.csv"
DEFAULT_FOURHOURS_OUTPUT = "output/technical_4hours_output.csv"
DEFAULT_MONTHLY_OUTPUT = "output/technical_monthly_output.csv"

FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
RSI_PERIOD = 14
# -----------------------------------------------


def calculate_macd_rsi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Symbol", "Date"]).copy()

    def indicators(g):
        close = g["Close"]

        ema_fast = close.ewm(span=FAST_PERIOD, adjust=False).mean()
        ema_slow = close.ewm(span=SLOW_PERIOD, adjust=False).mean()

        macd_fast = ema_fast - ema_slow
        macd_slow = macd_fast.ewm(span=SIGNAL_PERIOD, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        g["MACD_Fast"] = macd_fast.round(2)
        g["MACD_Slow"] = macd_slow.round(2)
        g["RSI"] = rsi.round(2)
        return g

    return (
        df.groupby("Symbol", group_keys=True)
          .apply(indicators, include_groups=False)
          .reset_index(level=0)
    )


def process_file(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"[SKIP] Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    required_cols = {"Symbol", "Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {input_path}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = calculate_macd_rsi(df)

    df = df[
        ["Symbol", "Date", "Close", "MACD_Fast", "MACD_Slow", "RSI"]
    ]

    df.to_csv(output_path, index=False)
    print(f"[OK] Technical file generated: {output_path} ({len(df)} rows)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-input", default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--daily-output", default=DEFAULT_DAILY_OUTPUT)

    parser.add_argument("--weekly-input", default=DEFAULT_WEEKLY_INPUT)
    parser.add_argument("--weekly-output", default=DEFAULT_WEEKLY_OUTPUT)

    parser.add_argument("--fourhours-input", default=DEFAULT_FOURHOURS_INPUT)
    parser.add_argument("--fourhours-output", default=DEFAULT_FOURHOURS_OUTPUT)

    parser.add_argument("--monthly-input", default=DEFAULT_MONTHLY_INPUT)
    parser.add_argument("--monthly-output", default=DEFAULT_MONTHLY_OUTPUT)

    args = parser.parse_args()
    os.makedirs("output", exist_ok=True)

    process_file(args.daily_input, args.daily_output)
    process_file(args.weekly_input, args.weekly_output)
    process_file(args.fourhours_input, args.fourhours_output)
    process_file(args.monthly_input, args.monthly_output)
