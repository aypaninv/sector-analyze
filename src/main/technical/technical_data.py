import pandas as pd
import os
import argparse

DEFAULT_FOURHOURS_TECH_FILE = "output/technical_4hours_output.csv"
DEFAULT_DAILY_TECH_FILE = "output/technical_daily_output.csv"
DEFAULT_WEEKLY_TECH_FILE = "output/technical_weekly_output.csv"
DEFAULT_MONTHLY_TECH_FILE = "output/technical_monthly_output.csv"

# 🔑 Monthly PRICE file (for ALL-TIME HIGH)
DEFAULT_MONTHLY_PRICE_FILE = "output/yfinance_monthly_output.csv"

DEFAULT_OUTPUT_FILE = "output/technical_data.csv"


def count_macd_strength(df: pd.DataFrame) -> int:
    df = df.sort_values("Date")
    count = 0
    for _, row in df[::-1].iterrows():
        if row["MACD_Fast"] > row["MACD_Slow"]:
            if count < 0:
                break
            count += 1
        elif row["MACD_Fast"] < row["MACD_Slow"]:
            if count > 0:
                break
            count -= 1
        else:
            break
    return count


def prepare_strength_map(path: str) -> dict:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return {
        symbol: count_macd_strength(g)
        for symbol, g in df.groupby("Symbol")
    }


def prepare_latest_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return (
        df.sort_values("Date")
          .groupby("Symbol")
          .tail(1)
          .reset_index(drop=True)
    )


def prepare_latest_rsi_map(path: str) -> dict:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if "RSI" not in df.columns:
        return {}
    return (
        df.sort_values("Date")
          .groupby("Symbol")
          .tail(1)
          .set_index("Symbol")["RSI"]
          .round(2)
          .to_dict()
    )


def prepare_monthly_ath_map(path: str) -> dict:
    """
    MONTHLY ALL-TIME HIGH per symbol
    """
    df = pd.read_csv(path)
    return df.groupby("Symbol")["High"].max().round(2).to_dict()


def build_data(
    daily_file,
    weekly_file,
    fourhours_file,
    monthly_tech_file,
    monthly_price_file
):
    # 🔑 Latest DAILY close
    daily_latest_df = prepare_latest_map(daily_file)

    # 🔑 Monthly ALL-TIME HIGH
    monthly_ath_map = prepare_monthly_ath_map(monthly_price_file)

    macd_4h = prepare_strength_map(fourhours_file)
    macd_day = prepare_strength_map(daily_file)
    macd_week = prepare_strength_map(weekly_file)
    macd_month = prepare_strength_map(monthly_tech_file)

    day_rsi_map = prepare_latest_rsi_map(daily_file)
    week_rsi_map = prepare_latest_rsi_map(weekly_file)
    month_rsi_map = prepare_latest_rsi_map(monthly_tech_file)

    rows = []

    for _, row in daily_latest_df.iterrows():
        symbol = row["Symbol"]
        daily_close = row["Close"]

        ath = monthly_ath_map.get(symbol)

        dd_high = (
            round(((daily_close - ath) / ath) * 100, 2)
            if ath and ath != 0
            else ""
        )

        rows.append({
            "Symbol": symbol,
            # ✅ Rolled back to DAILY CLOSE
            "Close": round(daily_close, 2),
            # ✅ DD from MONTHLY ALL-TIME HIGH
            "DD_High": dd_high,
            "4H_MACD": macd_4h.get(symbol, 0),
            "Day_MACD": macd_day.get(symbol, 0),
            "Week_MACD": macd_week.get(symbol, 0),
            "Month_MACD": macd_month.get(symbol, 0),
            "Day_RSI": day_rsi_map.get(symbol, ""),
            "Week_RSI": week_rsi_map.get(symbol, ""),
            "Month_RSI": month_rsi_map.get(symbol, ""),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-tech", default=DEFAULT_DAILY_TECH_FILE)
    parser.add_argument("--weekly-tech", default=DEFAULT_WEEKLY_TECH_FILE)
    parser.add_argument("--fourhours-tech", default=DEFAULT_FOURHOURS_TECH_FILE)
    parser.add_argument("--monthly-tech", default=DEFAULT_MONTHLY_TECH_FILE)
    parser.add_argument("--monthly-price", default=DEFAULT_MONTHLY_PRICE_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data_df = build_data(
        daily_file=args.daily_tech,
        weekly_file=args.weekly_tech,
        fourhours_file=args.fourhours_tech,
        monthly_tech_file=args.monthly_tech,
        monthly_price_file=args.monthly_price
    )

    data_df.to_csv(args.output, index=False)
    print(f"[OK] Technical data generated: {args.output} ({len(data_df)} stocks)")
