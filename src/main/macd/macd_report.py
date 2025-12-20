import pandas as pd
import os
import argparse

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_FOURHOURS_MACD_FILE = "output/macd_4hours_output.csv"
DEFAULT_DAILY_MACD_FILE = "output/macd_daily_output.csv"
DEFAULT_WEEKLY_MACD_FILE = "output/macd_weekly_output.csv"

DEFAULT_OUTPUT_FILE = "output/macd_report.csv"
# -----------------------------------------------


def count_macd_strength(df: pd.DataFrame) -> int:
    """
    Count consecutive periods MACD_Fast vs MACD_Slow from latest backwards.
    +ve  -> MACD_Fast above MACD_Slow
    -ve  -> MACD_Fast below MACD_Slow
    """
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


def prepare_weekly_high_map(path: str) -> dict:
    df = pd.read_csv(path)
    return df.groupby("Symbol")["Close"].max().to_dict()


def prepare_latest_daily_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    return (
        df.sort_values("Date")
          .groupby("Symbol")
          .tail(1)
          .reset_index(drop=True)
    )


def build_report(
    daily_macd_file: str,
    weekly_macd_file: str,
    fourhours_macd_file: str
) -> pd.DataFrame:

    # ---------- Load reference data ----------
    daily_latest_df = prepare_latest_daily_map(daily_macd_file)
    weekly_high_map = prepare_weekly_high_map(weekly_macd_file)

    strength_4h = prepare_strength_map(fourhours_macd_file)
    strength_daily = prepare_strength_map(daily_macd_file)
    strength_weekly = prepare_strength_map(weekly_macd_file)

    rows = []

    for _, row in daily_latest_df.iterrows():
        symbol = row["Symbol"]
        close = row["Close"]

        weekly_high = weekly_high_map.get(symbol)
        down_from_high = (
            round(((close - weekly_high) / weekly_high) * 100, 2)
            if weekly_high and weekly_high != 0
            else ""
        )

        rows.append({
            "Symbol": symbol,
            "Sector": row.get("Sector", ""),
            "MarketCap": row.get("MarketCap", ""),
            "Close": round(close, 2),
            "Down_%": down_from_high,
            "4hStrength": strength_4h.get(symbol, 0),
            "DayStrength": strength_daily.get(symbol, 0),
            "WeekStrength": strength_weekly.get(symbol, 0),
        })

    return pd.DataFrame(rows)


# ---------------- Main (CLI) ----------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate consolidated MACD strength report (Daily / Weekly / 4H)"
    )

    parser.add_argument(
        "--daily-macd",
        default=DEFAULT_DAILY_MACD_FILE,
        help="Daily MACD CSV file"
    )

    parser.add_argument(
        "--weekly-macd",
        default=DEFAULT_WEEKLY_MACD_FILE,
        help="Weekly MACD CSV file"
    )

    parser.add_argument(
        "--fourhours-macd",
        default=DEFAULT_FOURHOURS_MACD_FILE,
        help="4 Hours MACD CSV file"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output MACD report CSV file"
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report_df = build_report(
        daily_macd_file=args.daily_macd,
        weekly_macd_file=args.weekly_macd,
        fourhours_macd_file=args.fourhours_macd
    )

    report_df.to_csv(args.output, index=False)

    print(f"[OK] MACD report generated: {args.output} ({len(report_df)} stocks)")
