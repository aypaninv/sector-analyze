import pandas as pd
import os

# ---------------- CONFIG ----------------
FOURHOURS_MACD_FILE = "output/macd_4hours_output.csv"
DAILY_MACD_FILE = "output/macd_daily_output.csv"
WEEKLY_MACD_FILE = "output/macd_weekly_output.csv"

OUTPUT_FILE = "output/macd_report.csv"
# --------------------------------------


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

    strength = {}
    for symbol, g in df.groupby("Symbol"):
        strength[symbol] = count_macd_strength(g)

    return strength


def prepare_weekly_high_map(path: str) -> dict:
    df = pd.read_csv(path)

    weekly_high = (
        df.groupby("Symbol")["Close"]
          .max()
          .to_dict()
    )

    return weekly_high


def prepare_latest_daily_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    latest = (
        df.sort_values("Date")
          .groupby("Symbol")
          .tail(1)
          .reset_index(drop=True)
    )

    return latest


def build_report():
    # ---------- Load reference data ----------
    daily_latest_df = prepare_latest_daily_map(DAILY_MACD_FILE)
    weekly_high_map = prepare_weekly_high_map(WEEKLY_MACD_FILE)

    strength_4h = prepare_strength_map(FOURHOURS_MACD_FILE)
    strength_daily = prepare_strength_map(DAILY_MACD_FILE)
    strength_weekly = prepare_strength_map(WEEKLY_MACD_FILE)

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
            "Folio": row["Folio"],
            "Sector": row["Sector"],
            "MarketCap": row["MarketCap"],
            "Close": round(close, 2),
            "DownFromHigh_%": down_from_high,
            "cross_4h": strength_4h.get(symbol, 0),
            "cross_daily": strength_daily.get(symbol, 0),
            "cross_weekly": strength_weekly.get(symbol, 0),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    report_df = build_report()
    report_df.to_csv(OUTPUT_FILE, index=False)

    print(f"[OK] MACD report generated: {OUTPUT_FILE} ({len(report_df)} stocks)")
