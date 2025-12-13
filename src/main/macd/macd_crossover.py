import pandas as pd
import os

# ---------------- CONFIG ----------------
DAILY_MACD_FILE = "output/macd_daily_output.csv"
WEEKLY_MACD_FILE = "output/macd_weekly_output.csv"

OUTPUT_FILE = "output/macd_crossover_consolidated.csv"
# --------------------------------------


def find_latest_crossover(df: pd.DataFrame):
    """
    Find latest MACD crossover for a single symbol
    """
    df = df.sort_values("Date").copy()

    df["prev_fast"] = df["MACD_Fast"].shift(1)
    df["prev_slow"] = df["MACD_Slow"].shift(1)

    cross_up = (df["prev_fast"] <= df["prev_slow"]) & (df["MACD_Fast"] > df["MACD_Slow"])
    cross_down = (df["prev_fast"] >= df["prev_slow"]) & (df["MACD_Fast"] < df["MACD_Slow"])

    df["crossover"] = None
    df.loc[cross_up, "crossover"] = "UP"
    df.loc[cross_down, "crossover"] = "DOWN"

    crosses = df.dropna(subset=["crossover"])
    if crosses.empty:
        return None, None, None

    last_cross = crosses.iloc[-1]

    cross_date = last_cross["Date"]
    direction = last_cross["crossover"]

    latest_date = df.iloc[-1]["Date"]
    days_diff = (latest_date - cross_date).days

    if direction == "DOWN":
        days_diff = -abs(days_diff)
    else:
        days_diff = abs(days_diff)

    return cross_date, direction, days_diff


def process_macd_file(path: str, prefix: str, is_weekly: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    rows = []

    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Date")
        last_row = g.iloc[-1]

        cross_date, direction, days_from = find_latest_crossover(g)

        row = {
            "Symbol": symbol,
            "Sector": last_row["Sector"],
            "MarketCap": last_row["MarketCap"],
            f"{prefix}_Close": last_row["Close"],
            f"{prefix}_MACD_Fast": round(last_row["MACD_Fast"], 2),
            f"{prefix}_MACD_Slow": round(last_row["MACD_Slow"], 2),
            f"{prefix}_Crossover_Date": cross_date.strftime("%Y-%m-%d") if cross_date is not None else "",
            f"{prefix}_Crossover_Direction": direction if direction else "",
            f"{prefix}_Days_From_Crossover": days_from if days_from is not None else ""
        }

        # ---------------- NEW WEEKLY METRICS ----------------
        if is_weekly:
            high_close = g["Close"].max()
            latest_close = last_row["Close"]

            down_pct = ((latest_close - high_close) / high_close) * 100 if high_close else 0

            row["Weekly_High_Close"] = round(high_close, 2)
            row["Weekly_Down_Percent_From_High"] = round(down_pct, 2)
        # ----------------------------------------------------

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    daily_df = process_macd_file(DAILY_MACD_FILE, "Daily", is_weekly=False)
    weekly_df = process_macd_file(WEEKLY_MACD_FILE, "Weekly", is_weekly=True)

    final_df = daily_df.merge(
        weekly_df,
        on=["Symbol", "Sector", "MarketCap"],
        how="outer"
    )

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] MACD crossover file generated: {OUTPUT_FILE} ({len(final_df)} stocks)")
