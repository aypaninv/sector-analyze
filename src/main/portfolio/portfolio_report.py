import pandas as pd
import argparse
import os


def merge_files(stock_file: str, technical_file: str, output_file: str):
    df_stock = pd.read_csv(stock_file)
    df_tech = pd.read_csv(technical_file)

    df_stock.columns = [c.strip() for c in df_stock.columns]
    df_tech.columns = [c.strip() for c in df_tech.columns]

    if "Symbol" not in df_stock.columns or "Symbol" not in df_tech.columns:
        raise ValueError("Both input files must contain 'Symbol' column")

    df = pd.merge(df_stock, df_tech, on="Symbol", how="left")

    # -----------------------------
    # Numeric cleanup
    # -----------------------------
    numeric_cols = [
        "Qty", "AvgPrice", "StopLoss",
        "LastClose", "Close", "DD_High",
        "4H_MACD", "Day_MACD", "Week_MACD", "Month_MACD",
        "Day_RSI", "Week_RSI", "Month_RSI"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "LastClose" not in df.columns and "Close" in df.columns:
        df.rename(columns={"Close": "LastClose"}, inplace=True)

    # -----------------------------
    # Invested Capital (IC)
    # -----------------------------
    if {"Qty", "AvgPrice"}.issubset(df.columns):
        df["InvestedValue"] = df["Qty"] * df["AvgPrice"]
        total_invested = df["InvestedValue"].sum(skipna=True)
        df["IC"] = (df["InvestedValue"] / total_invested * 100).round(1) if total_invested > 0 else pd.NA
    else:
        df["IC"] = pd.NA

    # -----------------------------
    # Current Capital (CC)
    # -----------------------------
    if {"Qty", "LastClose"}.issubset(df.columns):
        df["CurrentValue"] = df["Qty"] * df["LastClose"]
        total_current = df["CurrentValue"].sum(skipna=True)
        df["CC"] = (df["CurrentValue"] / total_current * 100).round(1) if total_current > 0 else pd.NA
    else:
        df["CC"] = pd.NA

    # -----------------------------
    # RSI formatting (0 decimals)
    # -----------------------------
    for rsi in ["Day_RSI", "Week_RSI", "Month_RSI"]:
        if rsi in df.columns:
            df[rsi] = df[rsi].round(0)

    # -----------------------------
    # Rename headers
    # -----------------------------
    df.rename(columns={
        "Day_MACD": "D_MACD",
        "Week_MACD": "W_MACD",
        "Month_MACD": "M_MACD",
        "Day_RSI": "D_RSI",
        "Week_RSI": "W_RSI",
        "Month_RSI": "M_RSI",
    }, inplace=True)

    # Drop helpers
    df.drop(columns=[c for c in ["InvestedValue", "CurrentValue"] if c in df.columns], inplace=True)

    # -----------------------------
    # Column order
    # -----------------------------
    desired_order = [
        "Symbol", "Folio", "IC", "CC",
        "AvgPrice", "LastClose", "StopLoss",
        "MarketCap", "Sector", "Notes",
        "DD_High", "4H_MACD",
        "D_MACD", "W_MACD", "M_MACD",
        "D_RSI", "W_RSI", "M_RSI"
    ]

    ordered = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[OK] Portfolio report generated: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-file", required=True)
    parser.add_argument("--technical-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merge_files(args.stock_file, args.technical_file, args.output)
