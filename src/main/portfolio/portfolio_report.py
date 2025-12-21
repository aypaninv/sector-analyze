import pandas as pd
import argparse
import os


def merge_files(
    stock_file: str,
    technical_file: str,
    output_file: str
):
    # Read input files
    df_stock = pd.read_csv(stock_file)
    df_tech = pd.read_csv(technical_file)

    # Normalize column names
    df_stock.columns = [c.strip() for c in df_stock.columns]
    df_tech.columns = [c.strip() for c in df_tech.columns]

    if "Symbol" not in df_stock.columns:
        raise ValueError("list_stock file must contain 'Symbol' column")

    if "Symbol" not in df_tech.columns:
        raise ValueError("technical_data file must contain 'Symbol' column")

    # Merge (LEFT JOIN)
    df = pd.merge(
        df_stock,
        df_tech,
        on="Symbol",
        how="left"
    )

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

    # Ensure LastClose exists
    if "LastClose" not in df.columns and "Close" in df.columns:
        df.rename(columns={"Close": "LastClose"}, inplace=True)

    # -----------------------------
    # Invested Capital (IC)
    # -----------------------------
    if {"Qty", "AvgPrice"}.issubset(df.columns):
        df["InvestedValue"] = df["Qty"] * df["AvgPrice"]
        total_invested = df["InvestedValue"].sum(skipna=True)

        if total_invested > 0:
            df["IC"] = (df["InvestedValue"] / total_invested * 100).round(1)
        else:
            df["IC"] = pd.NA
    else:
        df["IC"] = pd.NA

    # -----------------------------
    # Current Capital (CC)  ✅ NEW
    # -----------------------------
    if {"Qty", "LastClose"}.issubset(df.columns):
        df["CurrentValue"] = df["Qty"] * df["LastClose"]
        total_current = df["CurrentValue"].sum(skipna=True)

        if total_current > 0:
            df["CC"] = (df["CurrentValue"] / total_current * 100).round(1)
        else:
            df["CC"] = pd.NA
    else:
        df["CC"] = pd.NA

    # Drop helper columns
    df.drop(columns=[c for c in ["InvestedValue", "CurrentValue"] if c in df.columns],
            inplace=True)

    # -----------------------------
    # Column order
    # -----------------------------
    desired_order = [
        "Symbol",
        "Folio",
        "IC",
        "CC",
        "AvgPrice",
        "LastClose",
        "StopLoss",
        "MarketCap",
        "Sector",
        "Notes",
        "DD_High",
        "4H_MACD",
        "Day_MACD",
        "Week_MACD",
        "Month_MACD",
        "Day_RSI",
        "Week_RSI",
        "Month_RSI",
    ]

    existing_cols = list(df.columns)
    ordered_cols = [c for c in desired_order if c in existing_cols]
    remaining_cols = [c for c in existing_cols if c not in ordered_cols]

    df = df[ordered_cols + remaining_cols]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"[OK] Portfolio report generated: {output_file} ({len(df)} rows)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create portfolio report with Invested & Current Capital distribution"
    )

    parser.add_argument(
        "--stock-file",
        required=True,
        help="Input portfolio/watchlist CSV file"
    )

    parser.add_argument(
        "--technical-file",
        required=True,
        help="Input technical data CSV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output merged CSV file"
    )

    args = parser.parse_args()

    merge_files(
        stock_file=args.stock_file,
        technical_file=args.technical_file,
        output_file=args.output
    )
