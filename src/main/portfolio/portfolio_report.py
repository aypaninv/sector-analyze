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

    # Normalize column names (safety)
    df_stock.columns = [c.strip() for c in df_stock.columns]
    df_tech.columns = [c.strip() for c in df_tech.columns]

    if "Symbol" not in df_stock.columns:
        raise ValueError("list_stock file must contain 'Symbol' column")

    if "Symbol" not in df_tech.columns:
        raise ValueError("technical_data file must contain 'Symbol' column")

    # Merge (LEFT JOIN keeps all portfolio/watchlist rows)
    df_merged = pd.merge(
        df_stock,
        df_tech,
        on="Symbol",
        how="left"
    )

    # Optional: clean numeric columns
    numeric_cols = [
        "Qty", "AvgPrice", "StopLoss",
        "Close", "DD_High",
        "4H_MACD", "Day_MACD", "Week_MACD", "Month_MACD",
        "Day_RSI", "Week_RSI", "Month_RSI"
    ]

    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = (
                df_merged[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("", pd.NA)
            )
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df_merged.to_csv(output_file, index=False)
    print(f"[OK] Merged file created: {output_file} ({len(df_merged)} rows)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge portfolio/watchlist data with technical indicators"
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
