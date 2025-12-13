import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
CSV_FILE   = "output/stocks_monthly.csv"           # input file
OUTPUT_DIR = "output"
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plot")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "angles_summary.csv")

THRESHOLD_DEG          = 39.0   # only stocks above this angle get plots
RSI_PERIOD             = 14     # RSI period on monthly closes
DOWN_FROM_ATH_MAX_PCT  = 20.0   # only stocks <= 30% below ATH qualify for plots


# ==========================
# HELPERS
# ==========================
def compute_visual_angle(group: pd.DataFrame):
    """
    Angle in 'chart-like' coords:
      x = time index (0..N-1), y = Close
      both axes min–max normalised to 0..1
    Returns: angle_deg, (m, c), (x, y)
    """
    group = group.sort_values("Date")
    n = len(group)
    if n < 3:
        return None

    x = np.arange(n, dtype=float)
    y = group["Close"].astype(float).values

    # Fit line y = m*x + c
    m, c = np.polyfit(x, y, 1)

    # Endpoints in data space
    x0, x1 = x[0], x[-1]
    y0, y1 = m * x0 + c, m * x1 + c

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if x_max == x_min or y_max == y_min:
        return None

    # Normalise to 0..1
    x0n = (x0 - x_min) / (x_max - x_min)
    x1n = (x1 - x_min) / (x_max - x_min)
    y0n = (y0 - y_min) / (y_max - y_min)
    y1n = (y1 - y_min) / (y_max - y_min)

    slope_norm = (y1n - y0n) / (x1n - x0n)
    angle_rad  = math.atan(slope_norm)
    angle_deg  = math.degrees(angle_rad)

    return angle_deg, (m, c), (x, y)


def compute_rsi_latest(prices: np.ndarray, period: int = 14) -> float | None:
    """
    Wilder-style RSI; returns ONLY the latest RSI value.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period + 1:
        return None

    delta  = np.diff(prices)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs  = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for the rest
    for i in range(period, len(delta)):
        gain = gains[i]
        loss = losses[i]

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs  = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)

    return float(rsi)


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


# ==========================
# MAIN
# ==========================
def main():
    ensure_dirs()

    df = pd.read_csv(CSV_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])

    records = []

    for symbol, group in df.groupby("Symbol"):
        res = compute_visual_angle(group)
        if res is None:
            continue

        angle_deg, (m, c), (x, y) = res

        closes     = group["Close"].astype(float).values
        ath_close  = closes.max()
        last_close = closes[-1]

        # latest monthly RSI
        latest_rsi = compute_rsi_latest(closes, period=RSI_PERIOD)

        # Down from ATH (%)
        if ath_close != 0:
            down_from_ath_pct = (ath_close - last_close) / ath_close * 100.0
        else:
            down_from_ath_pct = 0.0

        # Sector & MarketCap
        if "Sector" in group.columns:
            sector = str(group["Sector"].iloc[0])
        else:
            sector = ""

        if "MarketCap" in group.columns:
            marketcap = str(group["MarketCap"].iloc[0])
        else:
            marketcap = ""

        records.append(
            {
                "Symbol":          symbol,
                "MarketCap":       marketcap,
                "Sector":          sector,
                "VisualAngleDeg":  round(angle_deg, 2),
                "ATHClose":        round(ath_close, 2),
                "LastClose":       round(last_close, 2),
                "DownFromATHPct":  round(down_from_ath_pct, 2),
                "RSI_Monthly":     round(latest_rsi, 2) if latest_rsi is not None else None,
            }
        )

        # ------- Plot only if above angle threshold AND within ATH drawdown threshold -------
        if angle_deg >= THRESHOLD_DEG and down_from_ath_pct <= DOWN_FROM_ATH_MAX_PCT:
            dates = group["Date"].values
            y_fit = m * x + c

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(dates, y, marker="o", label="Monthly Close")
            ax.plot(dates, y_fit, linestyle="--", label="Trend Line (fit)")
            ax.axhline(ath_close, linestyle=":", label=f"ATH Close {ath_close:.2f}")

            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

            rsi_text = f"{latest_rsi:.2f}" if latest_rsi is not None else "NA"
            down_text = f"{down_from_ath_pct:.2f}%"

            title = (
                f"{symbol} - {marketcap} - {sector} - "
                f"{angle_deg:.2f}° - RSI: {rsi_text} - Down from ATH: {down_text}"
            )
            ax.set_title(title)

            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            plot_path = os.path.join(PLOT_DIR, f"{symbol}.png")
            fig.savefig(plot_path)
            plt.close(fig)

    # ------- Save CSV summary -------
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved summary CSV to: {OUTPUT_CSV}")
    print(
        f"Plots (angle >= {THRESHOLD_DEG}° and down_from_ath <= {DOWN_FROM_ATH_MAX_PCT}%) "
        f"saved to: {PLOT_DIR}"
    )


if __name__ == "__main__":
    main()
