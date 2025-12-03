#!/usr/bin/env python3
"""
compute_indicators.py

Compute daily and weekly technical indicators per symbol and write a merged CSV
with D_ (daily) and W_ (weekly) prefixed columns, remove duplicated columns,
and round numeric values to 2 decimals.

Usage:
    python compute_indicators.py
    python compute_indicators.py --workers 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------- indicator helpers ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(df):
    prev_close = df['Close'].shift(1)
    t1 = df['High'] - df['Low']
    t2 = (df['High'] - prev_close).abs()
    t3 = (df['Low'] - prev_close).abs()
    tr = pd.concat([t1, t2, t3], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def directional_indicators(df, period=14):
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    plus_di.index = df.index
    minus_di.index = df.index
    adx.index = df.index
    return plus_di, minus_di, adx

# ---------------- per-symbol worker ----------------
def compute_for_symbol(args):
    sym, df_sym, ema_period, rsi_period, di_period, atr_period = args
    try:
        g = df_sym.sort_values('Date').reset_index(drop=True).copy()
        # ensure numeric
        for c in ['Open','High','Low','Close','Volume']:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors='coerce')
        # indicators
        g['EMA10'] = ema(g['Close'], ema_period)
        g['RSI'] = rsi(g['Close'], rsi_period)
        g['ATR'] = atr(g, atr_period)
        plus_di, minus_di, adx = directional_indicators(g, di_period)
        g['DI+'] = plus_di.values
        g['DI-'] = minus_di.values
        g['ADX'] = adx.values
        g['ATR_pct'] = g['ATR'] / (g['Close'].replace(0, pd.NA)) * 100

        last = g.iloc[-1]
        prev_close = g['Close'].iloc[-2] if len(g) >= 2 else pd.NA
        prev_high = g['High'].iloc[-2] if len(g) >= 2 else pd.NA

        out = {
            'Symbol': sym,
            'Sector': last.get('Sector', pd.NA),
            'MarketCap': last.get('MarketCap', pd.NA),
            'Date': last['Date'],
            'Close': last['Close'],
            'EMA10': last['EMA10'],
            'RSI': last['RSI'],
            'ATR': last['ATR'],
            'ATR_pct': last['ATR_pct'],
            'DI+': last['DI+'],
            'DI-': last['DI-'],
            'ADX': last['ADX'],
            'Prev_Close': prev_close,
            'Prev_High': prev_high
        }
    except Exception:
        out = {'Symbol': sym}
    return out

# ---------------- weekly resample ----------------
def make_weekly(df):
    rows = []
    for sym, g in df.groupby('Symbol'):
        gg = g.sort_values('Date').set_index('Date')
        w = gg.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Sector': 'last', 'MarketCap': 'last'
        }).dropna(subset=['Close'])
        if not w.empty:
            w = w.reset_index()
            w['Symbol'] = sym
            rows.append(w)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=df.columns)

# ---------------- main precompute flow ----------------
def main(raw_csv='output/yfinance_output.csv', out_csv='output/technical_output.csv', workers=4,
         ema_period=10, rsi_period=14, di_period=14, atr_period=14, use_parallel=True):

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} not found. Put your yfinance CSV at that path.")

    raw = pd.read_csv(raw_csv, parse_dates=['Date'])
    raw.columns = [c.strip() for c in raw.columns]

    # ensure numeric where appropriate
    for col in ['Open','High','Low','Close','Volume']:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    # daily compute
    symbols = sorted(raw['Symbol'].unique())
    tasks = [(sym, raw.loc[raw['Symbol'] == sym].copy(), ema_period, rsi_period, di_period, atr_period) for sym in symbols]

    results_daily = []
    if use_parallel and len(tasks) > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(compute_for_symbol, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                try:
                    results_daily.append(fut.result())
                except Exception:
                    sym = futures.get(fut, "unknown")
                    results_daily.append({'Symbol': sym})
    else:
        for t in tasks:
            results_daily.append(compute_for_symbol(t))

    df_daily = pd.DataFrame(results_daily).rename(columns={
        'Symbol': 'D_Symbol', 'Sector': 'D_Sector', 'MarketCap': 'D_MarketCap', 'Date': 'D_Date',
        'Close': 'D_Close', 'EMA10': 'D_EMA10', 'RSI': 'D_RSI', 'ATR': 'D_ATR', 'ATR_pct': 'D_ATR_pct',
        'DI+': 'D_DI+', 'DI-': 'D_DI-', 'ADX': 'D_ADX', 'Prev_Close': 'D_Prev_Close', 'Prev_High': 'D_Prev_High'
    })

    # weekly compute
    weekly_raw = make_weekly(raw)
    tasks_w = [(sym, weekly_raw.loc[weekly_raw['Symbol'] == sym].copy(), ema_period, rsi_period, di_period, atr_period) for sym in weekly_raw['Symbol'].unique()]

    results_weekly = []
    if use_parallel and len(tasks_w) > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(compute_for_symbol, t): t[0] for t in tasks_w}
            for fut in as_completed(futures):
                try:
                    results_weekly.append(fut.result())
                except Exception:
                    sym = futures.get(fut, "unknown")
                    results_weekly.append({'Symbol': sym})
    else:
        for t in tasks_w:
            results_weekly.append(compute_for_symbol(t))

    df_weekly = pd.DataFrame(results_weekly).rename(columns={
        'Symbol': 'W_Symbol', 'Sector': 'W_Sector', 'MarketCap': 'W_MarketCap', 'Date': 'W_Date',
        'Close': 'W_Close', 'EMA10': 'W_EMA10', 'RSI': 'W_RSI', 'ATR': 'W_ATR', 'ATR_pct': 'W_ATR_pct',
        'DI+': 'W_DI+', 'DI-': 'W_DI-', 'ADX': 'W_ADX', 'Prev_Close': 'W_Prev_Close', 'Prev_High': 'W_Prev_High'
    })

    # merge daily + weekly on symbol
    merged = pd.merge(df_daily, df_weekly, left_on='D_Symbol', right_on='W_Symbol', how='inner')

    # drop duplicated columns if any (keeps first occurrence)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # round numeric columns to 2 decimals
    num_cols = merged.select_dtypes(include=[float, int]).columns.tolist()
    if num_cols:
        merged[num_cols] = merged[num_cols].round(2)

    # ensure output dir
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    merged.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"Precomputed indicators saved to {out_csv} — rows: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute daily and weekly technical indicators and write precomputed CSV")
    parser.add_argument("--raw", default="output/yfinance_output.csv", help="raw yfinance CSV path")
    parser.add_argument("--out", default="output/technical_output.csv", help="output precomputed CSV path")
    parser.add_argument("--workers", type=int, default=4, help="number of parallel workers")
    parser.add_argument("--no-parallel", dest='use_parallel', action='store_false', help="disable parallel processing")
    args = parser.parse_args()
    main(raw_csv=args.raw, out_csv=args.out, workers=args.workers, use_parallel=args.use_parallel)
