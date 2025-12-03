#!/usr/bin/env python3
"""
streamlit_app.py

Streamlit UI that reads the precomputed indicators CSV (cleaned with 2-decimal numbers)
and performs ranking (ATR excluded). Use:
    streamlit run streamlit_app.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Technical Ranking (precomputed)", layout="wide")

# ---------------- helpers ----------------
def find_col(df, preferred_prefix, base_name):
    cand = f"{preferred_prefix}_{base_name}"
    if cand in df.columns:
        return cand
    base_lower = base_name.lower()
    for c in df.columns:
        cl = c.lower()
        if cl == base_lower or cl.endswith('_' + base_lower) or cl.endswith(base_lower):
            return c
    return None

def ensure_columns_for_ranking(merged, raw):
    # ensure Symbol column exists
    if 'D_Symbol' in merged.columns:
        merged['Symbol'] = merged['D_Symbol']
    elif 'W_Symbol' in merged.columns:
        merged['Symbol'] = merged['W_Symbol']
    elif 'Symbol' in merged.columns:
        merged['Symbol'] = merged['Symbol']
    else:
        raise KeyError("No Symbol column found in merged precomputed CSV.")

    # Sector
    d_sector = find_col(merged, 'D', 'Sector')
    w_sector = find_col(merged, 'W', 'Sector')
    if d_sector:
        merged['Sector'] = merged[d_sector].fillna(merged[w_sector] if w_sector in merged.columns else np.nan)
    elif w_sector:
        merged['Sector'] = merged[w_sector]
    else:
        merged['Sector'] = merged['Symbol'].map(raw.groupby('Symbol')['Sector'].last().to_dict())

    # MarketCap
    d_mcap = find_col(merged, 'D', 'MarketCap')
    w_mcap = find_col(merged, 'W', 'MarketCap')
    if d_mcap:
        merged['MarketCap'] = merged[d_mcap].fillna(merged[w_mcap] if w_mcap in merged.columns else np.nan)
    elif w_mcap:
        merged['MarketCap'] = merged[w_mcap]
    else:
        merged['MarketCap'] = merged['Symbol'].map(raw.groupby('Symbol')['MarketCap'].last().to_dict())

    # Ensure key prefixed numeric columns exist (fallback to NaN when missing)
    keys = ['Close','EMA10','RSI','ADX','DI+','DI-','ATR_pct','Prev_Close','Prev_High']
    for p in ('D','W'):
        for k in keys:
            col = f"{p}_{k}"
            if col not in merged.columns:
                merged[col] = np.nan

    # convert numeric columns to numeric dtype (if strings due to CSV)
    for col in merged.columns:
        if merged[col].dtype == object:
            # try numeric conversion for columns that look numeric
            try:
                merged[col] = pd.to_numeric(merged[col], errors='ignore')
            except Exception:
                pass

    return merged

def compute_score_no_atr(df, weights=None):
    if weights is None:
        weights = {'RSI': 0.35, 'ADX': 0.35, 'DI_gap': 0.2, 'EMA_dist': 0.1}
    rsi_norm = (df['RSI'] - 55) / (80 - 55)
    rsi_norm = rsi_norm.clip(0,1)
    adx_norm = (df['ADX'] - 20) / (50 - 20)
    adx_norm = adx_norm.clip(0,1)
    di_gap = (df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-'] + 1e-12)
    di_gap_norm = (di_gap + 1) / 2
    ema_dist = (df['Close'] - df['EMA10']) / (df['EMA10'] + 1e-12)
    ema_dist_norm = (ema_dist - 0) / (0.10 - 0)
    ema_dist_norm = ema_dist_norm.clip(0,1)
    score = (weights['RSI'] * rsi_norm.fillna(0) +
             weights['ADX'] * adx_norm.fillna(0) +
             weights['DI_gap'] * di_gap_norm.fillna(0) +
             weights['EMA_dist'] * ema_dist_norm.fillna(0))
    return score

# ---------------- UI ----------------
st.title("Technical Ranking (Precomputed Indicators)")

# Sidebar
precomputed_path = st.sidebar.text_input("Precomputed indicators CSV", value="output/technical_output.csv")
raw_path = st.sidebar.text_input("Raw yfinance CSV (for fallback sector/mcap)", value="output/yfinance_output.csv")

if not os.path.exists(precomputed_path):
    st.error(f"Precomputed file not found: {precomputed_path}")
    st.stop()

# load merged precomputed and raw
merged = pd.read_csv(precomputed_path, parse_dates=True)
raw = pd.read_csv(raw_path, parse_dates=['Date']) if os.path.exists(raw_path) else pd.DataFrame()

# normalize and ensure columns
merged = ensure_columns_for_ranking(merged, raw)

# Sidebar filters
st.sidebar.markdown("### Filters")
filter_price_above_ema = st.sidebar.checkbox("Require Price > EMA10 (daily)", value=True)
filter_rsi_min = st.sidebar.slider("Min Daily RSI", 40, 90, 55)
filter_adx_min = st.sidebar.slider("Min Daily ADX", 0, 60, 20)
filter_di_plus = st.sidebar.checkbox("Require DI+ > DI- (daily)", value=True)
require_weekly_confirm = st.sidebar.checkbox("Require Weekly confirmation", value=True)

# Momentum higher checks
st.sidebar.markdown("### Momentum checks")
require_daily_higher = st.sidebar.checkbox("Require daily Close > previous day Close", value=True)
require_weekly_higher = st.sidebar.checkbox("Require weekly Close > previous week Close", value=True)

# Ranking weights (ATR excluded)
st.sidebar.markdown("### Ranking weights")
w_rsi = st.sidebar.slider("RSI weight", 0.0, 1.0, 0.35, 0.05)
w_adx = st.sidebar.slider("ADX weight", 0.0, 1.0, 0.35, 0.05)
w_di = st.sidebar.slider("DI gap weight", 0.0, 1.0, 0.2, 0.05)
w_ema = st.sidebar.slider("EMA distance weight", 0.0, 1.0, 0.1, 0.05)
total = w_rsi + w_adx + w_di + w_ema
if total == 0:
    total = 1.0
weights = {'RSI': w_rsi/total, 'ADX': w_adx/total, 'DI_gap': w_di/total, 'EMA_dist': w_ema/total}

# Apply filters function
def apply_filters(df):
    cond = pd.Series(True, index=df.index)
    if filter_price_above_ema:
        cond &= (df['D_Close'] > df['D_EMA10'])
        if require_weekly_confirm:
            cond &= (df['W_Close'] > df['W_EMA10'])
    if filter_rsi_min:
        cond &= (df['D_RSI'] > filter_rsi_min)
        if require_weekly_confirm:
            cond &= (df['W_RSI'] > filter_rsi_min)
    if filter_adx_min:
        cond &= (df['D_ADX'] > filter_adx_min)
        if require_weekly_confirm:
            cond &= (df['W_ADX'] > filter_adx_min)
    if filter_di_plus:
        cond &= (df['D_DI+'] > df['D_DI-'])
        if require_weekly_confirm:
            cond &= (df['W_DI+'] > df['W_DI-'])
    if require_daily_higher:
        cond &= (df['D_Close'] > df['D_Prev_Close'])
    if require_weekly_higher:
        cond &= (df['W_Close'] > df['W_Prev_Close'])
    return df[cond]

filtered = apply_filters(merged)

if filtered.empty:
    st.warning("No symbols passed filters. Try lowering thresholds or disabling strict 'higher' requirements.")
else:
    score_input = pd.DataFrame({
        'RSI': filtered['D_RSI'],
        'ADX': filtered['D_ADX'],
        'DI+': filtered['D_DI+'],
        'DI-': filtered['D_DI-'],
        'Close': filtered['D_Close'],
        'EMA10': filtered['D_EMA10']
    })
    filtered['Score'] = compute_score_no_atr(score_input, weights)
    filtered = filtered.sort_values('Score', ascending=False).reset_index(drop=True)
    filtered['Rank'] = filtered.index + 1

    # format numeric columns to 2 decimals for display
    display_cols = [
        'Rank', 'Symbol', 'Sector', 'D_Close', 'D_EMA10', 'D_RSI', 'D_ADX', 'D_DI+', 'D_DI-', 'D_ATR_pct',
        'W_Close', 'W_EMA10', 'W_RSI', 'W_ADX', 'Score'
    ]
    st.subheader(f"Filtered symbols: {len(filtered)} — Top picks")
    col1, col2 = st.columns([2,1])
    top_n = col2.number_input("Show top N", min_value=1, max_value=500, value=50)
    top_df = filtered.head(top_n)

    sort_by = st.selectbox("Sort by column", options=display_cols, index=display_cols.index('Score'))
    ascending = st.checkbox("Ascending", value=False)
    table_df = top_df[display_cols].sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # Use pandas formatting for 2 decimals in displayed table
    st.dataframe(table_df.style.format({
        'D_Close': '{:,.2f}', 'D_EMA10': '{:,.2f}', 'D_RSI': '{:.2f}', 'D_ADX': '{:.2f}', 'D_DI+': '{:.2f}', 'D_DI-': '{:.2f}', 'D_ATR_pct': '{:.2f}',
        'W_Close': '{:,.2f}', 'W_EMA10': '{:,.2f}', 'W_RSI': '{:.2f}', 'W_ADX': '{:.2f}', 'Score': '{:.4f}'
    }))

    # Plots
    st.markdown("### Ranking plot (Top N)")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.bar(top_df['Symbol'], top_df['Score'])
    ax1.set_ylabel("Score"); ax1.set_title("Composite Score (Top N)"); ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.markdown("### RSI vs ADX (Top N)")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.scatter(top_df['D_RSI'], top_df['D_ADX'])
    for i, s in enumerate(top_df['Symbol']):
        ax2.annotate(s, (top_df['D_RSI'].iat[i], top_df['D_ADX'].iat[i]), fontsize=8)
    ax2.set_xlabel("Daily RSI"); ax2.set_ylabel("Daily ADX"); ax2.set_title("RSI vs ADX (Top N)")
    st.pyplot(fig2)

    csv = top_df.to_csv(index=False, float_format="%.2f")
    st.download_button("Download top table as CSV", csv, file_name="tech_ranking_top.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: Precomputed CSV has numeric indicators rounded to 2 decimals. ATR is shown in the table but excluded from scoring.")
