# app.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Technical Ranking — Weekly Only", layout="wide")

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

def ensure_weekly_columns(merged, raw_df):
    """
    Ensure weekly prefixed columns exist so rest of app can use W_Close, W_EMA10, W_RSI, W_ADX, W_DI+, W_DI-, W_ATR_pct, W_Prev_Close
    Also ensure Symbol, Sector and MarketCap columns exist.
    """
    # Symbol
    if 'W_Symbol' in merged.columns:
        merged['Symbol'] = merged['W_Symbol']
    elif 'D_Symbol' in merged.columns:
        merged['Symbol'] = merged['D_Symbol']
    elif 'Symbol' in merged.columns:
        merged['Symbol'] = merged['Symbol']
    else:
        raise KeyError("No Symbol column found. Check precomputed CSV headers.")

    # Sector (prefer W_, fallback to D_ or raw)
    w_sector = find_col(merged, 'W', 'Sector')
    d_sector = find_col(merged, 'D', 'Sector')
    if w_sector:
        merged['Sector'] = merged[w_sector].fillna(merged[d_sector] if d_sector in merged.columns else np.nan)
    elif d_sector:
        merged['Sector'] = merged[d_sector]
    else:
        merged['Sector'] = merged['Symbol'].map(raw_df.groupby('Symbol')['Sector'].last().to_dict() if not raw_df.empty else {})

    # MarketCap
    w_mcap = find_col(merged, 'W', 'MarketCap')
    d_mcap = find_col(merged, 'D', 'MarketCap')
    if w_mcap:
        merged['MarketCap'] = merged[w_mcap].fillna(merged[d_mcap] if d_mcap in merged.columns else np.nan)
    elif d_mcap:
        merged['MarketCap'] = merged[d_mcap]
    else:
        merged['MarketCap'] = merged['Symbol'].map(raw_df.groupby('Symbol')['MarketCap'].last().to_dict() if not raw_df.empty else {})

    # Ensure weekly numeric columns exist (fallback to NaN)
    required_weekly = ['Close','EMA10','RSI','ADX','DI+','DI-','ATR_pct','Prev_Close','Prev_High']
    for k in required_weekly:
        col = f"W_{k}"
        if col not in merged.columns:
            merged[col] = np.nan

    # Convert numeric-looking columns to numeric dtype
    for col in merged.columns:
        if merged[col].dtype == object:
            # try to coerce numeric columns (safe - errors='ignore' will keep non-numeric unchanged)
            merged[col] = pd.to_numeric(merged[col], errors='ignore')

    return merged

def compute_weekly_score(df_weekly, weights=None):
    """
    df_weekly must contain columns:
      'W_RSI','W_ADX','W_DI+','W_DI-','W_Close','W_EMA10'
    Returns a pd.Series of scores.
    """
    if weights is None:
        weights = {'RSI': 0.35, 'ADX': 0.35, 'DI_gap': 0.2, 'EMA_dist': 0.1}

    # Prepare normalized components
    rsi = df_weekly['W_RSI'].astype(float)
    adx = df_weekly['W_ADX'].astype(float)
    di_plus = df_weekly['W_DI+'].astype(float)
    di_minus = df_weekly['W_DI-'].astype(float)
    close = df_weekly['W_Close'].astype(float)
    ema10 = df_weekly['W_EMA10'].astype(float)

    rsi_norm = (rsi - 55) / (80 - 55)
    rsi_norm = rsi_norm.clip(0,1).fillna(0)

    adx_norm = (adx - 20) / (50 - 20)
    adx_norm = adx_norm.clip(0,1).fillna(0)

    di_gap = (di_plus - di_minus) / (di_plus + di_minus + 1e-12)
    di_gap_norm = ((di_gap + 1) / 2).clip(0,1).fillna(0)

    ema_dist = (close - ema10) / (ema10 + 1e-12)
    ema_dist_norm = ((ema_dist - 0) / (0.10 - 0)).clip(0,1).fillna(0)

    score = (weights['RSI'] * rsi_norm +
             weights['ADX'] * adx_norm +
             weights['DI_gap'] * di_gap_norm +
             weights['EMA_dist'] * ema_dist_norm)
    return score

# ---------------- UI ----------------
st.title("Technical Ranking — Weekly Only")

# Sidebar minimal controls
st.sidebar.markdown("### Data & options")
precomputed_path = st.sidebar.text_input("Precomputed CSV path", value="output/technical_output.csv")
raw_path = st.sidebar.text_input("Raw CSV (optional, for sector fallback)", value="output/yfinance_output.csv")

# Sector filter (multiselect, max 5 selections)
# We'll load file to get sector list
if not os.path.exists(precomputed_path):
    st.error(f"Precomputed file not found: {precomputed_path}")
    st.stop()

merged = pd.read_csv(precomputed_path, parse_dates=True)
raw_df = pd.read_csv(raw_path, parse_dates=['Date']) if os.path.exists(raw_path) else pd.DataFrame()

# ensure weekly columns & helper columns
merged = ensure_weekly_columns(merged, raw_df)

# Build sector list and offer multiselect limited to 5 choices
all_sectors = sorted(pd.Series(merged['Sector'].dropna().unique()).astype(str).tolist())
selected_sectors = st.sidebar.multiselect("Filter by sector (select up to 5)", options=all_sectors, default=[])

# enforce selection limit of 5
if len(selected_sectors) > 5:
    st.sidebar.warning("You selected more than 5 sectors — trimming to first 5 selections.")
    selected_sectors = selected_sectors[:5]

# Ranking weights (weekly)
st.sidebar.markdown("### Ranking weights (weekly)")
w_rsi = st.sidebar.slider("RSI weight", 0.0, 1.0, 0.35, 0.05)
w_adx = st.sidebar.slider("ADX weight", 0.0, 1.0, 0.35, 0.05)
w_di = st.sidebar.slider("DI gap weight", 0.0, 1.0, 0.2, 0.05)
w_ema = st.sidebar.slider("EMA distance weight", 0.0, 1.0, 0.1, 0.05)
total = w_rsi + w_adx + w_di + w_ema
if total == 0:
    total = 1.0
weights = {'RSI': w_rsi/total, 'ADX': w_adx/total, 'DI_gap': w_di/total, 'EMA_dist': w_ema/total}

# compute weekly score (only weekly columns)
with st.spinner("Computing weekly scores..."):
    merged['Score'] = compute_weekly_score(merged, weights)
    # sort descending (best first)
    merged = merged.sort_values('Score', ascending=False).reset_index(drop=True)
    merged['Rank'] = merged.index + 1

# Display logic:
# - If no sector selected: show default top 50
# - If sector(s) selected: show all stocks in those sectors sorted by Rank
default_top_n = 50
if len(selected_sectors) == 0:
    display_df = merged.head(default_top_n).copy()
    st.subheader(f"Top {default_top_n} weekly-ranked stocks")
else:
    display_df = merged[merged['Sector'].isin(selected_sectors)].sort_values('Score', ascending=False).reset_index(drop=True)
    display_df['Rank'] = display_df.index + 1
    st.subheader(f"All stocks in selected sector(s) — {len(selected_sectors)} sector(s), {len(display_df)} symbols")

# Columns to display: weekly technicals + ATR pct (for info)
display_cols = [
    'Rank', 'Symbol', 'Sector',
    'W_Close', 'W_EMA10', 'W_RSI', 'W_ADX', 'W_DI+', 'W_DI-', 'W_ATR_pct', 'Score'
]

# Ensure columns exist in dataset, otherwise fill NaN
for c in display_cols:
    if c not in display_df.columns:
        display_df[c] = np.nan

# Round numeric display columns to 2 decimals for UI
numeric_display = ['W_Close', 'W_EMA10', 'W_RSI', 'W_ADX', 'W_DI+', 'W_DI-', 'W_ATR_pct', 'Score']
for col in numeric_display:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)

# Show table
st.dataframe(display_df[display_cols].style.format({
    'W_Close': '{:,.2f}', 'W_EMA10': '{:,.2f}', 'W_RSI': '{:.2f}', 'W_ADX': '{:.2f}', 'W_DI+': '{:.2f}', 'W_DI-': '{:.2f}', 'W_ATR_pct': '{:.2f}', 'Score': '{:.4f}'
}), use_container_width=True)

# Quick plot of top items (if no sector selected, top 50; if sector selected, top 25 from the result)
plot_n = 25 if len(selected_sectors) > 0 else min(default_top_n, 50)
plot_df = display_df.head(plot_n)

st.markdown("### Weekly ranking bar chart")
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(plot_df['Symbol'].astype(str), plot_df['Score'].astype(float))
ax.set_ylabel("Weekly composite Score")
ax.set_title("Weekly composite score — top symbols")
ax.tick_params(axis='x', rotation=60)
st.pyplot(fig)

# Download CSV of displayed data
csv = display_df[display_cols].to_csv(index=False, float_format="%.2f")
st.download_button("Download displayed list (CSV)", csv, file_name="weekly_ranking_display.csv", mime="text/csv")

st.markdown("---")
st.caption("Ranking uses weekly indicators only (W_RSI, W_ADX, W_DI+, W_DI-, W_EMA10, W_Close). Select up to 5 sectors to view all symbols in those sectors; otherwise default shows top 50 weekly-ranked stocks.")
