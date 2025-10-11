import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Page Setup ----------
st.set_page_config(
    page_title="NSE RSI Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("output/output.csv", parse_dates=['Date'])
    return df

df = load_data()

# ---------- Filter last 30 days ----------
thirty_days_ago = datetime.now() - timedelta(days=60)
df = df[df["Date"] >= thirty_days_ago]

# ---------- Layout Setup ----------
left_col, middle_col, right_col = st.columns([1.2, 2.5, 1.5])

# ---------- Left Panel: All Sectors ----------
with left_col:
    st.header("🏭 Sectors")
    st.markdown("Select one or more sectors:")

    sectors = sorted(df['Sector'].unique())
    selected_sectors = []
    for sector in sectors:
        if st.checkbox(sector, key=f"sector_{sector}"):
            selected_sectors.append(sector)

# ---------- Filter Data by Selected Sectors ----------
df_sector = df[df['Sector'].isin(selected_sectors)] if selected_sectors else pd.DataFrame()

# ---------- Right Panel: Symbols Filtered by Sector ----------
with right_col:
    st.header("💼 Symbols")
    selected_symbols = []

    if not df_sector.empty:
        market_caps = df_sector['Market Cap'].unique()
        for cap in market_caps:
            st.subheader(f"{cap}:")
            symbols = sorted(df_sector[df_sector['Market Cap'] == cap]['Symbol'].unique())
            for symbol in symbols:
                if st.checkbox(symbol, key=f"symbol_{symbol}"):
                    selected_symbols.append(symbol)
    else:
        st.info("Select a sector to view available symbols.")

# ---------- Middle Panel: Charts ----------
with middle_col:
    st.title("📊 NSE Stocks RSI Dashboard (Last 30 Days)")

    # Always show sector-level RSI average as soon as sector is selected
    if not df_sector.empty:
        df_sector_smooth = df_sector.copy()
        df_sector_smooth['dailyRSI_smooth'] = df_sector_smooth.groupby('Symbol')['dailyRSI'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df_sector_smooth['weeklyRSI_smooth'] = df_sector_smooth.groupby('Symbol')['weeklyRSI'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # ----- Chart 1: Average Sector RSI -----
        avg_rsi = (
            df_sector_smooth.groupby('Date')[['dailyRSI_smooth', 'weeklyRSI_smooth']].mean().reset_index()
        )
        fig_avg = px.line(
            avg_rsi, x='Date', y=['dailyRSI_smooth', 'weeklyRSI_smooth'],
            title=f"📅 Average RSI for Selected Sector(s)",
            labels={'value': 'Average RSI', 'Date': 'Date'},
        )
        fig_avg.update_traces(line_shape='spline')  # smooth curve
        fig_avg.update_layout(legend_title_text='RSI Type', height=450)
        st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.info("Select a sector to see the sector RSI trends.")

    # Show symbol-level RSI chart only when symbols are selected
    if selected_symbols:
        df_symbol = df_sector[df_sector['Symbol'].isin(selected_symbols)].copy()

        df_symbol['dailyRSI_smooth'] = df_symbol.groupby('Symbol')['dailyRSI'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df_symbol['weeklyRSI_smooth'] = df_symbol.groupby('Symbol')['weeklyRSI'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        st.markdown("---")

        fig_symbol = px.line(
            df_symbol, x='Date', y=['dailyRSI_smooth', 'weeklyRSI_smooth'],
            color='Symbol',
            title="📈 RSI Trend for Selected Stocks (Smoothed)",
            labels={'value': 'RSI', 'Date': 'Date'},
        )
        fig_symbol.update_traces(line_shape='spline')
        fig_symbol.update_layout(legend_title_text='Stock Symbol', height=450)
        st.plotly_chart(fig_symbol, use_container_width=True)
    elif not df_sector.empty:
        st.warning("Select one or more symbols to view individual stock RSI trends.")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Plotly • Showing last 30 days RSI trends")
