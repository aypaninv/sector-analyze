import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
DAILY_FILE = "output/macd_daily_output.csv"
WEEKLY_FILE = "output/macd_weekly_output.csv"
# ---------------------------------------


@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_FILE, parse_dates=["Date"])
    weekly = pd.read_csv(WEEKLY_FILE, parse_dates=["Date"])
    return daily, weekly


def sector_average(df, sector):
    return (
        df[df["Sector"] == sector]
        .groupby("Date")[["MACD_Fast", "MACD_Slow"]]
        .mean()
        .reset_index()
        .sort_values("Date")
    )


def symbol_data(df, sector, symbol):
    return (
        df[(df["Sector"] == sector) & (df["Symbol"] == symbol)]
        .sort_values("Date")
    )


def latest_macd_above(df):
    """Return True if latest MACD_fast > MACD_slow"""
    if df.empty:
        return False
    last = df.sort_values("Date").iloc[-1]
    return last["MACD_Fast"] > last["MACD_Slow"]


def find_macd_crossovers(df):
    df = df.sort_values("Date").copy()
    df["prev_fast"] = df["MACD_Fast"].shift(1)
    df["prev_slow"] = df["MACD_Slow"].shift(1)

    cross_up = (df["prev_fast"] <= df["prev_slow"]) & (df["MACD_Fast"] > df["MACD_Slow"])
    cross_down = (df["prev_fast"] >= df["prev_slow"]) & (df["MACD_Fast"] < df["MACD_Slow"])

    return df[cross_up], df[cross_down]


def plot_macd(df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD_Fast"],
        mode="lines",
        name="MACD Fast",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MACD_Slow"],
        mode="lines",
        name="MACD Slow",
        line=dict(width=2)
    ))

    up, down = find_macd_crossovers(df)

    fig.add_trace(go.Scatter(
        x=up["Date"],
        y=up["MACD_Fast"],
        mode="markers",
        marker=dict(symbol="triangle-up", color="green", size=9),
        name="Cross Up"
    ))

    fig.add_trace(go.Scatter(
        x=down["Date"],
        y=down["MACD_Fast"],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=9),
        name="Cross Down"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="MACD",
        height=420,
        margin=dict(l=40, r=40, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def compute_symbol_drawdown(daily_df, weekly_df):
    weekly_high = weekly_df.groupby("Symbol")["Close"].max()
    latest_daily = (
        daily_df.sort_values("Date")
        .groupby("Symbol")
        .tail(1)
        .set_index("Symbol")["Close"]
    )

    dd = pd.concat([weekly_high, latest_daily], axis=1)
    dd.columns = ["Weekly_High", "Latest_Close"]
    dd = dd.dropna()
    dd = dd[dd["Weekly_High"] > 0]

    dd["DownPct"] = ((dd["Latest_Close"] - dd["Weekly_High"]) / dd["Weekly_High"]) * 100
    dd["DownPct"] = dd["DownPct"].replace([float("inf"), float("-inf")], 0)

    return dd["DownPct"].round(0).astype(int).to_dict()


def compute_sector_weekly_strength(weekly_df):
    """
    Returns list of (sector, weeks) sorted by weeks ASC
    """
    data = []
    for sector in sorted(weekly_df["Sector"].dropna().unique()):
        sec_df = sector_average(weekly_df, sector)
        if sec_df.empty:
            continue

        sec_df = sec_df.sort_values("Date", ascending=False)

        count = 0
        for _, row in sec_df.iterrows():
            if row["MACD_Fast"] > row["MACD_Slow"]:
                count += 1
            else:
                break

        if count > 0:
            data.append((sector, count))

    return sorted(data, key=lambda x: x[1])


# ---------------- STREAMLIT APP ----------------
st.set_page_config(layout="wide", page_title="Sector-wise MACD Analyzer")

daily_df, weekly_df = load_data()
drawdown_map = compute_symbol_drawdown(daily_df, weekly_df)

# -------- SESSION STATE --------
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = sorted(daily_df["Sector"].dropna().unique())[0]

# 🔝 COMPACT WEEKLY MACD STRENGTH (TOP)
sector_strength = compute_sector_weekly_strength(weekly_df)
if sector_strength:
    st.markdown("### 🔝 Weekly MACD Strength")

    chips_per_row = 8
    cols = st.columns(chips_per_row)
    col_idx = 0

    for sector, cnt in sector_strength:
        with cols[col_idx]:
            if st.button(f"{sector} ({cnt})", key=f"top_{sector}", use_container_width=True):
                st.session_state.selected_sector = sector

        col_idx += 1
        if col_idx >= chips_per_row:
            col_idx = 0
            cols = st.columns(chips_per_row)

# ----------- MAIN LAYOUT -----------
left, middle, right = st.columns([1.2, 4.5, 1.5])

# ================= LEFT PANEL (SECTOR) =================
with left:
    st.subheader("🔍 Sector")

    all_sectors = sorted(daily_df["Sector"].dropna().unique())

    sector_labels = []
    for s in all_sectors:
        bullish = latest_macd_above(sector_average(weekly_df, s))
        sector_labels.append(f"**{s}**" if bullish else s)

    selected_label = st.radio(
        "Sector",
        sector_labels,
        index=sector_labels.index(
            f"**{st.session_state.selected_sector}**"
            if latest_macd_above(sector_average(weekly_df, st.session_state.selected_sector))
            else st.session_state.selected_sector
        ),
        label_visibility="collapsed"
    )

    st.session_state.selected_sector = selected_label.replace("**", "")

# ================= RIGHT PANEL (SYMBOL) =================
with right:
    st.subheader("📌 Symbol")

    sector_symbols = (
        daily_df[daily_df["Sector"] == st.session_state.selected_sector]
        [["Symbol", "MarketCap"]]
        .drop_duplicates()
        .sort_values(["MarketCap", "Symbol"])
    )

    selected_symbol = "(Sector Average)"

    for cap, grp in sector_symbols.groupby("MarketCap"):
        st.markdown(f"**{cap}**")

        # ---- ORDER BY DRAWDOWN (LOW → HIGH) ----
        grp = grp.copy()
        grp["Drawdown"] = grp["Symbol"].map(lambda x: abs(drawdown_map.get(x, 0)))
        grp = grp.sort_values("Drawdown")

        labels = ["(Sector Average)"]

        for _, row in grp.iterrows():
            s = row["Symbol"]
            dd = row["Drawdown"]
            bullish = latest_macd_above(weekly_df[weekly_df["Symbol"] == s])
            labels.append(f"**{s} ({dd})**" if bullish else f"{s} ({dd})")

        choice = st.radio(
            cap,
            labels,
            key=f"sym_{cap}",
            label_visibility="collapsed"
        )

        if choice != "(Sector Average)":
            selected_symbol = choice.replace("**", "").split(" ")[0]

# ================= MIDDLE PANEL (CHARTS) =================
with middle:
    st.markdown(f"## 📊 MACD Analysis — **{st.session_state.selected_sector}**")

    if selected_symbol == "(Sector Average)":
        weekly_df_plot = sector_average(weekly_df, st.session_state.selected_sector)
        daily_df_plot = sector_average(daily_df, st.session_state.selected_sector)
        w_title = "Weekly MACD — Sector Average"
        d_title = "Daily MACD — Sector Average"
    else:
        weekly_df_plot = symbol_data(weekly_df, st.session_state.selected_sector, selected_symbol)
        daily_df_plot = symbol_data(daily_df, st.session_state.selected_sector, selected_symbol)
        w_title = f"Weekly MACD — {selected_symbol}"
        d_title = f"Daily MACD — {selected_symbol}"

    st.plotly_chart(plot_macd(weekly_df_plot, w_title), use_container_width=True)
    st.plotly_chart(plot_macd(daily_df_plot, d_title), use_container_width=True)
