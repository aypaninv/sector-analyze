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


@st.cache_data
def compute_sector_averages(df):
    return {
        s: (
            df[df["Sector"] == s]
            .groupby("Date")[["MACD_Fast", "MACD_Slow"]]
            .mean()
            .reset_index()
            .sort_values("Date")
        )
        for s in df["Sector"].dropna().unique()
    }


@st.cache_data
def split_by_symbol(df):
    return {k: v.sort_values("Date") for k, v in df.groupby("Symbol")}


@st.cache_data
def compute_symbol_drawdown(daily_df, weekly_df):
    weekly_high = weekly_df.groupby("Symbol")["Close"].max()
    latest_daily = (
        daily_df.sort_values("Date")
        .groupby("Symbol")
        .tail(1)
        .set_index("Symbol")["Close"]
    )

    dd = pd.concat([weekly_high, latest_daily], axis=1)
    dd.columns = ["High", "Last"]
    dd = dd.dropna()
    dd = dd[dd["High"] > 0]

    dd["Pct"] = ((dd["Last"] - dd["High"]) / dd["High"]) * 100
    return dd["Pct"].round(0).abs().astype(int).to_dict()


@st.cache_data
def compute_sector_weekly_strength(sector_avg_map):
    out = {}
    for sector, df in sector_avg_map.items():
        cnt = 0
        for _, r in df.sort_values("Date", ascending=False).iterrows():
            if r["MACD_Fast"] > r["MACD_Slow"]:
                cnt += 1
            else:
                break
        out[sector] = {"count": cnt, "bullish": cnt > 0}
    return out


@st.cache_data
def compute_symbol_weekly_strength(weekly_df):
    bullish_map = {}
    count_map = {}

    for sym, df in weekly_df.groupby("Symbol"):
        df = df.sort_values("Date", ascending=False)

        cnt = 0
        for _, r in df.iterrows():
            if r["MACD_Fast"] > r["MACD_Slow"]:
                cnt += 1
            else:
                break

        bullish_map[sym] = cnt > 0
        count_map[sym] = cnt

    return bullish_map, count_map


def plot_macd(df, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD_Fast"],
        mode="lines", name="MACD Fast", line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD_Slow"],
        mode="lines", name="MACD Slow", line=dict(width=2)
    ))

    df = df.sort_values("Date")
    prev_fast = df["MACD_Fast"].shift(1)
    prev_slow = df["MACD_Slow"].shift(1)

    cross_up = (prev_fast <= prev_slow) & (df["MACD_Fast"] > df["MACD_Slow"])
    cross_dn = (prev_fast >= prev_slow) & (df["MACD_Fast"] < df["MACD_Slow"])

    fig.add_trace(go.Scatter(
        x=df.loc[cross_up, "Date"],
        y=df.loc[cross_up, "MACD_Fast"],
        mode="markers", marker=dict(symbol="triangle-up", size=9, color="green"),
        name="Cross Up"
    ))
    fig.add_trace(go.Scatter(
        x=df.loc[cross_dn, "Date"],
        y=df.loc[cross_dn, "MACD_Fast"],
        mode="markers", marker=dict(symbol="triangle-down", size=9, color="red"),
        name="Cross Down"
    ))

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=40, r=40, t=45, b=40),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    return fig


# ================= APP =================
st.set_page_config(layout="wide", page_title="Sector-wise MACD Analyzer")

daily_df, weekly_df = load_data()

weekly_sector_avg = compute_sector_averages(weekly_df)
daily_sector_avg = compute_sector_averages(daily_df)
weekly_by_symbol = split_by_symbol(weekly_df)
daily_by_symbol = split_by_symbol(daily_df)

drawdown_map = compute_symbol_drawdown(daily_df, weekly_df)
sector_strength_map = compute_sector_weekly_strength(weekly_sector_avg)
symbol_bullish_map, symbol_week_count_map = compute_symbol_weekly_strength(weekly_df)

if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = sorted(weekly_sector_avg.keys())[0]

left, middle, right = st.columns([1.2, 4.5, 1.5])

# ================= LEFT PANEL (SECTOR) =================
with left:
    st.subheader("🔍 Sector")

    bullish = []
    non_bullish = []

    for sector, info in sector_strength_map.items():
        if info["bullish"]:
            bullish.append((sector, info["count"]))
        else:
            non_bullish.append(sector)

    bullish = sorted(bullish, key=lambda x: x[1])   # LOW → HIGH
    non_bullish = sorted(non_bullish)               # A → Z

    labels = []
    lookup = []

    for s, c in bullish:
        labels.append(f"**{s} ({c})**")
        lookup.append(s)

    for s in non_bullish:
        labels.append(s)
        lookup.append(s)

    idx = lookup.index(st.session_state.selected_sector)

    sel = st.radio("Sector", labels, index=idx, label_visibility="collapsed")
    st.session_state.selected_sector = sel.replace("**", "").rsplit(" (", 1)[0]

# ================= RIGHT PANEL (SYMBOL) =================
with right:
    st.subheader("📌 Symbol")

    df = daily_df[daily_df["Sector"] == st.session_state.selected_sector]
    groups = df[["Symbol", "MarketCap"]].drop_duplicates().groupby("MarketCap")

    selected_symbol = "(Sector Average)"

    for cap, grp in groups:
        st.markdown(f"**{cap}**")

        grp = grp.copy()
        grp["Drawdown"] = grp["Symbol"].map(lambda s: drawdown_map.get(s, 0))
        grp = grp.sort_values("Drawdown")

        labels = ["(Sector Average)"]

        for _, row in grp.iterrows():
            s = row["Symbol"]
            dd = row["Drawdown"]
            bullish = symbol_bullish_map.get(s, False)
            wk = symbol_week_count_map.get(s, 0)

            if bullish:
                lbl = f"**{s} ({dd}) ({wk})**"
            else:
                lbl = f"{s} ({dd})"

            labels.append(lbl)

        choice = st.radio(cap, labels, key=f"sym_{cap}", label_visibility="collapsed")

        if choice != "(Sector Average)":
            selected_symbol = choice.replace("**", "").split(" (")[0]

# ================= MIDDLE PANEL (CHARTS) =================
with middle:
    st.markdown(f"## 📊 MACD Analysis — **{st.session_state.selected_sector}**")

    if selected_symbol == "(Sector Average)":
        wdf = weekly_sector_avg[st.session_state.selected_sector]
        ddf = daily_sector_avg[st.session_state.selected_sector]
        wt, dt = "Weekly MACD — Sector Average", "Daily MACD — Sector Average"
    else:
        wdf = weekly_by_symbol[selected_symbol]
        ddf = daily_by_symbol[selected_symbol]
        wt, dt = f"Weekly MACD — {selected_symbol}", f"Daily MACD — {selected_symbol}"

    st.plotly_chart(plot_macd(wdf, wt), use_container_width=True)
    st.plotly_chart(plot_macd(ddf, dt), use_container_width=True)
