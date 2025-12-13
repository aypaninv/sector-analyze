import pandas as pd
import dash
from dash import html, dash_table, Input, Output, State

# === Configuration ===
file_path = "output/adx_output.csv"  # update if needed
LAST_N_DAYS = 20

# === Load Data ===
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])

# Keep only last N unique dates (latest → oldest)
latest_dates = sorted(df["Date"].unique())[-LAST_N_DAYS:]
df = df[df["Date"].isin(latest_dates)]

# === Sector-level average (di_plus_10) ===
sector_avg = df.groupby(["Sector", "Date"])["di_plus_10"].mean().reset_index()
pivot_sector = sector_avg.pivot(index="Sector", columns="Date", values="di_plus_10")

# Sort columns (newest → oldest)
pivot_sector = pivot_sector[sorted(pivot_sector.columns, reverse=True)]

# Rename headers as day-month only (e.g., 01-Apr)
pivot_sector = pivot_sector.rename(columns={d: d.strftime("%d-%b") for d in pivot_sector.columns})

# Round for cleaner view
pivot_sector = pivot_sector.round(2)

# Convert all date columns to numeric (important for coloring)
sector_display = pivot_sector.reset_index()
for c in sector_display.columns:
    if c != "Sector":
        sector_display[c] = pd.to_numeric(sector_display[c], errors="coerce")

# === Color mapping logic (user's final thresholds) ===
def get_color_for_value(val):
    # val is expected to be a float or None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "white"
    try:
        v = float(val)
    except Exception:
        return "white"
    if v < 16:
        return "#ffb3b3"      # mild red
    elif 16 <= v < 20:
        return "#ccffcc"      # light green
    elif 20 <= v < 25:
        return "#99ff99"      # mid green
    elif 25 <= v < 30:
        return "#66ff66"      # green
    elif 30 <= v < 40:
        return "#33cc33"      # dark green
    elif v >= 40:
        return "#66b3ff"      # blue
    return "white"

def color_scale(df, col_list=None):
    """Apply color map to DataTable based on DI+ range. Handles string/NaN values robustly."""
    if col_list is None:
        col_list = [c for c in df.columns if c not in ["Sector", "Market Cap", "Symbol"]]
    styles = []
    for col in col_list:
        for i, raw_val in enumerate(df[col]):
            color = get_color_for_value(raw_val)
            styles.append({
                "if": {"row_index": i, "column_id": col},
                "backgroundColor": color,
                "color": "black"
            })
    return styles

# === Dash App ===
app = dash.Dash(__name__)
app.title = "ADX DI+ Dashboard (fixed coloring)"

app.layout = html.Div([
    html.H2(f"Sector-wise Average DI+ (Last {LAST_N_DAYS} Days)", style={'textAlign': 'center'}),

    # === Sector Table ===
    dash_table.DataTable(
        id='sector-table',
        columns=[{"name": "Sector", "id": "Sector"}] +
                [{"name": c, "id": c} for c in sector_display.columns if c != "Sector"],
        data=sector_display.to_dict("records"),
        row_selectable="single",
        selected_rows=[],
        style_data_conditional=color_scale(sector_display),
        style_cell={'textAlign': 'center', 'padding': '5px', 'fontFamily': 'Arial', 'fontSize': '14px'},
        style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
        sort_action='native',
        page_size=20,
    ),

    # === Color Legend ===
    html.Div([
        html.H4("DI+ Color Legend", style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Div([
            html.Div(" <16 ", style={'backgroundColor': '#ffb3b3', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
            html.Div("16–19", style={'backgroundColor': '#ccffcc', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
            html.Div("20–24", style={'backgroundColor': '#99ff99', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
            html.Div("25–29", style={'backgroundColor': '#66ff66', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
            html.Div("30–39", style={'backgroundColor': '#33cc33', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
            html.Div("≥40", style={'backgroundColor': '#66b3ff', 'padding': '4px 8px', 'margin': '2px', 'borderRadius': '4px'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'flexWrap': 'wrap'})
    ]),

    # === Symbol Modal ===
    html.Div(
        id='modal',
        style={'display': 'none', 'position': 'fixed', 'top': '0', 'left': '0',
               'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.5)',
               'zIndex': 1000, 'overflow': 'auto'},
        children=html.Div(
            style={'backgroundColor': 'white', 'margin': '5% auto', 'padding': '20px',
                   'width': '90%', 'borderRadius': '10px', 'position': 'relative'},
            children=[
                html.Button("Close", id='close-modal', style={'float': 'right', 'marginBottom': '10px'}),
                html.H3(id='modal-title', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='symbol-table',
                    style_cell={'textAlign': 'center', 'padding': '5px', 'fontFamily': 'Arial', 'fontSize': '14px'},
                    style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
                    sort_action='native',
                    page_size=20
                )
            ]
        )
    )
])

# === Callbacks ===
@app.callback(
    Output('modal', 'style'),
    Output('modal-title', 'children'),
    Output('symbol-table', 'columns'),
    Output('symbol-table', 'data'),
    Output('symbol-table', 'style_data_conditional'),
    Input('sector-table', 'selected_rows'),
    Input('close-modal', 'n_clicks'),
    State('sector-table', 'data'),
    prevent_initial_call=True
)
def handle_modal(selected_rows, close_clicks, sector_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'none'}, "", [], [], []

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'close-modal':
        return {'display': 'none'}, "", [], [], []

    if trigger_id == 'sector-table':
        if not selected_rows:
            return {'display': 'none'}, "", [], [], []

        sector_name = sector_data[selected_rows[0]]["Sector"]
        modal_title = f"Symbols in Sector: {sector_name}"

        # Filter sector data
        sector_df = df[df["Sector"] == sector_name]

        # Pivot by symbol for exact di_plus_10 values
        pivot_symbol = sector_df.pivot(index=["Market Cap", "Symbol"], columns="Date", values="di_plus_10")
        pivot_symbol = pivot_symbol[sorted(pivot_symbol.columns, reverse=True)]
        pivot_symbol = pivot_symbol.round(2)
        pivot_symbol = pivot_symbol.rename(columns={d: d.strftime("%d-%b") for d in pivot_symbol.columns})
        display_symbol = pivot_symbol.reset_index()

        # Ensure numeric columns
        for c in display_symbol.columns:
            if c not in ["Market Cap", "Symbol"]:
                display_symbol[c] = pd.to_numeric(display_symbol[c], errors="coerce")

        columns = [{"name": "Market Cap", "id": "Market Cap"},
                   {"name": "Symbol", "id": "Symbol"}] + \
                  [{"name": c, "id": c} for c in display_symbol.columns if c not in ["Market Cap", "Symbol"]]

        style_data_conditional = color_scale(display_symbol)

        return {'display': 'block'}, modal_title, columns, display_symbol.to_dict("records"), style_data_conditional

    return {'display': 'none'}, "", [], [], []

if __name__ == "__main__":
    app.run(debug=True, port=8051)
