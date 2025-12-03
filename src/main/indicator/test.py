import pandas as pd
import dash
from dash import html, dash_table, Input, Output, State

# === 1️⃣ Read and process data ===
file_path = "output/output.csv"
df = pd.read_csv(file_path)

# Ensure correct date format
df['Date'] = pd.to_datetime(df['Date'])

# === Sector-level Data ===
sector_avg = df.groupby(['Sector', 'Date'])['dailyRSI'].mean().reset_index()
pivot_sector = sector_avg.pivot(index='Sector', columns='Date', values='dailyRSI')
diff_sector = pivot_sector.diff(axis=1)
diff_sector = diff_sector.iloc[:, -10:]
diff_sector = diff_sector[sorted(diff_sector.columns, reverse=True)]
diff_sector = diff_sector.round(2)
diff_sector = diff_sector.rename(columns={d: d.strftime("%Y-%m-%d") for d in diff_sector.columns})
sector_display = diff_sector.reset_index()

# === Smooth red-white-green gradient function ===
def smooth_red_green_gradient(df, col_list=None):
    """
    Smooth red-white-green gradient for Dash DataTable.
    Negative → red, zero → white, positive → green
    """
    if col_list is None:
        col_list = [c for c in df.columns if c not in ['Sector', 'Market Cap', 'Symbol']]
    
    min_val = df[col_list].min().min()
    max_val = df[col_list].max().max()
    styles = []

    for col in col_list:
        for i, val in enumerate(df[col]):
            if pd.isna(val):
                continue
            if val > 0:
                intensity = int(255 - (val / max_val * 255)) if max_val != 0 else 255
                color = f'rgb({intensity},255,{intensity})'  # green shade
            elif val < 0:
                intensity = int(255 - (val / min_val * 255)) if min_val != 0 else 255
                color = f'rgb(255,{intensity},{intensity})'  # red shade
            else:
                color = 'rgb(255,255,255)'  # zero = white
            styles.append({
                'if': {'row_index': i, 'column_id': col},
                'backgroundColor': color,
                'color': 'black'
            })
    return styles

# === Initialize Dash app ===
app = dash.Dash(__name__)
app.title = "Sector RSI Difference Dashboard"

# === Layout ===
app.layout = html.Div([
    html.H2("Sector RSI Difference Dashboard (Last 10 Days)", style={'textAlign': 'center'}),
    
    # Sector Table
    dash_table.DataTable(
        id='sector-table',
        columns=[{"name": "Sector", "id": "Sector"}] +
                [{"name": pd.to_datetime(d).strftime("%d-%b-%Y"), "id": d} 
                 for d in sector_display.columns if d != "Sector"],
        data=sector_display.to_dict('records'),
        row_selectable='single',
        selected_rows=[],
        style_data_conditional=smooth_red_green_gradient(sector_display),
        style_cell={'textAlign': 'center', 'padding': '5px', 'fontFamily': 'Arial', 'fontSize': '14px'},
        style_header={'backgroundColor': '#f4f4f4','fontWeight': 'bold'},
        sort_action='native',
        page_size=20,
    ),

    # Drill-down modal
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
                    style_header={'backgroundColor': '#f4f4f4','fontWeight': 'bold'},
                    sort_action='native',
                    page_size=20
                )
            ]
        )
    )
])

# === Combined callback for modal open/close ===
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

    elif trigger_id == 'sector-table':
        if not selected_rows:
            return {'display': 'none'}, "", [], [], []

        sector_name = sector_data[selected_rows[0]]['Sector']
        modal_title = f"Symbols in Sector: {sector_name}"

        sector_df = df[df['Sector'] == sector_name]

        symbol_avg = sector_df.groupby(['Market Cap', 'Symbol', 'Date'])['dailyRSI'].mean().reset_index()
        pivot_symbol = symbol_avg.pivot(index=['Market Cap', 'Symbol'], columns='Date', values='dailyRSI')
        diff_symbol = pivot_symbol.diff(axis=1)
        diff_symbol = diff_symbol.iloc[:, -10:]
        diff_symbol = diff_symbol[sorted(diff_symbol.columns, reverse=True)]
        diff_symbol = diff_symbol.round(2)
        diff_symbol = diff_symbol.rename(columns={d: d.strftime("%Y-%m-%d") for d in diff_symbol.columns})
        display_symbol = diff_symbol.reset_index()

        columns = [{"name": "Market Cap", "id": "Market Cap"},
                   {"name": "Symbol", "id": "Symbol"}] + \
                  [{"name": pd.to_datetime(d).strftime("%d-%b-%Y"), "id": d} 
                   for d in display_symbol.columns if d not in ["Market Cap", "Symbol"]]

        style_data_conditional = smooth_red_green_gradient(display_symbol)

        return {'display': 'block'}, modal_title, columns, display_symbol.to_dict('records'), style_data_conditional

    return {'display': 'none'}, "", [], [], []

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True, port=8050)
