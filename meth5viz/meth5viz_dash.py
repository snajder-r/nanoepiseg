# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Test", id="test"),
    html.Label('Chromosome:'),
    dcc.Input(id="Chromosome", value='13', type='Text'),
    html.Label('Start:'),
    dcc.Input(id='Start', value=57630334, type='Number'),
    html.Label('End:'),
    dcc.Input(id='End', value=57637150, type='Number'),
    html.Button("Go", id="btn_go")
], style={'columnCount': 4})

@app.callback(
    Output(component_id="test", component_property="children"),
    Input(component_id="btn_go", component_property="value")
)
def update_test(input_value):
    return str(input_value)

if __name__ == '__main__':
    app.run_server(debug=False)