import datetime
from os import path
import sys
import platform

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly import tools
import plotly.graph_objs as go

# Load metrics
if len(sys.argv) != 5:
    print("python interactive_portfolio.py [pair] [year] [month] [metric file]")
    exit(0)

pair = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]
metric_location = sys.argv[4]

data_file = f'{pair}_{year}.csv'

plat = platform.system()

base = 'D:'
if plat != 'Windows':
    base = '/data'

print(data_file)
data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', 'yearly', data_file)

df = pd.read_csv(data_location)
value_df = pd.read_csv(metric_location)

df['tick'] = pd.to_datetime(df['tick'])
value_df['tick'] = pd.to_datetime(value_df['tick'])

df.set_index('tick', inplace=True)
value_df.set_index('tick', inplace=True)

print(df.head())
print(value_df.head())

value_plot = go.Line(x = df.index, y = value_df['value'], name='Portfolio value')
price_plot = go.Line(x = df.index, y = df['close'], name='Instrument price')

fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.append_trace(value_plot, 1, 1)
fig.append_trace(price_plot, 2, 1)

# def metrics_table():
#     pnl = 

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Portoflio value and related metrics', style={'color': '#7FDBFF'}),



    html.Div(children='''
        Evaluating the performance of the agent on real world market data.
    '''),

    # Portfolio value and instrument price vs time
    dcc.Graph(figure = fig, id='portfolio-graph', style={"height" : "70vh", "width" : "100%"}),

    # Metrics
    html.H2(children='''
        Metrics
    '''),

    # html.Div(metrics_table())
])

if __name__ == '__main__':
    app.run_server(debug=True)


# Sample call:
# python .\interactive_portfolio.py EURUSD 2017 01 ..\src\metrics\metrics_1553434705.csv
