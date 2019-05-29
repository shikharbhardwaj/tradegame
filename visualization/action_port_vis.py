"""Visualize trading actions with price movement.
"""

import sys
from os import path
import json

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
from datetime import datetime

if len(sys.argv) != 2:
    print("python action_vis.py [metrics file]")
    exit(0)

metrics_file = sys.argv[1]
stamp = str(path.basename(metrics_file)).split('.')[0]
metadata_file = path.join(path.dirname(metrics_file), stamp + ".json")

config = json.load(open(metadata_file))
pair = config['trade_pair']
year = config['begin_year']

metrics_df = pd.read_csv(metrics_file)

metrics_df['tick'] = pd.to_datetime(metrics_df['tick'])

metrics_df.set_index('tick', inplace=True)

buy_points = []
sell_points = []
hold_points = []

start_month = metrics_df.index[0].month

for tick in metrics_df.index:
    try:
        metrics_df.loc[tick]
    except:
        hold_points.append(tick)
        continue
    if metrics_df.loc[tick][1] == 1:
        buy_points.append(tick)
    elif metrics_df.loc[tick][1] == 2:
        sell_points.append(tick)
    else:
        hold_points.append(tick)

    if tick.month != start_month:
        break

plt.style.use('seaborn-whitegrid')

plt.scatter(buy_points, metrics_df.loc[buy_points]['value'], c='green', marker='.', s=8, label='Buy')
plt.scatter(sell_points, metrics_df.loc[sell_points]['value'], c='salmon', marker='.', s=8, label='Sell')
plt.scatter(hold_points, metrics_df.loc[hold_points]['value'], c='dodgerblue', marker='.', s=8, label='Hold')

plt.title(f"{pair}-{year}", fontsize=24)

green_patch = mpatches.Patch(color='green', label='Buy')
salmon_patch = mpatches.Patch(color='salmon', label='Sell')
blue_patch = mpatches.Patch(color='dodgerblue', label='Hold')

plt.legend(handles=[green_patch, salmon_patch, blue_patch])

plt.xlabel('Time')

# plt.ylabel(0.06, 0.5, 'Portfolio value', ha='center', va='center', rotation='vertical', fontsize=12)
plt.ylabel('Portfolio value')

plt.show()
