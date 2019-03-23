"""Visualize trading actions
"""

import sys
from os import path

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
from datetime import datetime

if len(sys.argv) != 5:
    print("python action_vis.py [pair] [year] [month] [action file]")
    exit(0)

pair = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]
action_file = sys.argv[4]

data_file = f'{pair}-{year}-{month}_15T.csv'

data_location = path.join('D:', 'tradegame_data', 'sampled_data_15T', data_file)
actions_location = path.join('..', 'src', 'metrics', action_file)

df = pd.read_csv(data_location)
action_df = pd.read_csv(actions_location)

df['tick'] = pd.to_datetime(df['tick'])
action_df['tick'] = pd.to_datetime(action_df['tick'])

df.set_index('tick', inplace=True)
action_df.set_index('tick', inplace=True)

buy_points = []
sell_points = []
hold_points = []

for tick in df.index:
    try:
        action_df.loc[tick]
    except:
        hold_points.append(tick)
        continue
    if action_df.loc[tick][1] == 1:
        buy_points.append(tick)
    elif action_df.loc[tick][1] == 2:
        sell_points.append(tick)
    else:
        hold_points.append(tick)

plt.style.use('seaborn-whitegrid')

plt.scatter(buy_points, df.loc[buy_points]['close'], c='green', marker='.', s=6, label='Buy')
plt.scatter(sell_points, df.loc[sell_points]['close'], c='salmon', marker='.', s=6, label='Sell')
plt.scatter(hold_points, df.loc[hold_points]['close'], c='dodgerblue', marker='.', s=6, label='Hold')

plt.ylabel('Closing price')
plt.xlabel('Time')
plt.title(data_file[0:data_file.find('_15T')])


green_patch = mpatches.Patch(color='green', label='Buy')
salmon_patch = mpatches.Patch(color='salmon', label='Sell')
blue_patch = mpatches.Patch(color='dodgerblue', label='Hold')

plt.legend(handles=[green_patch, salmon_patch, blue_patch])

plt.show()
