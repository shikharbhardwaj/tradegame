from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pandas as pd
from datetime import datetime


location = '/data/tradegame_data/sampled_data_15T/EURUSD-2013-01_15T.csv'
actions = '../src/metrics/actions_1553092044.csv'

df = pd.read_csv(location)
action_df = pd.read_csv(actions)

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
plt.title('EURUSD-JAN-2013')


green_patch = mpatches.Patch(color='green', label='Buy')
salmon_patch = mpatches.Patch(color='salmon', label='Sell')
blue_patch = mpatches.Patch(color='dodgerblue', label='Hold')

plt.legend(handles=[green_patch, salmon_patch, blue_patch])

plt.show()
