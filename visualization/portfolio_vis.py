"""Visualize portfolio price with price movement.
"""

import sys
from os import path

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as txt

import pandas as pd
from datetime import datetime

if len(sys.argv) != 5:
    print("python portfolio_vis.py [pair] [year] [month] [metric file]")
    exit(0)

pair = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]
metric_location = sys.argv[4]

data_file = f'{pair}-{year}-{month}_15T.csv'

data_location = path.join('/data', 'tradegame_data', 'sampled_data_15T', data_file)

df = pd.read_csv(data_location)
value_df = pd.read_csv(metric_location)

df['tick'] = pd.to_datetime(df['tick'])
value_df['tick'] = pd.to_datetime(value_df['tick'])

df.set_index('tick', inplace=True)
value_df.set_index('tick', inplace=True)

plt.style.use('seaborn-whitegrid')

plt.subplot(2, 1, 1)

plt.plot(df.index.values, df['close'])
plt.title(data_file[0:data_file.find('_15T')])
plt.ylabel('Closing price')
plt.xlabel('Time')

plt.subplot(2, 1, 2)

plt.plot(df.index.values, value_df.loc[df.index.values]['value'])
plt.title("Agent portfolio value")
plt.ylabel('Portfolio value')
plt.xlabel('Time')

plt.show()
