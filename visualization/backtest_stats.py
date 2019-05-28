import json
import sys
from os import path
import sys
import platform

import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
if len(sys.argv) != 3:
    print("python backtest_stats.py [config] [metric file]")
    exit(0)

config_file = sys.argv[1]
metric_location = sys.argv[2]

config = json.load(open(config_file))

# Unpack config
location = config['data_location']
model_location = config['model_location']
pairs = config['pairs']
begin_year = config['begin_year']
end_year = config['end_year']
trade_pair = config['trade_pair']
start_cash = config['start_cash']
trade_size = config['trade_size']
spread = config['spread']

data_file = f'{trade_pair}_{begin_year}.csv'

plat = platform.system()

base = 'D:'
if plat != 'Windows':
    base = '~/data'

data_location = path.join(base, 'tradegame_data', 'sampled_data_15T', 'yearly', data_file)

df = pd.read_csv(data_location)
value_df = pd.read_csv(metric_location)

df['tick'] = pd.to_datetime(df['tick'])
value_df['tick'] = pd.to_datetime(value_df['tick'])

df.set_index('tick', inplace=True)
value_df.set_index('tick', inplace=True)

duration = (begin_year, end_year)

hold = (value_df['action'] == 0).sum()
buy = (value_df['action'] == 1).sum()
sell = (value_df['action'] == 2).sum()

final_portfolio_value = value_df.loc[value_df.index[-1]]['value']
highs = value_df['value'].cummax()
lows = value_df['value'].cummin()
drawdowns = highs - lows

print("------------------------------")
print("       Backtest report        ")
print("------------------------------")
print("Test qualititative variables")
print("------------------------------")
print("Instrument:", trade_pair)
print("Currency pairs:", pairs)
print("Duration:", duration)
print("Ticks analysed:", len(value_df))
print("Model description:", "Deep Q-Learning with fixed targets and experience replay.")
print("------------------------------")
print("Test quantitative variables")
print("------------------------------")
print("Initial cash:", start_cash)
print("Trade unit size:", trade_size)
print("Spread:", spread)
print("------------------------------")
print("Evaluation metrics")
print("------------------------------")
print("Trades analysed:", sell + buy)
print("Total net profit:", final_portfolio_value - start_cash)
print("Gross profit:", )
print("Gross loss:", )
print("Absolute drawdown:", drawdowns[-1])
print("Max drawdown:", max(drawdowns))
print("Max drawdown(%):", max(100 * drawdowns / highs))
print("------------------------------")