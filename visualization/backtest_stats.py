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
# model_location = config['model_location']
pairs = config['pairs']
begin_year = config['begin_year']
end_year = config['end_year']
trade_pair = config['trade_pair']
start_cash = config['start_cash']
trade_size = config['trade_size']
# spread = config['spread']
spread = 0.0

data_file = f'{trade_pair}_{begin_year}.csv'

plat = platform.system()

base = 'D:'
if plat != 'Windows':
    base = '/data'

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

value_df['pnl'] = value_df['value'].diff()
gross_profit = value_df[value_df['pnl'] > 0].sum()['pnl']
gross_loss = value_df[value_df['pnl'] < 0].sum()['pnl']

total_volatility = (value_df['value'] - start_cash).std() / start_cash
ror = (final_portfolio_value  - start_cash) / start_cash

risk_free_rate = 0.011

downside_volatility = (value_df[value_df['pnl'] < 0]['value'] - start_cash).std() / start_cash

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
# print("Spread:", spread)
print("------------------------------")
print("Evaluation metrics")
print("------------------------------")
print("Trades analysed:", sell + buy)
print("Total net profit:", round(final_portfolio_value - start_cash, 2))
print("Gross profit:", round(gross_profit))
print("Gross loss:", round(gross_loss))
print("Rate of return(%):", round(100 * ror, 3))
print("Absolute drawdown:", round(drawdowns[-1], 2))
print("Max drawdown:", round(max(drawdowns), 2))
print("Max drawdown(%):", round(max(100 * drawdowns / highs), 2))
print("Return volatility:", round(total_volatility, 4))
print("Downside volatility:", round(downside_volatility, 4))
print("Sharpe ratio:", (ror - risk_free_rate) / total_volatility)
print("Sortino ratio:", (ror - risk_free_rate) / downside_volatility)
print("------------------------------")
print(trade_pair, sell + buy, round(100 * ror, 3),(ror - risk_free_rate) / total_volatility, (ror - risk_free_rate) / downside_volatility)
# EURUSD	10156	5	6.25	2.7301	2.7697:
