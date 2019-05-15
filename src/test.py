"""Evaluate the model
"""

import json
import sys
import time

import pandas as pd

from environment.data import BatchIterator
from environment.portfolio import Portfolio
from environment.environment import Environment
from agent.agent import Agent

config_file = sys.argv[1] if len(sys.argv) == 2 else "/home/shikhar/dev/tradegame_new/src/test.json"

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

# Output metrics
metrics_series_location = "metrics/metrics_" + str(int(time.time())) + ".csv"

# Data iterators
state_iter = BatchIterator(location, pairs, begin_year, end_year)
price_iter = BatchIterator(location, [trade_pair], begin_year, end_year, False)

# Set up the environment
portfolio = Portfolio(start_cash, trade_size, price_iter)
env = Environment(pairs, state_iter, portfolio)

state_shape = env.state().shape

agent = Agent(state_shape[0], is_eval = True, model_location = model_location)

num_steps = 1

metrics = pd.DataFrame(columns=['tick', 'action', 'value'])

while True:
    tick = env.current_tick[0]
    cur_state = env.state()
    action = agent.act(cur_state)

    try:
        reward = env.execute(action)
        value = env.portfolio.valueAtTime(env.current_tick[0])
        metrics = metrics.append({'tick': tick, 'action': action, 'value': value}, ignore_index=True)
    except StopIteration:
        print("Training ended after processing", num_steps - 1, "ticks")
        print(str(env))
        print(str(agent))
        print()
        break

    # Get the next state.
    next_state = env.state()

    if num_steps % 1000 == 0:
        print("Evaluation checkpoint (", num_steps, ") ", sep="")
        print(str(env))
        print(str(agent))
        print()

    num_steps += 1

metrics.to_csv(metrics_series_location)
