"""Train the model with data from specified years.
"""

import json
import sys
import numpy as np

from environment.data import BatchIterator
from environment.portfolio import Portfolio
from environment.environment import Environment
from agent.agent import Agent

# if len(sys.argv) != 2:
#     print("Usage: python train.py [config]")
#     exit(0)

config_file = sys.argv[1] if len(sys.argv) == 2 else "/home/shikhar/dev/tradegame_new/src/sample.json"

config = json.load(open(config_file))

# Unpack config
location = config['data_location']
pairs = config['pairs']
begin_year = config['begin_year']
end_year = config['end_year']
trade_pair = config['trade_pair']
start_cash = config['start_cash']
trade_size = config['trade_size']

# Data iterators
state_iter = BatchIterator(location, pairs, begin_year, end_year)
price_iter = BatchIterator(location, [trade_pair], begin_year, end_year, False)

# Set up the environment
portfolio = Portfolio(start_cash, trade_size, price_iter)
env = Environment(pairs, state_iter, portfolio)

state_shape = env.state().shape

# Initialize the agent
agent = Agent(state_shape[0])

# Go through the ticks and learn
num_steps = 1

while True:
    cur_state = env.state()
    action = agent.act(cur_state)

    # Get rewards for all possible actions.
    try:
        rewards = env.executeAugment(action)
    except StopIteration:
        print("Training ended after processing", num_steps, "ticks")
        break

    # Get the next state.
    next_state = env.state()

    # Append the possible actions to replay memory
    next_state[4:7] = np.zeros(3)
    hold_state = np.copy(next_state)
    hold_state[4] = 1
    agent.memory.append((cur_state, 0, rewards[0], hold_state))
    buy_state = np.copy(next_state)
    buy_state[5] = 1
    agent.memory.append((cur_state, 1, rewards[1], buy_state))
    sell_state = np.copy(next_state)
    sell_state[6] = 1
    agent.memory.append((cur_state, 2, rewards[2], sell_state))

    if num_steps % 96 == 0 and len(agent.memory) == 480:
        agent.expReplay()

    agent.targetUpdate()

    if num_steps % 1000 == 0:
        print("Model checkpoint (", num_steps, ") ", sep="", end="")
        try:
            agent.model.save("models/flat_state_exp/model_" + str(num_steps))
            print("✓")
        except NotImplementedError:
            print("❌")
        print(str(env))
        print(str(agent))
        print()

    num_steps += 1


