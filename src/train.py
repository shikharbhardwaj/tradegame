"""Train the model with data from specified years.
"""

import json
import numpy as np
from os import path, makedirs
import platform
import sys
from time import time

from environment.data import BatchIterator
from environment.portfolio import Portfolio
from environment.environment import Environment
from agent.agent import Agent

# if len(sys.argv) != 2:
#     print("Usage: python train.py [config]")
#     exit(0)

config_file = sys.argv[1] if len(sys.argv) == 2 else "./sample.json"

config = json.load(open(config_file))

# Unpack config
location = config['data_location']
pairs = config['pairs']
begin_year = config['begin_year']
end_year = config['end_year']
trade_pair = config['trade_pair']
start_cash = config['start_cash']
trade_size = config['trade_size']

# Create output model directory
model_dir = path.join(path.dirname(__file__), '..', 'models',
                      'train_'+str(int(time())))

if not path.exists(model_dir):
    makedirs(model_dir)

# Data iterators
state_iter = BatchIterator(location, pairs, begin_year, end_year)
price_iter = BatchIterator(location, [trade_pair], begin_year, end_year, False)

# Set up the environment
portfolio = Portfolio(start_cash, trade_size, price_iter)
env = Environment(pairs, state_iter, portfolio)

state_shape = env.state().shape

# Initialize the agent
agent = Agent(state_shape[0])
print(state_shape)

# Go through the ticks and learn
num_steps = 1

t0 = time()
while True:
    cur_state = env.state()
    action = agent.act(cur_state, env.valid_actions())

    # Get rewards for all possible actions.
    try:
        # rewards = env.executeAugment(action)
        reward = env.execute(action)
    except StopIteration:
        break

    # Get the next state.
    next_state = env.state()

    # Append the possible actions to replay memory
    # next_state[4:7] = np.zeros(3)
    # hold_state = np.copy(next_state)
    # hold_state[4] = 1
    # agent.memory.append((cur_state, 0, rewards[0], hold_state))
    # buy_state = np.copy(next_state)
    # buy_state[5] = 1
    # agent.memory.append((cur_state, 1, rewards[1], buy_state))
    # sell_state = np.copy(next_state)
    # sell_state[6] = 1
    # agent.memory.append((cur_state, 2, rewards[2], sell_state))
    agent.memory.append((cur_state, action, reward, next_state))

    if num_steps % 96 == 0 and len(agent.memory) == 480:
        agent.expReplay()

    agent.targetUpdate()

    if num_steps % 1000 == 0:
        print("Training checkpoint (", num_steps, ") ", sep="", end="")
        try:
            agent.model.save(path.join(model_dir, "model_" + str(num_steps)))
            print(".")
        except NotImplementedError:
            print("x")
        print(str(env))
        print(str(agent))
        print()

    num_steps += 1

# Save training configuration
print("Training ended after processing", num_steps, "ticks")
time_taken = int(time() - t0)

print("Time taken:", time_taken, "s")

config['_time_taken'] = time_taken
config['_ticks_processed'] = num_steps

with open(path.join(model_dir, 'train_config.json'), 'w') as f:
    f.write(json.dumps(config, indent=4))

