#!/usr/bin/env python3

from agent import Agent
from btgym import BTgymEnv
from gym import spaces
import numpy as np

import matplotlib.pyplot as plt

# Path to the saved model
model_path = './model100'

# Path to the data being tested
data_path = '/home/shikhar/dev/btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

# Evaluation environment parameters
cash = 10000
state_len = 32

env     = BTgymEnv(filename=data_path,
                   state_shape={'raw': spaces.Box(low=0.8, high=2,shape=(state_len, 4))},
                   skip_frame=5,
                   start_cash=cash,
                   broker_commission=0.02,
                   fixed_stake=100,
                   drawdown_call=90,
                   render_ylabel='Price Lines',
                   render_size_episode=(12,8),
                   render_size_human=(8, 3.5),
                   render_size_state=(10, 3.5),
                   render_dpi=75,
                   verbose=0,) 

print(env.strategy)
ag = Agent(env)
ag.load_model(model_path)

trial_len = 500
cur_state = list(env.reset().items())[0][1]
cur_state = np.reshape(cur_state, (1, state_len, 4))

portfolio_history = []

print('Intial portfolio worth = ', cash)
for i in range(trial_len):
    action = ag.get_action(cur_state, mode='eval')
    new_state, reward, done, info = env.step(action)
    new_state = list(new_state.items())[0][1]
    new_state = np.reshape(new_state, (1, state_len, 4))
    cur_state = new_state
    portfolio_history.append(info[0]['broker_value'])
    print("Agent took action", action)
    print("Broker value ", portfolio_history[-1])

# plt.plot(np.linspace(0, trial_len - 1, trial_len), portfolio_history)
# plt.show()
print('Final portfolio worth = ', portfolio_history[-1])

env.close()