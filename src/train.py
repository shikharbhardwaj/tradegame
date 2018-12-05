#!/usr/bin/env python3

from agent import Agent
from btgym import BTgymEnv
from gym import spaces
import numpy as np
import math

# Path to the data being tested
data_path = '/home/shikhar/dev/btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

# Training environment parameters
cash = 100000
state_len = 128

env     = BTgymEnv(filename=data_path,
                   state_shape={'raw': spaces.Box(low=0.8, high=2, shape=(state_len, 4))},
                #    skip_frame=5,
                   start_cash=cash,
                #    broker_commission=0.001,
                #    fixed_stake=1,
                #    drawdown_call=90,
                   render_ylabel='Price Lines',
                   render_size_episode=(12,8),
                   render_size_human=(8, 3.5),
                   render_size_state=(10, 3.5),
                   render_dpi=75,
                   verbose=0,) 

episodes = 101
trial_len = 200

agent = Agent(env=env)

# State shape is (batch_size, timesteps, dim)
# https://stackoverflow.com/questions/43234504/understanding-lstm-input-shape-in-keras-with-different-sequence
# https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras

for e in range(episodes): 
    cur_state = np.array(list(env.reset().items())[0][1])
    cur_state = np.reshape(cur_state, (1, state_len, 4))
    score = 0
    max_positive_reward = 0

    for step in range(trial_len):
        action = agent.get_action(cur_state)
        new_state, reward, done, info = env.step(action)
        # print('ACTION: {}\nREWARD: {}\nINFO: {}'.format(action, reward, info))

        reward = reward * 10000000  if not done else -10

        max_positive_reward = max(max_positive_reward, reward)

        new_state = list(new_state.items())[0][1]
        new_state = np.reshape(new_state, (1, state_len, 4))
        action_idx = agent.encode_action(action)

        agent.remember(cur_state, action_idx, reward, new_state, done)
        agent.train_model()            # internally iterates default (prediction) model

        score += reward
        cur_state = new_state
        if done:
            break
    
    agent.update_target_model()
    print("Episode :", e, ", score :", score, "memory length : ", len(agent.memory), "epsilon :", agent.epsilon, 'max pos reward : ', max_positive_reward)

    # dqn_agent.render_all_modes(env)
    if e % 20 == 0:
        agent.save_model('model{}'.format(e))
    
env.close()
