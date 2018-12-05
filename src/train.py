#!/usr/bin/env python3

from agent import Agent
from btgym import BTgymEnv
from gym import spaces
import numpy as np

# Path to the data being tested
data_path = '/home/shikhar/dev/btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv'

# Training environment parameters
cash = 100000
state_len = 30

env     = BTgymEnv(filename=data_path,
                   state_shape={'raw': spaces.Box(low=-100, high=100, shape=(state_len,4))},
                   skip_frame=5,
                   start_cash=100000,
                   broker_commission=0.02,
                   fixed_stake=100,
                   drawdown_call=90,
                   render_ylabel='Price Lines',
                   render_size_episode=(12,8),
                   render_size_human=(8, 3.5),
                   render_size_state=(10, 3.5),
                   render_dpi=75,
                   verbose=0,) 
gamma   = 0.9
epsilon = .95

trials  = 101
trial_len = 1000

# updateTargetNetwork = 1000
dqn_agent = Agent(env=env)

for trial in range(trials): 
    #dqn_agent.model= load_model("./model.model")
    cur_state = np.array(list(env.reset().items())[0][1])
    cur_state= np.reshape(cur_state, (state_len,4,1))
    for step in range(trial_len):
        action = dqn_agent.act(cur_state)
        new_state, reward, done, _ = env.step(action)
        reward = reward*10 if not done else -10
        new_state =list(new_state.items())[0][1]
        new_state= np.reshape(new_state, (state_len,4,1))
        dqn_agent.target_train() # iterates target model

        cur_state = new_state
        if done:
            break
    
    print("Completed trial #{} ".format(trial))
    # dqn_agent.render_all_modes(env)
    if trial % 20 == 0:
        dqn_agent.save_model("model_{0}.model".format(trial))
        

if __name__ == "__main__":
    main()
