from keras.models import load_model
from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
        print("Usage: python evaluate.py [stock] [model]")
        exit()

stock_name, model_name = sys.argv[1], sys.argv[2]

window_size = 96

agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)

l = 200
batch_size = 128
state_dim = 22


action = 0
initial_state = np.zeros((window_size + 1, state_dim))
initial_state[:, 0:(state_dim - 3)] = data[0:window_size + 1]
state = getState(data, 0, window_size + 1, action, initial_state)
cash = 10000
trade_size = 1000
prev_portfolio_value = cash
total_profit = 0
agent.inventory = []

for t in range(window_size, l):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1, action, state)
        cur_price = next_state[-1][0]

        if action == 1 and cash >= cur_price * trade_size: # buy
                agent.inventory.append(cur_price)
                print("Buy: " + formatPrice(cur_price))
                cash -= cur_price * trade_size
        elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                cur_profit = (cur_price - bought_price) * trade_size
                total_profit += cur_profit
                print("Sell: " + formatPrice(cur_price) + " | Profit: "
                      + formatPrice(cur_profit))
                cash += cur_price * trade_size

        done = True if t == l - 1 else False
        # agent.memory.append((state, 0, hold_reward, next_state, done))
        # agent.memory.append((state, 1, buy_reward, next_state, done))
        # agent.memory.append((state, 2, sell_reward, next_state, done))
        state = next_state

        if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

