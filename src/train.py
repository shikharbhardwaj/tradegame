from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
        print("Usage: python train.py [stock] [window] [episodes]")
        exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)

data = getStockDataVec(stock_name)
l = data.shape[0] - 1
batch_size = 400
state_dim = 22


for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))

        # Get inital state
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
                prev_price = state[-1][0]
                action = agent.act(state)

                # Get next state.
                next_state = getState(data, t + 1, window_size + 1, action, state)
                cur_price = next_state[-1][0]
                reward = 0

                # The reward is
                # If we sell,
                portfolio_value_sale = cash + len(agent.inventory) * cur_price

                if agent.inventory != []:
                    portfolio_value_sale += trade_size * (cur_price - agent.inventory[0])

                sell_reward = np.log(portfolio_value_sale / prev_portfolio_value)

                # If we buy,
                portfolio_value_buy = cash + len(agent.inventory) * cur_price
                buy_reward = np.log(portfolio_value_buy / prev_portfolio_value)

                # If we hold,
                portfolio_value_hold = cash + len(agent.inventory) * cur_price
                hold_reward = np.log(portfolio_value_hold / prev_portfolio_value)

                prev_portfolio_value = portfolio_value_hold
                reward = hold_reward

                if action == 1 and cash >= cur_price: # buy
                        agent.inventory.append(cur_price)
                        print("Buy: " + formatPrice(cur_price))
                        prev_portfolio_value = portfolio_value_buy
                        reward = buy_reward
                        cash -= cur_price
                elif action == 2 and len(agent.inventory) > 0: # sell
                        bought_price = agent.inventory.pop(0)
                        total_profit += (cur_price - bought_price) * trade_size
                        print("Sell: " + formatPrice(cur_price) + " | Profit: "
                              + formatPrice(cur_price - bought_price))
                        prev_portfolio_value = portfolio_value_sale
                        cash += cur_price
                        reward = sell_reward

                done = True if t == l - 1 else False
                agent.memory.append((state, 0, hold_reward, next_state, done))
                agent.memory.append((state, 1, buy_reward, next_state, done))
                agent.memory.append((state, 2, sell_reward, next_state, done))
                state = next_state

                if done:
                        print("--------------------------------")
                        print("Total Profit: " + formatPrice(total_profit))
                        print("Agent params:")
                        print("ε :", agent.epsilon)
                        print("--------------------------------")

                if len(agent.memory) > batch_size:
                        agent.expReplay(batch_size)

                agent.targetUpdate()

        if e % 10 == 0:
                agent.model.save("models/lstm/model_ep" + str(e))
