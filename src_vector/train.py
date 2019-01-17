from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
    print("Usage: python train.py [stock] [episodes]")
    exit()

stock_name, episode_count = sys.argv[1], int(sys.argv[2])

l = 1000
batch_size = 96
state_dim = 22

agent = Agent(state_dim, batch_size)

data = getStockDataVec(stock_name)

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))

    # Get inital state
    action = 0
    state = getState(data, 0, action)

    cash = 10000
    trade_size = 100
    prev_portfolio_value = cash
    total_profit = 0
    agent.inventory = []

    for t in range(l):
        prev_price = state[0]
        action = agent.act(state)

        # Get next state.
        next_state = getState(data, t+1, action)
        cur_price = next_state[0]

        # Reward if we sell
        portfolio_value_sale = cash + len(agent.inventory) * cur_price * trade_size

        # if agent.inventory != []:
        #     portfolio_value_sale += trade_size * (cur_price - agent.inventory[0])

        sell_reward = np.log(portfolio_value_sale / prev_portfolio_value)

        # Reward if we buy
        portfolio_value_buy = (cash - cur_price*trade_size) + (len(agent.inventory) + 1) * cur_price * trade_size
        buy_reward = np.log(portfolio_value_buy / prev_portfolio_value)

        # Reward if we hold
        portfolio_value_hold = cash + len(agent.inventory) * cur_price * trade_size
        hold_reward = np.log(portfolio_value_hold / prev_portfolio_value)

        prev_portfolio_value = portfolio_value_hold

        buy_flag = True
        if action == 1 and cash >= cur_price*trade_size: # buy
            agent.inventory.append(cur_price)
            print("Buy: " + formatPrice(cur_price))
            prev_portfolio_value = portfolio_value_buy
            cash -= cur_price * trade_size
        elif action == 1 and cash < cur_price*trade_size:
            buy_flag = False
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            # cur_profit = (cur_price - bought_price) * trade_size
            # total_profit += cur_profit
            # print("Sell: " + formatPrice(cur_price) + " | Profit: " 
            #     + formatPrice(cur_profit))
            print("Sell: " + formatPrice(cur_price))
            prev_portfolio_value = portfolio_value_sale
            cash += cur_price * trade_size

        done = True if t == l - 1 else False
        agent.memory.append((state, 0, hold_reward, next_state, done))
        if buy_flag:
            agent.memory.append((state, 1, buy_reward, next_state, done))
        agent.memory.append((state, 2, sell_reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            total_log_profit = np.log(prev_portfolio_value/cash)
            total_profit = prev_portfolio_value - cash
            print("Total Log Profit: " + formatPrice(total_log_profit))
            print("Total Profit: " + formatPrice(total_profit))
            print("Agent params:")
            print("epsilon :", agent.epsilon)
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay()

        agent.targetUpdate()

    if e % 10 == 0:
            print('Saving model...\n')
            agent.model.save("models/lstm/model_ep" + str(e))
