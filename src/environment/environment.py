""" environment.py

Represents the trading environment providing price data, simulating trades and
generating reward signals for the agent.
"""

import numpy as np

def createTimeFeatures(time):
    """Create time features for the given timestamp

    Arguments:
        time {datetime} -- A datetime object for the tick

    Returns:
        np.array -- numpy array of shape (4,) with the 4 time features
    """

    features = np.zeros(4)
    # Minute, hour, DoW and Month
    features[0] = np.sin(time.minute * np.pi / 60)
    features[1] = np.sin(time.hour * np.pi / 24)
    features[2] = np.sin(time.dayofweek * np.pi / 7)
    features[3] = np.sin(time.month * np.pi / 12)

    return features


class Environment:
    def __init__(self, currencyPairs, batchIterator, portfolio, history=8):
        """Iniitialize the trading environment.

        Arguments:
            currencyPairs {list} -- List of currency pairs to consider
            batchIterator {BatchIterator} -- Data batch iterator

        Keyword Arguments:
            history {int} -- Size of the market past to keep in market state (default: {8})
        """
        self.num_currency_pairs = len(currencyPairs)
        self.batch_iterator = batchIterator
        self.portfolio = portfolio
        self.tick_iterator = next(self.batch_iterator).iterrows()
        self.current_tick = next(self.tick_iterator)
        self.market_state = self.current_tick[1].as_matrix()
        self.last_action = 0

        # Advance the first few ticks to accumulate market history.
        for _ in range(history - 1):
            self.current_tick = next(self.tick_iterator)
            self.market_state = np.concatenate((self.market_state, self.current_tick[1].as_matrix()))

    def next(self, action):
        """Advance state by one tick.
        """
        # Update last action
        self.last_action = action
        try:
            self.current_tick = next(self.tick_iterator)
        except StopIteration:
            # Move to the next batch.
            self.tick_iterator = next(self.batch_iterator).iterrows()
            self.current_tick = next(self.tick_iterator)

        # The market state leaves the information of the last tick
        # and adds the new incoming tick.

        # A single tick has 2 * num_currency_pairs data points
        num_points = 2 * self.num_currency_pairs
        self.market_state = np.roll(self.market_state, num_points)
        self.market_state[-1-num_points:-1] = self.current_tick[1].as_matrix()

    def execute(self, action):
        """Execute the specified action, generate reward and advance to the
        next state.

        Arguments:
            action {int} -- Action index - 0 (hold), 1 (buy), 2 (sell)

        Returns:
            [float] -- Reward for the action.
        """
        prev_portfolio_value = self.portfolio.value(self.current_tick[0])

        if action == 1:
            self.portfolio.buy(self.current_tick[0])
        elif action == 2:
            self.portfolio.sell(self.current_tick[0])

        # Advance to the next state.
        self.next(action)

        cur_portfolio_value = self.portfolio.value(self.current_tick[0])

        # The reward is the logarithmic return on the portfolio value.
        reward = np.log(cur_portfolio_value / prev_portfolio_value)

        return reward

    def executeAugment(self, action):
        """Execute the specified action and generate reward for all possible
        actions at the current tick.

        Arguments:
            action {int} -- Action index - 0 (hold), 1 (buy), 2 (sell)
        Returns:
            [(float, float, float)] -- A tuple of rewards for each action.
        """
        tick = self.current_tick[0]
        prev_portfolio_value = self.portfolio.value(tick)

        # Move to the next state.
        self.next(action)

        value_hold = self.portfolio.value(self.current_tick[0])
        self.portfolio.buy(tick)
        value_buy = self.portfolio.value(self.current_tick[0])
        # Revert the buy action.
        self.portfolio.sell(tick)
        self.portfolio.sell(tick)
        value_sell = self.portfolio.value(self.current_tick[0])
        # Revert the sell action.
        self.portfolio.buy(tick)

        # Execute the specified action
        if action == 1:
            self.portfolio.buy(tick)
        elif action == 2:
            self.portfolio.sell(tick)

        hold_reward = np.log(value_hold / prev_portfolio_value)
        buy_reward = np.log(value_buy / prev_portfolio_value)
        sell_reward = np.log(value_sell / prev_portfolio_value)

        return (hold_reward, buy_reward, sell_reward)


    def state(self):
        """Get the current state of the environment

        Returns:
            np.array -- state matrix of size (7 + n * 16)
        """
        time_state = createTimeFeatures(self.current_tick[0])
        action_state = np.zeros(3)
        action_state[self.last_action] = 1

        return np.concatenate((time_state, action_state, self.market_state))

    def __str__(self):
        string_rep  = "======================\n"
        string_rep += " Trading environment\n"
        string_rep += "======================\n"
        string_rep += "Last action: " + str(self.last_action) + "\n"
        string_rep += "Current tick: " + str(self.current_tick[0]) + "\n"
        string_rep += "Portfolio value: " + str(self.portfolio.value(self.current_tick[0])) + "\n"
        string_rep += str(self.portfolio)

        return string_rep


# Test usage
if __name__ == '__main__':
    import data
    import portfolio
    pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURCHF', 'EURGBP',
         'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

    stateIter = data.BatchIterator('/data/tradegame_data/sampled_data_15T', pairs, 2012, 2015)
    tradePair = 'EURUSD'
    priceIter = data.BatchIterator('/data/tradegame_data/sampled_data_15T', [tradePair], 2012, 2015, False)

    portfolio = portfolio.Portfolio(100000, 10000, priceIter)

    env = Environment(pairs, stateIter, portfolio)

    print(str(env))

    print("Buy reward : ", env.execute(1))
    print("Buy reward : ", env.execute(1))
    print("Buy reward : ", env.execute(1))

    print(str(env))

    print("Sell reward : ", env.execute(2))
    print("Hold reward : ", env.execute(0))
    print("Sell reward : ", env.execute(2))

    print(str(env))

