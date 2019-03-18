""" environment.py

Represents the trading environment providing price data, simulating trades and
generating reward signals for the agent.
"""

import numpy as np

def createTimeFeatures(time):
        features = np.zeros(4)
        # Minute, hour, DoW and Month
        features[0] = np.sin(time.minute * np.pi / 60)
        features[1] = np.sin(time.hour * np.pi / 24)
        features[2] = np.sin(time.dayofweek * np.pi / 7)
        features[3] = np.sin(time.month * np.pi / 12)

        return features

class Environment:
    def __init__(self, startCash, tradeSize, currencyPairs, batchIterator, history=8):
        """Iniitialize the trading environment.

        Arguments:
            startCash {float} -- Starting cash for the trading account
            tradeSize {float} -- Size of a single trading action
            currencyPairs {list} -- List of currency pairs to consider
            batchIterator {BatchIterator} -- Data batch iterator

        Keyword Arguments:
            history {int} -- Size of the market past to keep in market state (default: {8})
        """
        self.cash = startCash
        self.trade_size = tradeSize
        self.num_currency_pairs = len(currencyPairs)
        self.batch_iterator = batchIterator
        self.last_action = 0
        self.portfolio = {}
        self.tick_iterator = next(self.batch_iterator).iterrows()
        self.current_tick = next(self.tick_iterator)
        self.market_state = self.current_tick[1].as_matrix()

        # Advance the first few ticks to accumulate market history.
        for _ in range(history - 1):
            self.current_tick = next(self.tick_iterator)
            self.market_state = np.concatenate((self.market_state, self.current_tick[1].as_matrix()))

    def next(self):
        """Advance state by one tick.
        """
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

    def portfolio_value(self):
        return 0.0

    def execute(self, action):
        """Execute the specified action, generate reward and advance to the
        next state.

        Arguments:
            action {int} -- Action index - 0 (hold), 1 (buy), 2 (sell)

        Returns:
            [float] -- Reward for the action.
        """
        reward = 0.0
        return reward

    def getState(self):
        """Get the current state of the environment

        Returns:
            np.array -- state matrix of size (7 + n * 16)
        """
        time_state = createTimeFeatures(self.current_tick[0])
        action_state = np.zeros(3)
        action_state[self.last_action] = 1

        return np.concatenate((time_state, action_state, self.market_state))

