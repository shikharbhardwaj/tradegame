""" environment.py

Represents the trading environment providing price data, simulating trades and
generating reward signals for the agent.
"""

import numpy as np

class Environment:
    def __init__(self, startCash, tradeSize, currencyPairs, batcheGen):
        self.cash = startCash
        self.trade_size = tradeSize
        self.currency_pairs = currencyPairs
        self.batch_gen = batcheGen
        self.last_action = 0
        self.time = 0
        self.portfolio = {}
        self.currentBatch = next(self.batch_gen)
        self.current_state = np.array(7 + len(currencyPairs) * 16)

    def process_batch(self):
        """Create the state


        State layout:
        Time features -> _ _ _ _ (4)
        Market state features -> _ _ _ (16 * n)
        Position features -> _ _ _ (3)
        """
        raise NotImplementedError


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
            np.array -- state matrix of size (6 + n * 16)
        """
        return self.current_state


