import numpy as np
import math

# prints formatted price
def formatPrice(n):
        return ("-$" if n < 0 else "$") + "{0:.7f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
        fname = "data/" + key + ".csv"
        vec = np.genfromtxt(fname, skip_header=1, delimiter=',')
        return vec

# returns the sigmoid
def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t, n, action, prev_state):
        next_state = np.zeros((prev_state.shape))
        d = t - n + 1
        block = None
        next_state[0:n-1] = prev_state[1:]
        # Append action 1-hot encoding
        action_one_hot = np.zeros(3)
        action_one_hot[action] = 1
        next_state[-1] = np.concatenate([data[t], action_one_hot])
        return next_state

