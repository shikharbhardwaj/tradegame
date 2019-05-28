import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, ELU, TimeDistributed
from keras.optimizers import Adam

import numpy as np
import sys 

from collections import deque

class Agent:
    def __init__(self, state_dim=22, memory_size=480, is_eval=False, model_location=""):
        self.action_size = 3			# sit, buy, sell
        self.memory = deque(maxlen=memory_size)
        self.model_name = model_location
        self.is_eval = is_eval
        self.state_dim = state_dim

        # DRQN Hyperparameters
        self.sample_size = 96
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001
        self.learning_rate = 0.00025

        self.model = load_model(model_location) if is_eval else self._model()
        self.target_model = self._model()

        # Minor exploration during evaluation for to prevent overfitting.
        # Ref: https://stats.stackexchange.com/questions/270618/why-does-q-learning-use-epsilon-greedy-during-testing?rq=1
        if is_eval:
            self.epsilon = self.epsilon_min

    def _model(self):
        model = Sequential()
        model.add(TimeDistributed(Dense(units=32, input_dim=(self.state_dim), kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(TimeDistributed(Dense(units=32, kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(LSTM(units=64))
        model.add(Dense(units=3, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=Adam(self.learning_rate))

        return model

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = np.expand_dims(state, axis=0)
            state = np.expand_dims(state, axis=0)
            options = self.model.predict(state)[0]
            action = np.argmax(options)

        # Deal with invalid actions.
        if action not in valid_actions:
            return 0

        return action

    def sampleMemory(self):
        idx = np.random.permutation(len(self.memory))[:self.sample_size]
        cols = [[], [], [], []] 	# state, action, reward, next_state
        for i in idx:
            memory = self.memory[i]
            for col, value in zip(cols, memory):
                col.append(value)

        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2], cols[3])


    def expReplay(self):
        states, actions, rewards, next_states = self.sampleMemory()

        # Predict actions from the online model
        states = np.expand_dims(states, axis=1)
        next_states = np.expand_dims(next_states, axis=1)
        # print(next_states.shape)
        action_values = self.model.predict(next_states)
        actions = np.argmax(action_values, axis=1)

        # Get Q Values from the target network
        target_q = self.target_model.predict(next_states)
        # print(np.dot(target_q.T,target_q))

        # Action augmentation
        target_q[np.arange(self.sample_size), actions].dot(self.gamma)
        target_q[np.arange(self.sample_size), actions] += rewards

        # Fit the model
        # print(target_q.shape)
        self.model.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def targetUpdate(self):
        # Transfer learned weights from the online model to the fixed target
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def __str__(self):
        string_rep  = "======================\n"
        string_rep += "        Agent\n"
        string_rep += "======================\n"
        string_rep += "Epsilon: " + str(self.epsilon)

        return string_rep
