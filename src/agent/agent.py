import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, ELU, TimeDistributed
from keras.optimizers import Adam

import numpy as np

from collections import deque

class Agent:
    def __init__(self, state_dim=22, batch_size=96, is_eval=False, model_name=""):
        self.action_size = 3			# sit, buy, sell
        self.memory = deque(maxlen=480)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.state_dim = state_dim
        self.batch_size = batch_size

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001
        self.learning_rate = 0.00025

        self.model = load_model("models/" + model_name) if is_eval else self._model()
        self.target_model = self._model()

    def _model(self):
        model = Sequential()
        model.add(TimeDistributed(Dense(units=32, input_dim=(1, self.state_dim), kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(TimeDistributed(Dense(units=32, kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(LSTM(units=64))
        model.add(Dense(units=3, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=Adam(self.learning_rate))

        return model

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)
        options = self.model.predict(state)[0]
        return np.argmax(options)

    def sampleMemory(self):
        idx = np.random.permutation(len(self.memory))[:self.batch_size]
        cols = [[], [], [], [], []] 	# state, action, reward, next_state, done 
        for i in idx:
            memory = self.memory[i]
            for col, value in zip(cols, memory):
                col.append(value)

        cols = [np.array(col) for col in cols]
        # print(cols[0].shape)
        return (cols[0], cols[1], cols[2], cols[3], cols[4])


    def expReplay(self):
        states, actions, rewards, next_states, dones = self.sampleMemory()
       	
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
        target_q[np.arange(self.batch_size), actions].dot(self.gamma)
        target_q[np.arange(self.batch_size), actions] += rewards
        
        # Fit the model
        # print(target_q.shape)
        self.model.fit(states, target_q, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    def targetUpdate(self):
        # Transfer learned weights from the online model to the fixed target
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)