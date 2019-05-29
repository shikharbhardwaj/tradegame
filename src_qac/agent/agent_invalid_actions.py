import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, ELU, TimeDistributed
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from collections import deque


class Agent:
    def __init__(self, state_dim=22, memory_size=480, is_eval=False, actor_model_location="", critic_model_location="", plot=None):
        self.action_size = 3			# sit, buy, sell
        self.memory = deque(maxlen=memory_size)
        # self.model_name = model_location
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
        self.critic_model = load_model(critic_model_location, custom_objects={'model_loss': self.model_loss}) if is_eval else self._critic_model()
        self.target_model = self._critic_model()
        self.actor_model = load_model(actor_model_location, custom_objects={'model_loss': self.model_loss}) if is_eval else self._actor_model()

        self.plot = plot
        # Minor exploration during evaluation for to prevent overfitting.
        # Ref: https://stats.stackexchange.com/questions/270618/why-does-q-learning-use-epsilon-greedy-during-testing?rq=1
        if is_eval:
            self.epsilon = self.epsilon_min

    def _critic_model(self):
        model = Sequential()
        model.add(TimeDistributed(Dense(units=64, input_dim=(self.state_dim), kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(TimeDistributed(Dense(units=64, kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(LSTM(units=64))
        model.add(Dense(units=3, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=Adam(self.learning_rate))

        return model

    def model_loss(self, q_values, theta_pred):
    	return tf.multiply(tf.math.log(theta_pred), q_values)

    def _actor_model(self):
        model = Sequential()
        model.add(TimeDistributed(Dense(units=64, input_dim=(self.state_dim), kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(TimeDistributed(Dense(units=64, kernel_initializer='he_normal')))
        model.add(ELU())
        model.add(LSTM(units=64))
        model.add(Dense(units=3, activation="linear"))
        model.compile(loss=self.model_loss, optimizer=Adam(lr=self.learning_rate))

        return model


    def act(self, state, valid_actions):
        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)

        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.actor_model.predict(state)[0]) 
        q_values = self.critic_model.predict(state)[0]
        self.actor_model.fit(state, np.expand_dims(q_values, axis=0), verbose=0)
        # self.plot.plot(history.history['loss'])

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
        action_values = self.actor_model.predict(next_states)
        actions = np.argmax(action_values, axis=1)

        # Get Q Values from the target network
        target_q = self.target_model.predict(next_states)
        # print(np.dot(target_q.T,target_q))

        # Action augmentation
        target_q[np.arange(self.sample_size), actions].dot(self.gamma)
        target_q[np.arange(self.sample_size), actions] += rewards

        # Fit the model
        # print(target_q.shape)
        self.critic_model.fit(states, target_q, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def targetUpdate(self):
        # Transfer learned weights from the online model to the fixed target
        weights = self.critic_model.get_weights()
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
