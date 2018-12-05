# Agent operating using DQN
import random
from collections import deque

import numpy as np

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

class Agent:
    def __init__(self, env):
        self.MEMORY_SIZE = 2000
        self.state_size = env.observation_space.shape['raw'][0]

        self.env     = env
        self.memory  = deque(maxlen=self.MEMORY_SIZE)
        
        # Hyperparams
        self.gamma = 1           # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.05
        self.tau = 0.125            # Smoothing rate
        self.min_exp = 40         # We start learning after some experience
        self.batch_size = 64

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        #  -- Fully connected net model --
        # state_shape  = list(self.env.observation_space.shape.items())[0][1]
        # Reshaping for LSTM 
        # state_shape=np.array(state_shape)
        # state_shape= np.reshape(state_shape, (30,4,1))
        # model.add(Dense(24, input_dim=state_shape[1], activation="relu"))
        # model.add(Dense(48, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        # model.add(Dense(4))
        # model.compile(loss="mean_squared_error",
        #     optimizer=Adam(lr=self.learning_rate))

        #  -- LSTM Model -- 
        # LSTM -> Dropout -> Fully Connected -> Linear output
        # model.add(Dense(24, input_dim=(self.state_len, 4), activation='relu'))
        # model.add(Dense(48, activation='relu'))
        model.add(LSTM(64,
               input_shape=(self.state_size, 4), # The state of each episode is last x-OHLC prices of the asset
               # return_sequences=True,
               stateful=False
        ))

        # model.add(Dropout(0.5))
        # The 4 possible actions are (‘hold’, ‘buy’, ‘sell’, ‘close’) 
        # With index encoding 0, 1, 2, 3 in the action space
        model.add(Dense(self.env.action_space.cardinality, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decode_action(self, idx):
        action = None
        # TODO: Ugly hack for converting to base2
        if idx == 0:
            action = self.env.action_space.decode(np.array([0]))
        elif idx == 1:
            action = self.env.action_space.decode(np.array([1]))
        elif idx == 2:
            action = self.env.action_space.decode(np.array([10]))
        elif idx == 3:
            action = self.env.action_space.decode(np.array([11]))
        return action

    def encode_action(self, action):
        bits = self.env.action_space.encode(action)
        idx = int(bits.dot(2 ** np.arange(bits.size)))

        return idx

    def get_action(self, state, mode='train'):
        # Explore by random action
        if np.random.random() <= self.epsilon and mode == 'train':
            return self.env.action_space.sample()

        # Select the best action (with highest Q-Value)
        action_idx = np.argmax(self.model.predict(state)[0])

        return self.decode_action(action_idx)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train_model(self):
        if len(self.memory) < self.min_exp:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size, 4))
        update_target = np.zeros((batch_size, self.state_size, 4))

        action, reward, done = [], [], []
        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_val[i])
        
        self.model.fit(update_input, target, batch_size = self.batch_size,
                       epochs=1, verbose=1)

    def save_model(self, fn):
        self.model.save(fn)
              
    def load_model(self, fn):
        self.model = load_model(fn)
    
    # def show_rendered_image(self, rgb_array):
    #     """
    #     Convert numpy array to RGB image using PILLOW and
    #     show it inline using IPykernel.
    #     """
    #     Display.display(Image.fromarray(rgb_array))

    # def render_all_modes(self, env):
    #     """
    #     Retrieve and show environment renderings
    #     for all supported modes.
    #     """
    #     for mode in self.env.metadata['render.modes']:
    #         print('[{}] mode:'.format(mode))
    #         self.show_rendered_image(self.env.render(mode))

