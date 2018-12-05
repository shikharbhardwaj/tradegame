# Agent operating using DQN
from collections import deque

import numpy as np

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam

class Agent:
    def __init__(self, env):
        self.MEMORY_SIZE = 20000

        self.env     = env
        self.memory  = deque(maxlen=self.MEMORY_SIZE)
        
        self.state_len = env.observation_space.shape['raw'][0]

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        #  -- Fully connected net model --
        # state_shape  = list(self.env.observation_space.shape.items())[0][1]
        # # Reshaping for LSTM 
        # state_shape=np.array(state_shape)
        # state_shape= np.reshape(state_shape, (30,4,1))
        # model.add(Dense(24, input_dim=state_shape[1], activation="relu"))
        # model.add(Dense(48, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        # model.add(Dense(self.env.action_space.n))
        # model.compile(loss="mean_squared_error",
        #     optimizer=Adam(lr=self.learning_rate))

        # Current model : -- LSTM -> Dropout -> Fully Connected -> Linear output
        model.add(LSTM(64,
               input_shape=(4, 1), # The state of each episode is last x-OHLC prices of the asset
               # return_sequences=True,
               stateful=False
        ))

        model.add(Dropout(0.5))
        model.add(Dense(self.env.action_space.cardinality, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        adam = Adam(lr = self.learning_rate) # TODO: Change hyperparams?
        model.compile(loss='mse', optimizer=adam)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # Explore by random action
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Select the best action (with highest Q-Value)
        action_idx = np.argmax(self.model.predict(state)[0])
        action = None

        # TODO: Ugly hack for converting to base2
        if action_idx == 0:
            action = self.env.action_space.decode(np.array([0]))
        elif action_idx == 1:
            action = self.env.action_space.decode(np.array([1]))
        elif action_idx == 2:
            action = self.env.action_space.decode(np.array([10]))
        elif action_idx == 3:
            action = self.env.action_space.decode(np.array([11]))
        return action

   
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

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

