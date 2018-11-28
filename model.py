#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential()

model.add(Dense(4, init='lecun_uniform', input_shape=(2,)))
model.add(Activation('relu'))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear'))


model.compile(loss = 'mse', optimizer='rmsprop')

