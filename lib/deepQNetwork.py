import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from collections import deque
import json
import os

class DeepQNetwork:
    def __init__(self, strategy):
        self.strategy = strategy
        self.STM = None
        self.WM = []
        self.LTM = deque(maxlen=1000)

        self.modelName = 'SORIN_1'

        self.batch_size = 8

        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 1-5e-5
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.createModel()

    def createModel(self):
        # The number of hidden neurons should be between the size of the input layer and the size of the output layer.
        # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
        # The number of hidden neurons should be less than twice the size of the input layer.

        model = Sequential()
        stateSize = len(self.strategy.indicators)
        model.add(Dense(stateSize+2, input_dim=stateSize, activation="relu"))
        # model.add(Dense(stateSize+2, activation="relu"))
        model.add(Dense(2))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))

        return model

    def saveModel(self, indicators):

        model_json = self.model.to_json()
        indicators = [
            {
                'TYPE' : str(indicator['TYPE'])[23:-2],
                'PARAMS' : indicator['PARAMS']
            }
        for indicator in indicators]
        params = {
            'indicators' : indicators,
            'batch_size' : self.batch_size,
            'gamma' : self.gamma,
            'epsilon' : self.epsilon,
            'epsilon_min' : self.epsilon_min,
            'epsilon_decay' : self.epsilon_decay,
            'learning_rate' : self.learning_rate,
            'tau' : self.tau
        }
        params_json = json.dumps(params)
        if not os.path.exists('./models/'+self.modelName):
            os.mkdir('./models/'+self.modelName)
        with open('./models/'+self.modelName+'/model.json', "w+") as json_model_file:
            json_model_file.write(model_json)
        with open('./models/'+self.modelName+'/params.json', "w+") as json_params_file:
            json_params_file.write(params_json)

        # serialize weights to HDF5
        self.model.save_weights('./models/'+self.modelName+'/weights.h5')

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.model.predict(state)[0])

    def QReward(self, action, oldPrice, newPrice):
        delta = (newPrice/oldPrice) - 1
        if delta>0 and action: reward = delta*1.2
        elif delta<0 and action: reward = delta
        elif delta<0 and not action: reward = -delta
        elif delta>0 and not action: reward = -delta
        return 0.5 if delta==0 else reward*100000

    def addToSTM(self, price, state, action):
        self.STM = [price, state, action]

    def moveToWM(self, STPrice, STNewState):
        reward = 0
        if self.STM is not None:
            price, _, action = self.STM
            if not action:
                reward = self.QReward(action, price, STPrice)
                self.STM.extend([STPrice, reward, STNewState])
            else:
                self.STM.extend([STPrice])
            self.WM.append(self.STM)
            self.STM = None
        return reward

    def moveToLTM(self, LTPrice, LTNewState):
        # Memory format:
        #   [price, state, action, STPrice, reward, newState, LTPrice]
        rewards = []
        while len(self.WM) > 0:
            memory = self.WM.pop(0)
            price = memory[0]
            action = memory[2]
            if action:
                reward = self.QReward(action, price, LTPrice)
                memory.extend([reward, LTNewState, LTPrice])
            else:
                memory.extend([LTPrice])
            self.LTM.append(memory)
            rewards.append(memory[4])
        return np.average(rewards)


    def replay(self):
        if len(self.LTM) < self.batch_size:
            return 0.

        samples = random.sample(self.LTM, self.batch_size)
        states = None
        targetActions = None
        for sample in samples:

            price, state, action, STPrice, reward, newState, LTPrice = sample

            targetAction = self.model.predict(state)[0]
            Q_future = self.QReward(np.argmax(targetAction), delta*(action+1))

            targetAction[action] = reward# + self.gamma*Q_future

            if states is None:
                states = state
                targetActions = targetAction
            else:
                states = np.vstack((states, state))
                targetActions = np.vstack((targetActions, targetAction))

        history = self.model.fit(states, targetActions, epochs=1, verbose=0)
        return np.average(history.history['loss'])
