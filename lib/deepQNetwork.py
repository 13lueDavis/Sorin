import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json
import matplotlib.pyplot as plt

from collections import deque
import json
import os

from config import config

class DeepQNetwork:
    def __init__(self, strategy):
        self.strategy = strategy
        self.STM = []
        self.WM = []


        if config['loadModel']:
            self.model = self.loadModel(config['loadName'])
        else:
            self.batch_size = config['batchSize']
            self.memoryLength = config['memoryLength']
            self.gamma = config['gamma']
            self.epsilon = config['epsilon']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_decay = config['epsilon_decay']
            self.learning_rate = config['learning_rate']
            self.tau = config['tau']

            self.model = self.createModel()

        self.LTM = deque(maxlen=self.memoryLength)


    def createModel(self):
        # The number of hidden neurons should be between the size of the input layer and the size of the output layer.
        # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
        # The number of hidden neurons should be less than twice the size of the input layer.

        model = Sequential()
        stateSize = len(self.strategy.indicators)
        hiddenNeurons = max(3, min(stateSize-2, int(np.floor(stateSize*2/3)+2)))
        model.add(Dense(hiddenNeurons, input_dim=stateSize, activation="relu"))
        model.add(Dense(2))
        model.compile(loss='mean_squared_error',
            optimizer=Adam(lr=self.learning_rate))

        return model

    def loadModel(self, modelName):
        with open('./models/'+modelName+'/model.json', "r") as json_model_file:
            loaded_model_json = json_model_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights('./models/'+modelName+'/weights.h5')
        with open('./models/'+modelName+'/params.json', "r") as json_params_file:
            params = json.loads(json_params_file.read())

        if config['useLoadedParams']:
            self.batch_size = params['batchSize']
            self.memoryLength = params['memoryLength']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            self.epsilon_min = params['epsilon_min']
            self.epsilon_decay = params['epsilon_decay']
            self.learning_rate = params['learning_rate']
            self.tau = params['tau']
        else:
            self.batch_size = config['batchSize']
            self.memoryLength = config['memoryLength']
            self.gamma = config['gamma']
            self.epsilon = config['epsilon']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_decay = config['epsilon_decay']
            self.learning_rate = config['learning_rate']
            self.tau = config['tau']

        model.compile(loss='mean_squared_error',
            optimizer=Adam(lr=self.learning_rate))

        return model

    def saveStrategy(self, indicators):

        model_json = self.model.to_json()
        params = {
            'indicators' : indicators,
            'memoryLength' : self.memoryLength,
            'batchSize' : self.batch_size,
            'gamma' : self.gamma,
            'epsilon' : self.epsilon,
            'epsilon_min' : self.epsilon_min,
            'epsilon_decay' : self.epsilon_decay,
            'learning_rate' : self.learning_rate,
            'tau' : self.tau
        }
        if config['saveAsNew']:
            saveName = config['saveName']
        else:
            saveName = config['loadName']

        params_json = json.dumps(params)
        if not os.path.exists('./models/'+saveName):
            os.mkdir('./models/'+saveName)
        with open('./models/'+saveName+'/model.json', "w+") as json_model_file:
            json_model_file.write(model_json)
        with open('./models/'+saveName+'/params.json', "w+") as json_params_file:
            json_params_file.write(params_json)

        # serialize weights to HDF5
        self.model.save_weights('./models/'+saveName+'/weights.h5')

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.model.predict(state)[0])

    def QReward(self, action, price, newPrice):
        delta = ((newPrice/price) - 1)
        # if delta>=0 and action: reward = delta
        # elif delta<0 and action: reward = delta
        # elif delta<0 and not action: reward = -delta
        # elif delta>=0 and not action: reward = -delta
        # return reward*100000
        return (3.312e-13 - 4.439e-12*action - 1.273*delta + 5.089e-12*(action**2)\
               + 2.727*action*delta - 2.289e-21*(delta**2))*100000

    def addToSTM(self, price, state, action):
        self.STM.append([price, state, action])

    def moveToWM(self, newState, nextPrice):
        if len(self.STM) > config['tradeInterval']:
            memory = self.STM.pop(0)
            price, _, action = memory
            reward = self.QReward(action, price, nextPrice)

            memory.extend([reward, newState, nextPrice])
            self.WM.append(memory)
            return reward
        return 0

    def moveToLTM(self, finalPrice):
        # [price, state, action, reward, newState, nextPrice, finalPrice]
        if len(self.WM) > config['tradeInterval']:
            memory = self.WM.pop(0)
            memory.append(finalPrice)
            self.LTM.append(memory)

    def replay(self):
        if len(self.LTM) < self.batch_size:
            return 0,0

        samples = random.sample(self.LTM, self.batch_size)

        states = np.zeros((self.batch_size, len(self.strategy.indicators)))
        targets = np.zeros((self.batch_size, 2))
        accuracyValues = np.ones(self.batch_size)
        for i,sample in enumerate(samples):

            price, state, action, reward, newState, nextPrice, finalPrice = sample

            target = self.model.predict(state)[0]
            accuracyValues[i] = np.argmax(target) == ((price/nextPrice) >= 1)
            # startPrice = price if action else nextPrice # If you invested, you have more or less to invest next time
            # Q_future = max(self.QReward(0, startPrice, finalPrice), \
            #                self.QReward(1, startPrice, finalPrice))
            # target[action] = reward + self.gamma*Q_future
            target[action] = reward + self.gamma*np.max(target)

            states[i,:] = state
            targets[i,:] = target

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        return np.average(history.history['loss']), np.average(accuracyValues)


    # def trainTarget(self):
    #     weights = self.model.get_weights()
    #     target_weights = self.target_model.get_weights()
    #     for i in range(len(target_weights)):
    #         target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
    #     self.target_model.set_weights(target_weights)
