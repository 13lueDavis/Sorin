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
        self.target_model = self.createModel()

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

    def QReward(self, action, delta):

        if delta>0 and action: reward = delta*1.2
        elif delta<0 and action: reward = delta
        elif delta<0 and not action: reward = -delta
        elif delta>0 and not action: reward = -delta
        return 0.5 if delta==0 else reward*100000

    def addToSTM(self, state, action):
        self.STM = [state, action]

    def moveToLTM(self, delta, newState):
        if self.STM is not None:
            _, action = self.STM
            reward = self.QReward(action, delta)

            self.STM.extend([reward, delta, newState])
            self.LTM.append(self.STM)
            self.STM = None

            return reward
        return 0


    def replay(self):
        if len(self.LTM) < self.batch_size:
            return 0

        samples = random.sample(self.LTM, self.batch_size)
        states = None
        targetActions = None
        for sample in samples:

            state, action, reward, delta, newState = sample

            targetAction = self.target_model.predict(newState)[0]
            Q_future = self.QReward(np.argmax(targetAction), delta*(action+1))

            targetAction[action] = reward + self.gamma*Q_future

            if states is None:
                states = state
                targetActions = targetAction
            else:
                states = np.vstack((states, state))
                targetActions = np.vstack((targetActions, targetAction))

        history = self.model.fit(states, targetActions, epochs=1, verbose=0)
        return np.average(history.history['loss'])


    def trainTarget(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


        # def addToSTM(self, state, action):
        #     self.STM = [state, action]
        #
        # def addToWM(self, STDelta, STNewState):
        #     if self.STM is not None:
        #         _, action = self.STM
        #         if !action:
        #             reward = self.QReward(action, STDelta)
        #             self.STM.extend([STDelta, reward, STNewState])
        #         else:
        #             self.STM.extend([STDelta])
        #         self.WM.append(self.STM)
        #         self.STM = None
        #
        # def moveToLTM(self, LTDelta, LTNewState):
        #     if self.WM is not None:
        #         action = self.WM[0][1]
        #         if action:
        #             reward = self.QReward(action, LTDelta)
        #             self.WM.extend([reward, LTNewState, LTDelta])
        #         else:
        #             reward = self.WM[0][3]
        #             self.WM.extend([LTDelta])
        #         self.LTM.append(selt.WM)
        #         self.WM = self.WM[:-1]
        #
        #         return self.
        #     return 0
