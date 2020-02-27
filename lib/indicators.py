import numpy as np
import pandas as pd
import lib.utils as ut

from pyalgotrade.technical import ma

class BullOrBear:
    def __init__(self, strategy, period):
        self.__strategy = strategy
        self.__indication = False
        self.__params = [0]

        self.sma = ma.SMA(strategy.prices, period)

    def isReady(self, bar):
        return self.sma[-2] is not None

    def run(self, bar):
        if self.sma[-1]-self.sma[-2] > self.__params[0]:
            self.__indication = True
        else:
            self.__indication = False

        return self.__indication

class PeakTrough:
    def __init__(self, strategy, interval):
        self.__strategy = strategy
        self.__indication = False
        self.__params = [1, 1]

        self.interval = interval # Minutes
        self.startTime = pd.Timedelta(hours=7, minutes=30+self.interval)
        self.lastWindowHigh = None
        self.lastWindoLow = None

    def isReady(self, bar):

        isReady = False
        if ut.isAfterTime(bar.getDateTime(), self.startTime):
            if self.lastWindowHigh is None or self.lastWindoLow is None:
                self.lastWindowHigh = np.max(self.__strategy.prices[-self.interval:])
                self.lastWindoLow = np.min(self.__strategy.prices[-self.interval:])
            else:
                isReady = True

        return isReady

    def run(self, bar):

        if ut.isOnInterval(bar.getDateTime(), minutes=self.interval):


            windowHigh = np.max(self.__strategy.prices[-self.interval-1:])
            windowLow = np.min(self.__strategy.prices[-self.interval-1:])

            if windowHigh > self.lastWindowHigh*self.__params[0] and self.__strategy.position is None:
                self.__indication = True

            if windowLow*self.__params[1] < self.lastWindoLow and self.__strategy.position is not None and not self.__strategy.position.exitActive():
                self.__indication = False

            self.lastWindowHigh = windowHigh
            self.lastWindoLow = windowLow

        return self.__indication


class FourierPrediction:
    def __init__(self, strategy, period):
        return;
