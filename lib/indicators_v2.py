import numpy as np
import pandas as pd
import lib.utils as ut

from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.technical import cross

from collections import deque

class TimeOfDay:
    def __init__(self, strategy):
        return

    def isReady(self, bar):
        return True

    def run(self, bar):
        delta = pd.to_timedelta(str(bar.getDateTime()).split()[1]) - pd.Timedelta(hours=7, minutes=30)
        deltaMinutes = delta.seconds/60
        return deltaMinutes / 390

class Momentum:
    def __init__(self, strategy, period):
        self.__strategy = strategy
        self.__indication = 0
        self.__params = [0]

        self.sma = ma.SMA(strategy.prices, period)

    def isReady(self, bar):
        return len(self.sma) >= 2 and self.sma[-1] is not None and self.sma[-2] is not None

    def run(self, bar):
        if self.sma[-1]-self.sma[-2] > self.__params[0]:
            self.__indication = 1
        else:
            self.__indication = -1

        return self.__indication

class PeakTrough:
    def __init__(self, strategy, interval):
        self.__strategy = strategy
        self.__indication = 0
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

            if windowHigh > self.lastWindowHigh*self.__params[0]:
                self.__indication = 1

            elif windowLow*self.__params[1] < self.lastWindoLow:
                self.__indication = -1

            else: self.__indication = 0

            self.lastWindowHigh = windowHigh
            self.lastWindoLow = windowLow

        return self.__indication


class FourierPrediction:
    def __init__(self, strategy, period):
        return;

class EMACross:
    def __init__(self, strategy, shortPeriod, longPeriod):
        self.__strategy = strategy
        self.__indication = 0
        self.__params = [1, 1]

        self.shortEMA = ma.EMA(strategy.prices, shortPeriod)
        self.longEMA = ma.EMA(strategy.prices, longPeriod)

        self.numCrossOvers = 0
        self.numCrossUnders = 0

    def isReady(self, bar):
        return True

    def run(self, bar):
        if cross.cross_above(self.shortEMA, self.longEMA) > 0:
            self.numCrossOvers += 1
            if self.numCrossOvers >= self.__params[0]:
                self.__indication = 1
                self.numCrossOvers = 0
        elif cross.cross_below(self.shortEMA, self.longEMA) > 0:
            self.numCrossUnders += 1
            if self.numCrossUnders >= self.__params[1]:
                self.__indication = -1
                self.numCrossUnders = 0
        return self.__indication

class SMACross:
    def __init__(self, strategy, shortPeriod, longPeriod):
        self.__strategy = strategy
        self.__indication = 0
        self.__params = [1, 1]

        self.shortSMA = ma.SMA(strategy.prices, shortPeriod)
        self.longSMA = ma.SMA(strategy.prices, longPeriod)

        self.numCrossOvers = 0
        self.numCrossUnders = 0

    def isReady(self, bar):
        return True

    def run(self, bar):
        if cross.cross_above(self.shortSMA, self.longSMA) > 0:
            self.numCrossOvers += 1
            if self.numCrossOvers >= self.__params[0]:
                self.__indication = 1
                self.numCrossOvers = 0
        elif cross.cross_below(self.shortSMA, self.longSMA) > 0:
            self.numCrossUnders += 1
            if self.numCrossUnders >= self.__params[1]:
                self.__indication = -1
                self.numCrossUnders = 0
        return self.__indication

class Ichimoku:
    def __init__(self, strategy):
        self.__strategy = strategy
        self.__indication = 0
        self.__params = []

        self.tenkanSen = deque(maxlen=2)
        self.kijunSen = deque(maxlen=2)
        self.senkouSpanA = deque(maxlen=26)
        self.senkouSpanB = deque(maxlen=52)

        self.aboveCloud = deque(maxlen=2)
        self.AAboveB = deque(maxlen=2)
        self.tenkanKijunCross = deque(maxlen=2)
        self.priceTenkanCross = deque(maxlen=2)

    def isReady(self, bar):

        if len(self.__strategy.prices) < 52:
            return False

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        p9High = max(self.__strategy.prices[-9:])
        p9Low = min(self.__strategy.prices[-9:])
        self.tenkanSen.append((p9High + p9Low)/2)

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        p26High = max(self.__strategy.prices[-26:])
        p26Low = min(self.__strategy.prices[-26:])
        self.kijunSen.append((p26High + p26Low)/2)

        if len(self.tenkanSen) < 2:
            return False

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        self.senkouSpanA.append((self.tenkanSen[1] + self.kijunSen[1])/2)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        p52High = max(self.__strategy.prices[-52:])
        p52Low = min(self.__strategy.prices[-52:])
        self.senkouSpanB.append((p52High + p52Low)/2)

        if len(self.senkouSpanA) < 26 or len(self.senkouSpanB) < 52:
            return False

        return True

    def run(self, bar):

        aboveCloud = bar.getPrice() > self.senkouSpanA[0] and bar.getPrice() > self.senkouSpanB[0]
        A_above_B = self.senkouSpanA[0] > self.senkouSpanB[0]

        if self.__indication==1 and self.tenkanSen[1] < self.kijunSen[1]: # Exit Criteria
            self.__indication = 0

        if self.__indication==-1 and self.tenkanSen[1] > self.kijunSen[1]:
            self.__indication = 0

        tenkanKijunCrossAbove = (self.tenkanSen[0] <= self.kijunSen[0]) & (self.tenkanSen[1] > self.kijunSen[1])
        priceTenkanCrossAbove = (self.__strategy.prices[-2] <= self.tenkanSen[1]) & (bar.getPrice() > self.tenkanSen[1])
        if aboveCloud and A_above_B and (tenkanKijunCrossAbove or priceTenkanCrossAbove): # Buy
            self.__indication = 1
            return self.__indication

        tenkanKijunCrossBelow = (self.tenkanSen[0] >= self.kijunSen[0]) & (self.tenkanSen[1] < self.kijunSen[1])
        priceTenkanCrossBelow = (self.__strategy.prices[-2] >= self.tenkanSen[1]) & (bar.getPrice() < self.tenkanSen[1])
        if not aboveCloud and not A_above_B and (tenkanKijunCrossBelow or priceTenkanCrossBelow): # Sell
            self.__indication = -1
            return self.__indication

        return self.__indication

class RSI:
    #https://gbeced.github.io/pyalgotrade/docs/v0.16/html/sample_rsi2.html
    def __init__(self, strategy, entrySMA, exitSMA, rsiPeriod):
        self.__strategy = strategy
        self.__indication = 0
        self.__params = [10, 90]

        self.entrySMA = ma.SMA(self.__strategy.prices, entrySMA)
        self.exitSMA = ma.SMA(self.__strategy.prices, exitSMA)
        self.rsi = rsi.RSI(self.__strategy.prices, rsiPeriod)

    def isReady(self, bar):
        if self.exitSMA[-1] is None or self.entrySMA[-1] is None or self.rsi[-1] is None:
            return False
        return True

    def run(self, bar):

        if self.__indication==1 & cross.cross_above(self.__strategy.prices, self.exitSMA):
            self.__indication = 0
        elif self.__indication==-1 & cross.cross_below(self.__strategy.prices, self.entrySMA):
            self.__indication = 0
        elif bar.getPrice() > self.entrySMA[-1] and self.rsi[-1] <= self.__params[0]:
            self.__indication = 1
        elif bar.getPrice() < self.exitSMA[-1] and self.rsi[-1] >= self.__params[1]:
            self.__indication = -1

        return self.__indication

class threeBarPlay:
    def __init__(self, strategy):
        self.__strategy = strategy
        self.__indication = False
        self.__params = [20, 1.5, 0.5]

        self.prevMax = 0
        self.prevMin = 0
        self.overNarrow = False
        self.underNarrow = False
        self.overExciting = False
        self.underExciting = False
        self.prevBarHeight = 0

    def isReady(self, bar):
        if len(self.__strategy.closes) < self.__params[0]:
            return False
        return True

    def run(self, bar):
        barHeights = self.__strategy.closes[-self.__params[0]:-1] - self.__strategy.opens[-self.__params[0]:-1]
        aveBarHeight = np.average(np.abs(barHeights))

        if self.__indication==1 & bar.getPrice() < self.prevMin:
            self.__indication = 0
        if self.__indication==-1 & bar.getPrice() > self.prevMax:
            self.__indication = 0

        if overNarrow & bar.getClose() > self.prevMax:
            self.__indication = 1
            return self.__indication
        elif underNarrow & bar.getClose() < self.prevMin:
            self.__indication = -1
            return self.__indication

        self.overNarrow = False
        self.underNarrow = False

        if self.overExiting & np.abs(bar.getClose()-bar.getOpen()) < self.__params[2]*self.prevBarHeight:
            self.overNarrow = True
            self.prevMax = np.max(bar.getClose(),bar.getOpen())
            self.prevMin = np.min(bar.getClose(),bar.getOpen())
        elif self.underExiting & np.abs(bar.getClose()-bar.getOpen()) < self.__params[2]*-self.prevBarHeight:
            self.underNarrow = True
            self.prevMax = np.max(bar.getClose(),bar.getOpen())
            self.prevMin = np.min(bar.getClose(),bar.getOpen())

        self.overExciting = False
        self.underExiting = False

        barHeight = bar.getClose() - bar.getOpen()
        if not self.__indication & barHeight > self.__params[1]*aveBarHeight:
            self.overExiting = True
            self.prevBarHeight = barHeight
        elif self.__indication & -barHeight > self.__params[1]*aveBarHeight:
            self.underExiting = True
            self.prevBarHeight = barHeight
