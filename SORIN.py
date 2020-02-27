from __future__ import print_function

import pyalgotrade
from pyalgotrade import barfeed
from pyalgotrade.barfeed import csvfeed
from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import cross
from pyalgotrade.stratanalyzer import returns
from pyalgotrade import plotter

from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades

import matplotlib.pyplot as plt
import sys
import numpy as np
import glob
import pandas as pd
import time
from pprint import pprint
import logging
from keras.utils import plot_model
from collections import deque

from lib.stockPlotter import StockPlotter
from lib.deepQNetwork import DeepQNetwork
from lib import utils as ut
from lib.indicators import *

logging.basicConfig(filename='sorin.log', filemode='w', level=logging.DEBUG)
logger = logging.getLogger()
logger.propagate = False

indicators = [
    {
        'TYPE' : PeakTrough,
        'PARAMS' : dict(
            interval=30
       )
    },
    {
        'TYPE' : BullOrBear,
        'PARAMS' : dict(
            period=1000
        )
    }
]

class SORIN(strategy.BacktestingStrategy):
    def __init__(self, feed, symbol):
        super(SORIN, self).__init__(feed)

        self.getBroker().getFillStrategy().setVolumeLimit(None)
        self.prices = feed[symbol].getPriceDataSeries()

        self.__symbol = symbol
        self.__initialPrice = None
        self.position = None

        self.__barsProcessed = 0
        self.__startDate = None
        self.__endDate = None

        self.indicators = []
        for i, indDict in enumerate(indicators):
            self.indicators.append(indDict['TYPE'](self, **indDict['PARAMS']))

        self.DQN = DeepQNetwork(self)

        self.updatePlotPeriod = 400
        self.__plotter = StockPlotter()
        self.__plotter.addPlot('Position')
        self.__plotter.addPlot('DQN')
        self.__plotter.updatePlot('DQN', 'Loss', 0,0, c='#2dedad')
        self.__plotter.plots['DQN']['Loss']['xData'] = list(range(99))
        self.__plotter.plots['DQN']['Loss']['yData']= np.zeros(99).tolist()
        self.__plotter.updatePlot('DQN', 'Loss', 100,0)
        self.__plotter.addInfo()

        self.lossHistory = deque(np.zeros(100), maxlen=100)

        print(60*'\n')

    def setStartDate(self, date=None):
        ''' Set date to start backtest '''
        if date is not None:
            self.__startDate = pd.to_datetime(date, infer_datetime_format=True)

    def setEndDate(self, date=None):
        ''' Set date to stop backtest '''
        if date is not None:
            self.__endDate = pd.to_datetime(date, infer_datetime_format=True)

    def onEnterCanceled(self, position):
        ''' When you cancle and order, you dont have a current position '''
        self.position = None

    def onExitOk(self, position):
        ''' When you exit the market, you dont have a current position '''
        self.position = None

    def onExitCanceled(self, position):
        ''' If the exit was canceled, re-submit it. '''
        self.position.exitMarket()

    def getState(self, bar):
        ## Generate current state from indicators
        state = []
        isReady = True
        for indicator in self.indicators:
            if not indicator.isReady(bar):
                isReady = False
                break
            else: state.append(int(indicator.run(bar)))

        if (np.array(state).ndim == 1):
            state = np.array([state])

        return isReady, state

    def onBars(self, bars):
        ''' Called when there is a new bar from the feed '''
        # s = time.time()

        ##===================== Initial Setup =====================##
        bar = bars[self.__symbol]

        ## Check if you are within the desired time window
        if self.__startDate is not None and bar.getDateTime() < self.__startDate:
            return
        if self.__endDate is not None and bar.getDateTime() > self.__endDate:
            return

        self.__barsProcessed += 1

        ## Save initial price (for benchmark)
        if self.__initialPrice is None:
            self.__initialPrice = bar.getClose()

        ##===== Get state & check if all indicators are ready =====##
        isReady, state = self.getState(bar)


        if isReady: # If all indicators are ready
            ##============= Deep Q Network =============##
            reward = self.DQN.moveToLTM((bar.getPrice()/self.prices[-2])-1, state)
            doInvest = self.DQN.act(state)
            self.DQN.addToSTM(state, doInvest)

            loss = self.DQN.replay()
            self.DQN.trainTarget()

            if self.__barsProcessed % self.updatePlotPeriod*10 == 0:
                self.DQN.saveModel(indicators)

            if self.__barsProcessed % self.updatePlotPeriod == 0:
                print('Loss: {:3.2} Reward: {}, Bars Processed: {}     '.format(loss, np.round(reward,2), self.__barsProcessed), end="\r")
                self.lossHistory.append(loss)
                self.__plotter.plots['DQN']['Loss']['plot'].set_ydata(list(self.lossHistory))
                self.__plotter.plots['DQN']['subplot'].relim()
                self.__plotter.plots['DQN']['subplot'].autoscale_view()
                self.__plotter.redraw()

            ##=============== Buy / Sell ===============##
            ## Buy
            if doInvest and self.position is None:
                shares = int(self.getBroker().getCash() * 0.9 / bar.getPrice())
                self.position = self.enterLong(self.__symbol, shares, True)
                # self.__plotter.addBuyPoint('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())

            ## Sell
            if not doInvest and self.position is not None and not self.position.exitActive():
                self.position.exitMarket()
                # self.__plotter.addSellPoint('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())



        ## Plot
        if self.__barsProcessed % self.updatePlotPeriod == 0:
            self.__plotter.updatePlot('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())
            self.__plotter.updatePlot('Position', 'Buy & Hold', bar.getDateTime(), bar.getClose()*1000000/self.__initialPrice, c=[.4,.4,.4])
            self.__plotter.updateInfo(self.getBroker().getEquity(), bar.getClose()*1000000/self.__initialPrice, self.DQN.epsilon)
            self.__plotter.redraw()


    def hold(self):
        while True:
            self.__plotter.redraw()


##===== Symbol Definition =====##
symbol = sys.argv[1]

##========= Load Data =========##
dataFilenames = glob.glob("../clean_data/{}*.csv".format(symbol))
if not len(dataFilenames): raise Exception('No data for symbol: '+symbol)
else: dataFilename = dataFilenames[0]

feed = csvfeed.GenericBarFeed(barfeed.Frequency.MINUTE, timezone=None, maxLen=1024)
feed.addBarsFromCSV(symbol, dataFilename)

##==== Initialize Strategy ====##
backtest = SORIN(feed, symbol)
backtest.setStartDate()
backtest.setEndDate()

##= Attach Strategy Analyzers =##
retAnalyzer = returns.Returns()
backtest.attachAnalyzer(retAnalyzer)
sharpeRatioAnalyzer = sharpe.SharpeRatio()
backtest.attachAnalyzer(sharpeRatioAnalyzer)
drawDownAnalyzer = drawdown.DrawDown()
backtest.attachAnalyzer(drawDownAnalyzer)
tradesAnalyzer = trades.Trades()
backtest.attachAnalyzer(tradesAnalyzer)

backtest.run()

#==============================================================================#
#                                                                              #
#                                Strategy Analysis                             #
#                                                                              #
#==============================================================================#
print("Final portfolio value: $%.2f" % backtest.getResult())
print("Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
print("Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100))
print("Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))

print()
print("Total trades: %d" % (tradesAnalyzer.getCount()))
if tradesAnalyzer.getCount() > 0:
    profits = tradesAnalyzer.getAll()
    print("Avg. profit: $%2.f" % (profits.mean()))
    print("Profits std. dev.: $%2.f" % (profits.std()))
    print("Max. profit: $%2.f" % (profits.max()))
    print("Min. profit: $%2.f" % (profits.min()))
    returns = tradesAnalyzer.getAllReturns()
    print("Avg. return: %2.f %%" % (returns.mean() * 100))
    print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
    print("Max. return: %2.f %%" % (returns.max() * 100))
    print("Min. return: %2.f %%" % (returns.min() * 100))

print()
print("Profitable trades: %d" % (tradesAnalyzer.getProfitableCount()))
if tradesAnalyzer.getProfitableCount() > 0:
    profits = tradesAnalyzer.getProfits()
    print("Avg. profit: $%2.f" % (profits.mean()))
    print("Profits std. dev.: $%2.f" % (profits.std()))
    print("Max. profit: $%2.f" % (profits.max()))
    print("Min. profit: $%2.f" % (profits.min()))
    returns = tradesAnalyzer.getPositiveReturns()
    print("Avg. return: %2.f %%" % (returns.mean() * 100))
    print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
    print("Max. return: %2.f %%" % (returns.max() * 100))
    print("Min. return: %2.f %%" % (returns.min() * 100))

print()
print("Unprofitable trades: %d" % (tradesAnalyzer.getUnprofitableCount()))
if tradesAnalyzer.getUnprofitableCount() > 0:
    losses = tradesAnalyzer.getLosses()
    print("Avg. loss: $%2.f" % (losses.mean()))
    print("Losses std. dev.: $%2.f" % (losses.std()))
    print("Max. loss: $%2.f" % (losses.min()))
    print("Min. loss: $%2.f" % (losses.max()))
    returns = tradesAnalyzer.getNegativeReturns()
    print("Avg. return: %2.f %%" % (returns.mean() * 100))
    print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
    print("Max. return: %2.f %%" % (returns.max() * 100))
    print("Min. return: %2.f %%" % (returns.min() * 100))

## Keeps figure ##
backtest.hold()
