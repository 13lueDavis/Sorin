import pyalgotrade
from pyalgotrade import barfeed
from pyalgotrade.barfeed import csvfeed
from pyalgotrade import strategy

import matplotlib.pyplot as plt
import sys
import numpy as np
import glob
import pandas as pd
import time
import json
from pprint import pprint
import logging
from keras.utils import plot_model
from collections import deque

from lib.stockPlotter import StockPlotter
from lib.deepQNetwork import DeepQNetwork
from lib import utils as ut
from lib import indicators as inds

from config import config

logging.basicConfig(filename='sorin.log', filemode='w', level=logging.DEBUG)
logger = logging.getLogger()
logger.propagate = False

class SORIN(strategy.BacktestingStrategy):
    def __init__(self, feed, symbol):
        super(SORIN, self).__init__(feed)

        self.getBroker().getFillStrategy().setVolumeLimit(None)
        self.prices = feed[symbol].getPriceDataSeries()
        self.opens = feed[symbol].getOpenDataSeries()
        self.closes = feed[symbol].getCloseDataSeries()

        self.__symbol = symbol
        self.__initialPrice = None
        self.position = None

        self.__barsProcessed = 0
        self.__startDate = None
        self.__endDate = None

        self.indicators = []
        if config['loadModel']:
            with open('./models/'+config['loadName']+'/params.json', "r") as json_params_file:
                self.params = json.loads(json_params_file.read())
            for indicator in self.params['indicators']:
                self.indicators.append(getattr(inds,indicator['TYPE'])(self, **indicator['PARAMS']))

        else:
            for indicator in config['indicators']:
                self.indicators.append(getattr(inds,indicator['TYPE'])(self, **indicator['PARAMS']))

        self.DQN = DeepQNetwork(self)

        self.updatePlotPeriod = config['plotPeriod']
        self.__plotter = StockPlotter()
        self.__plotter.addPlot('Position')
        self.__plotter.addPlot('DQN')
        self.__plotter.updatePlot('DQN', '', 0,0, '#1A1A1D')
        self.__plotter.updatePlot('DQN', '', 0,100, '#1A1A1D')
        self.__plotter.updatePlot('DQN', 'Ave', 0,50, '#ff9b19')
        self.__plotter.updatePlot('DQN', 'Baseline', 0, 50, c=[.4,.4,.4])
        self.__plotter.updatePlot('DQN', 'Accuracy', 0, 50, '#2dedad')
        self.__plotter.addInfo()

        self.runningAccuracy = []
        self.runningActions = []
        self.times = []

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
            else: state.append(indicator.run(bar))

        if (np.array(state).ndim == 1):
            state = np.array([state])

        return isReady, state

    def onBars(self, bars):
        ''' Called when there is a new bar from the feed '''
        s = time.time()

        ##===================== Initial Setup =====================##
        bar = bars[self.__symbol]

        ## Check if you are within the desired time window
        if self.__startDate is not None and bar.getDateTime() < self.__startDate:
            return
        if self.__endDate is not None and bar.getDateTime() > self.__endDate:
            return

        self.__barsProcessed += 1

        ##===== Get state & check if all indicators are ready =====##
        isReady, state = self.getState(bar)

        if isReady: # If all indicators are ready
            ## Save initial price (for benchmark)
            if self.__initialPrice is None:
                self.__initialPrice = bar.getClose()

            ##============= Deep Q Network =============##

            self.DQN.moveToLTM(bar.getPrice())
            reward = self.DQN.moveToWM(state, bar.getPrice())
            doInvest = self.DQN.act(state)
            self.DQN.addToSTM(bar.getPrice(), state, doInvest)

            if config['train']:
                loss,acc = self.DQN.replay()
            else:
                loss,acc = [0.,0.]
            self.runningAccuracy.append(acc)
            self.runningActions.append(doInvest)

            if self.__barsProcessed % self.updatePlotPeriod*20 == 0:
                self.DQN.saveStrategy(self.params['indicators'] if config['loadModel'] else config['indicators'])

            if self.__barsProcessed % self.updatePlotPeriod == 0:
                print('Loss: {:3.2} Reward: {:.2} Accuracy: {:.1%} Buy: {:.1%} Bars Processed: {} Average Time: {:.2}ms              '.format( \
                    loss, \
                    reward, \
                    np.average(self.runningAccuracy), \
                    np.average(self.runningActions), \
                    self.__barsProcessed, \
                    np.average(self.times)*1000), \
                    end="\r")

                accXData = self.__plotter.plots['DQN']['Accuracy']['xData']
                accYData = self.__plotter.plots['DQN']['Accuracy']['yData']
                self.__plotter.plots['DQN']['Ave']['plot'].set_data(accXData, np.ones(len(accYData))*np.average(accYData[-min(20,len(accYData)-1):]))
                self.__plotter.updatePlot('DQN', 'Baseline', self.__barsProcessed, 50, c=[.4,.4,.4])
                self.__plotter.updatePlot('DQN', 'Accuracy', self.__barsProcessed, np.average(self.runningAccuracy)*100, '#2dedad')

                self.runningAccuracy = []
                self.runningActions = []
                self.times = []

            ##=============== Buy / Sell ===============##
            # print(bar.getDateTime(), pd.to_timedelta(str(bar.getDateTime()).split()[1]).seconds)
            if pd.to_timedelta(str(bar.getDateTime()).split()[1]).seconds%config['tradeInterval'] == 0:
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
        if self.__initialPrice is not None and self.__barsProcessed % self.updatePlotPeriod == 0:
            self.__plotter.updatePlot('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())
            self.__plotter.updatePlot('Position', 'Buy & Hold', bar.getDateTime(), bar.getClose()*1000000/self.__initialPrice, c=[.4,.4,.4])
            self.__plotter.updateInfo(self.getBroker().getEquity(), bar.getClose()*1000000/self.__initialPrice, self.DQN.epsilon)
            self.__plotter.redraw()
        else:
            self.times.append(time.time()-s)


    def hold(self):
        while True:
            self.__plotter.redraw()


##===== Symbol Definition =====##
if len(sys.argv) > 1:
    symbol = sys.argv[1]
else:
    symbol = config['symbol']
if len(sys.argv) > 2:
    index = sys.argv[2]
else:
    index = 0

##========= Load Data =========##
dataFilenames = glob.glob("../clean_data/{}*.csv".format(symbol))
if not len(dataFilenames): raise Exception('No data for symbol: '+symbol)
else: dataFilename = dataFilenames[index]

print('Reading data from: ',dataFilename)

feed = csvfeed.GenericBarFeed(barfeed.Frequency.MINUTE, timezone=None, maxLen=2048)
feed.addBarsFromCSV(symbol, dataFilename)

##==== Initialize Strategy ====##
backtest = SORIN(feed, symbol)
backtest.setStartDate()
backtest.setEndDate()

analyzers = ut.attachAnalyzers(backtest)
backtest.run()
ut.analyze(backtest, analyzers)


## Keeps figure ##
backtest.hold()
