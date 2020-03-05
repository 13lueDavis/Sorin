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
import numpy as np
import glob
import pandas as pd

from lib.stockPlotter import StockPlotter
from lib.indicators import *

class Backtest(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(Backtest, self).__init__(feed)

        self.__symbol = instrument
        self.position = None
        self.__startDate = None
        self.__endDate = None
        self.__barsProcessed = 0
        self.__initialPrice = None

        self.getBroker().getFillStrategy().setVolumeLimit(None)
        self.prices = feed[instrument].getPriceDataSeries()

        self.__indicator = PeakTrough(self, interval=30)

        self.updatePlotPeriod = 2
        self.__plotter = StockPlotter()
        self.__plotter.addPlot('Position')
        self.__plotter.addInfo()


    def setStartDate(self, date=None):
        if date is not None:
            self.__startDate = pd.to_datetime(date, infer_datetime_format=True)

    def setEndDate(self, date=None):
        if date is not None:
            self.__endDate = pd.to_datetime(date, infer_datetime_format=True)

    def onEnterCanceled(self, position):
        self.position = None

    def onExitOk(self, position):
        self.position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.position.exitMarket()

    def onBars(self, bars):
        bar = bars[self.__symbol]
        if self.__startDate is not None and bar.getDateTime() < self.__startDate:
            return
        if self.__endDate is not None and bar.getDateTime() > self.__endDate:
            return

        self.__barsProcessed += 1

        # Save initial price for benchmark
        if self.__initialPrice is None:
            self.__initialPrice = bar.getClose()

        #==============================================================================#
        #                                  Strategy                                    #
        #==============================================================================#



        if self.__indicator.isReady(bar): # If indicator is ready
            ##============= Deep Q Network =============##
            doInvest = self.__indicator.run(bar)
            # print(doInvest)

            ##=============== Buy / Sell ===============##
            ## Buy
            if doInvest and self.position is None:
                shares = int(self.getBroker().getCash() * 0.9 / bar.getPrice())
                self.position = self.enterLong(self.__symbol, shares, True)
                self.__plotter.addBuyPoint('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())

            ## Sell
            if not doInvest and self.position is not None and not self.position.exitActive():
                self.position.exitMarket()
                self.__plotter.addSellPoint('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())

        ## Plot
        if self.__barsProcessed % self.updatePlotPeriod == 0:
            self.__plotter.updatePlot('Position', 'Portfolio', bar.getDateTime(), self.getBroker().getEquity())
            self.__plotter.updatePlot('Position', 'Buy & Hold', bar.getDateTime(), bar.getClose()*1000000/self.__initialPrice, c=[.4,.4,.4])
            self.__plotter.updateInfo(self.getBroker().getEquity(), bar.getClose()*1000000/self.__initialPrice)
            self.__plotter.redraw()

    def hold(self):
        while True:
            self.__plotter.redraw()


##===== Symbol Definition =====##
symbol = 'MO'

##========= Load Data =========##
dataFilename = glob.glob("../clean_data/{}*.csv".format(symbol))[0]
feed = csvfeed.GenericBarFeed(barfeed.Frequency.MINUTE, timezone=None, maxLen=1024)
feed.addBarsFromCSV(symbol, dataFilename)

##==== Initialize Strategy ====##
backtest = Backtest(feed, symbol)
backtest.setStartDate('08-22-2019')
backtest.setEndDate('9-01-2019')

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
