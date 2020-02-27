import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

COLOR = '#ffffff'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['figure.facecolor'] = '1A1A1D'
mpl.rcParams['axes.facecolor'] = '1A1A1D'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = '#222222'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#4286f4','#d615be','#11b231','#ff9b19','#2dedad','#b21111'])

class StockPlotter:
    def __init__(self):
        self.plots = {}

        self.fig = plt.figure(1, figsize=(15,6))
        self.fig.subplots_adjust(top=0.8) # or whatever
        plt.grid(axis="x")
        plt.autoscale(True)

        self.fig.canvas.mpl_connect('button_press_event', self.onMouseDown)
        self.fig.canvas.mpl_connect('button_release_event', self.onMouseUp)
        self.fig.canvas.mpl_connect('key_press_event', self.onEsc)

        self.mouseX = None

    def onMouseDown(self, event):
        self.mouseX = float(event.xdata)

    def onMouseUp(self, event):
        topPlot = self.plots[list(self.plots.keys())[0]]['subplot']
        if self.mouseX is not None:
            newX = float(event.xdata)
            print(self.mouseX, newX, type(self.mouseX), type(newX))
            for plot in self.plots.values():
                plot['subplot'].set_xlim([self.mouseX, newX])
                maxY = max([max(line.get_ydata()) for line in plot['subplot'].lines if len(line.get_ydata())])
                minY = max([min(line.get_ydata()) for line in plot['subplot'].lines if len(line.get_ydata())])
                plot['subplot'].set_ylim([minY, maxY])

            self.mouseX = None

    def onEsc(self, event):
        if event.key == 'escape':
            for plot in self.plots.values():
                # plot['subplot'].set_xlim(auto=True)
                # plot['subplot'].set_ylim(auto=True)
                maxY = max([max(line.get_ydata()) for line in plot['subplot'].lines if len(line.get_ydata())])
                minY = min([min(line.get_ydata()) for line in plot['subplot'].lines if len(line.get_ydata())])
                maxX = max([max(line.get_xdata()) for line in plot['subplot'].lines if len(line.get_xdata())])
                minX = min([min(line.get_xdata()) for line in plot['subplot'].lines if len(line.get_xdata())])
                plot['subplot'].set_ylim([minY, maxY])
                plot['subplot'].set_xlim([minX, maxX])


    def addPlot(self, plotName):

        for i, name in enumerate(self.plots.keys()):
            self.plots[name]['subplot'] = plt.subplot(len(self.plots.keys())+1, 1, i+1)

        self.plots[plotName] = {}
        self.plots[plotName]['subplot'] = plt.subplot(len(self.plots.keys()), 1, len(self.plots.keys()))

    def addDataPlot(self, plotName, dataName, x, y, c=None):
        self.plots[plotName][dataName] = {
            "xData" : [],
            "yData" : [],
            "xBuyData" : [],
            "yBuyData" : [],
            "xSellData" : [],
            "ySellData" : []
        }

        if c is not None:
            self.plots[plotName][dataName]["plot"], = self.plots[plotName]['subplot'].plot([x], [y], label=dataName, c=c)
        else:
            self.plots[plotName][dataName]["plot"], = self.plots[plotName]['subplot'].plot([x], [y], label=dataName)
        self.plots[plotName][dataName]["buyPlot"], = self.plots[plotName]['subplot'].plot([], [], marker="o", ls="", ms=0.9, mec='#00AE28')
        self.plots[plotName][dataName]["sellPlot"], = self.plots[plotName]['subplot'].plot([], [], marker="o", ls="", ms=0.9, mec='#F45531')

        self.plots[plotName]['subplot'].legend(bbox_to_anchor=(1, 1), loc='upper left')

    def addInfo(self):
        topPlot = self.plots[list(self.plots.keys())[0]]['subplot']
        self.returns = topPlot.annotate("0%", xy=(0.05, 1.14), xycoords='axes fraction', fontsize='xx-large', color="#00AE28")
        topPlot.annotate('Returns', xy=(0.05, 1.04), xycoords='axes fraction', fontsize='medium', color='#808080')

        self.capital = topPlot.annotate('$1,000,000', xy=(0.2, 1.14), xycoords='axes fraction', fontsize='xx-large', color="#00AE28")
        topPlot.annotate('capital', xy=(0.2, 1.04), xycoords='axes fraction', fontsize='medium', color='#808080')

        self.benchmarkReturns = topPlot.annotate("0%", xy=(0.45, 1.14), xycoords='axes fraction', fontsize='xx-large', color="#bfbdbd")
        topPlot.annotate('Returns', xy=(0.45, 1.04), xycoords='axes fraction', fontsize='medium', color='#808080')

        self.benchmarkcapital = topPlot.annotate('$1,000,000', xy=(0.6, 1.14), xycoords='axes fraction', fontsize='xx-large', color="#bfbdbd")
        topPlot.annotate('capital', xy=(0.6, 1.04), xycoords='axes fraction', fontsize='medium', color='#808080')

        self.epsilon = topPlot.annotate('100%', xy=(0.85, 1.14), xycoords='axes fraction', fontsize='xx-large', color="#2dedad")
        topPlot.annotate('Exploration', xy=(0.85, 1.04), xycoords='axes fraction', fontsize='medium', color='#808080')

    def updatePlot(self, plotName, dataName, x=None, y=None, c=None):
        if plotName not in self.plots.keys():
            raise Exception('No plot by name ', plotName)

        # If only 1 piece of data is give, assume its y
        if y is None:
            y = x
            x = None

        if dataName not in self.plots[plotName].keys():
            self.addDataPlot(plotName, dataName, x, y, c)

        self.plots[plotName][dataName]['xData'].append(x)
        self.plots[plotName][dataName]['yData'].append(y)
        if x is None:
            self.plots[plotName][dataName]['xData'] = np.arange(len(self.plots[plotName][dataName]['yData'])).tolist()
        self.plots[plotName][dataName]['plot'].set_data(self.plots[plotName][dataName]['xData'], self.plots[plotName][dataName]['yData'])

        self.plots[plotName]['subplot'].relim()
        self.plots[plotName]['subplot'].autoscale_view()


    def addBuyPoint(self, plotName, dataName, x=None, y=None):
        if plotName not in self.plots.keys():
            raise Exception('No plot by name '+plotName)

        if dataName not in self.plots[plotName].keys():
            self.addDataPlot(plotName, dataName, x, y)

        self.plots[plotName][dataName]['xBuyData'].append(x)
        self.plots[plotName][dataName]['yBuyData'].append(y)

        self.plots[plotName][dataName]["buyPlot"].set_data(self.plots[plotName][dataName]['xBuyData'], self.plots[plotName][dataName]['yBuyData'])

    def addSellPoint(self, plotName, dataName, x=None, y=None):
        if plotName not in self.plots.keys():
            raise Exception('No plot by name '+plotName)

        if dataName not in self.plots[plotName].keys():
            self.addDataPlot(plotName, dataName, x, y)

        self.plots[plotName][dataName]['xSellData'].append(x)
        self.plots[plotName][dataName]['ySellData'].append(y)

        self.plots[plotName][dataName]["sellPlot"].set_data(self.plots[plotName][dataName]['xSellData'], self.plots[plotName][dataName]['ySellData'])


    def updateInfo(self, capital, benchmark=None, epsilon=None):
        returns = (capital/10000)-100
        self.returns.set_text(str(np.round(returns,2))+"%")
        self.capital.set_text('$'+f'{round(capital,2):,}')

        if benchmark is not None:
            benchmarkReturns = (benchmark/10000)-100
            self.benchmarkReturns.set_text(str(np.round(benchmarkReturns,2))+"%")
            self.benchmarkcapital.set_text('$'+f'{round(benchmark,2):,}')

        if epsilon is not None:
            self.epsilon.set_text(str(round(epsilon*100))+'%')

        if returns < 0:
            self.returns.set_color('#F45531')
            self.capital.set_color('#F45531')
        else:
            self.returns.set_color('#00AE28')
            self.capital.set_color('#00AE28')

    def redraw(self):
        plt.draw()
        plt.pause(0.00001)
