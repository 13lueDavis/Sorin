from lib.stockPlotter import StockPlotter
import numpy as np

plotter = StockPlotter()
plotter.addPlot('Position')
plotter.addPlot('DQN')
plotter.addInfo()

for i in range(10000):
    plotter.updatePlot('Position', 'Portfolio', i, np.random.rand())
    plotter.updatePlot('Position', 'Benchmark', i, np.random.rand())
    plotter.updatePlot('DQN', 'Benchmark', np.random.rand())
    plotter.redraw()
