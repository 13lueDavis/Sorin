import pandas as pd
import numpy as np
import glob
import sys
import datetime

symbol = sys.argv[1]

for filename in glob.glob("../raw_data/{}*.csv".format(symbol)):
    print(filename)
    data = pd.read_csv(filename)
    print('Size: ', data.shape[0], ' datapoints')

    data = data.rename(columns={'Local time' : 'Date Time'})
    data['Date Time'] = data['Date Time'].apply(lambda d: datetime.datetime.strptime(d[:-14], '%d.%m.%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S'))
    data = data.set_index('Date Time')

    def isTrading(data):
        truth = np.array([dp.weekday() not in [5,6] for dp in data])
        truth = np.vstack((truth, data.hour < 14))
        truth = np.vstack((truth, np.vstack((data.hour > 7, np.vstack((data.hour == 7, data.minute >= 30)).all(axis=0))).any(axis=0)))

        return truth.all(axis=0)

    dateTimes = pd.to_datetime(data.index)
    tradingIdxs = isTrading(dateTimes)

    tradingData = data[tradingIdxs]
    volumeIdxs = tradingData['Volume'] > 0

    cleanData = tradingData[volumeIdxs]
    cleanData.loc[:,'Volume'] *= 1000000

    print('Total Datapoints: ', data.shape[0], ' datapoints')

    cleanData.to_csv('../clean_data/'+filename.split('/')[2])
