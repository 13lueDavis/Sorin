import numpy as np
import pandas as pd

def isOnInterval(dt, hours=None, minutes=None):
    hourTruth = True
    minuteTruth = True
    if hours is not None:
        hourTruth = dt.hour in range(0,24+hours, hours)
    if minutes is not None:
        minuteTruth = dt.minute in range(0,60+minutes, minutes)
    return hourTruth and minuteTruth

def isAfterTime(dataDt, dt):
    return pd.to_timedelta(str(dataDt).split()[1]) >= pd.to_timedelta(dt)

def isBeforeTime(dataDt, dt):
    return pd.to_timedelta(str(dataDt).split()[1]) <= pd.to_timedelta(dt)
