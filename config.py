config = {
    'symbol' : 'TSLA',
    'plotPeriod' : 400,

    'loadModel' : True,
    'loadName' : 'SORIN_2',
    'useLoadedParams' : True,

    'saveAsNew' : False,
    'saveName' : '',

    'train' : False,

    'batchSize' : 16,
    'memoryLength' : 10000,
    'gamma' : 0.9,
    'epsilon' : 1.0,
    'epsilon_min' : 0.01,
    'epsilon_decay' : 1-1e-5,
    'learning_rate' : 8e-5,
    'tau' : .125,

    'tradeInterval' : 1,

    'indicators' : [
        {
            'TYPE' : 'Ichimoku',
            'PARAMS' : dict()
        },
        {
            'TYPE' : 'Momentum',
            'PARAMS' : dict(
                period=500
            )
        },
        {
            'TYPE' : 'Momentum',
            'PARAMS' : dict(
                period=100
            )
        },
        {
            'TYPE' : 'Momentum',
            'PARAMS' : dict(
                period=10
            )
        },
        {
            'TYPE' : 'PeakTrough',
            'PARAMS' : dict(
                interval=30
           )
        },
        {
            'TYPE' : 'EMACross',
            'PARAMS' : dict(
                shortPeriod=9,
                longPeriod=21
            )
        },
        {
            'TYPE' : 'EMACross',
            'PARAMS' : dict(
                shortPeriod=20,
                longPeriod=50
            )
        },
        {
            'TYPE' : 'RSI',
            'PARAMS' : dict(
                entrySMA=200,
                exitSMA=5,
                rsiPeriod=2
            )
        }
    ]

}
