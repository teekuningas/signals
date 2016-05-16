import mne

import numpy as np

from lib.tse import PlottableTriggers
from lib.tse import PlottableTSE
from lib.tse import TSEPlot
from lib.tse import create_tse_data


FILES = {
    'KH001':
        {
            'med': '/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH001_MED-raw.fif',
            'rest': '/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH001_EOEC-raw.fif'
        },
    'KH002':
        {
            'med': '/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH002_MED-raw.fif',
            'rest': '/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH002_EOEC-raw.fif'
        },
 
    }

CHANNELS = {
    'Fz': 11, # middle front 
    'Oz': 75, # middle back
    'T3': 108, # middle right
    'T4': 45, # middle left
}

BANDS = {
    'alpha': (8, 12),
    'beta': (12, 20),
    'theta': (4, 8),
    'delta': (1, 4),
}


if __name__ == '__main__':
    fname = FILES['KH001']['med']
    interval = (-24000, 8000)
    raw = mne.io.Raw(fname, preload=True)
    data = raw._data
    band = BANDS['alpha']

    times = mne.find_events(raw)[:, 0]

    tse_oz = []
    for time in times:
        if time > -interval[0] and time < data.shape[1] - interval[1]:
            epoch = data[CHANNELS['Oz'], 
                         (interval[0] + time):(interval[1] + time)]
            tse_oz.append(create_tse_data(epoch, raw.info['sfreq'], band))

    tses = [
        PlottableTSE(np.average(tse_oz, axis=0), 
                     color='b', 
                     title='Alpha Oz'),
    ]


    tse_length = interval[1] - interval[0]

    plottables = [
	{
	    'title': "Focused attention TSE on occpital cortex",
	    'tse': tses,
            'trigger': [],
	    'unix': True,
	},
    ]
    TSEPlot(plottables, window=tse_length-1)

    print "kissa"


