import mne
import numpy as np

freqs = np.arange(4, 25, 4)
raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/meditaatio/KH004_MED-raw.fif', preload=True)
data = raw._data[np.newaxis, ...]
info = raw.info

power, itc = mne.time_frequency.tfr._induced_power_cwt(data, 
                                                       sfreq=info['sfreq'],
                                                       frequencies=freqs)
times = raw.times.copy()
nave = len(data)
tfr = mne.time_frequency.tfr.AverageTFR(info, power, times, freqs, nave, method='morlet-power')

tfr.plot(picks=[120], vmin=0, vmax=2e-8, cmap='seismic')

import pdb; pdb.set_trace()
print "miau"
