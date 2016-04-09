import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne

from lib.stft import STFTPlot

# raw = mne.io.Raw('/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif', preload=True)
raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH016_EOEC-raw.fif', preload=True)

info = raw.info
sfreq = info['sfreq']
wsize = int(sfreq/2)
tstep = int(wsize/2)
freq_limit = 40
channels = np.array([
    11, # middle front Fz
    75, # middle back Oz
    108, # middle right T3
    45, # middle left T4
]) - 1
data = raw._data[channels]
tfr = mne.time_frequency.stft(data, wsize, tstep)

if __name__ == '__main__':
    stft = STFTPlot(tfr, wsize, sfreq, tstep, freq_limit)
