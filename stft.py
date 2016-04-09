import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne

from lib.stft import STFTPlot

# raw = mne.io.Raw('/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif', preload=True)
raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH009_EOEC-pre.fif', preload=True)

info = raw.info
sfreq = info['sfreq']
wsize = 2000
tstep = int(wsize/2)
hpass, lpass = 4, 20
channels = np.array([
    11, # middle front Fz
    75, # middle back Oz
    108, # middle right T3
    45, # middle left T4
]) - 1
data = raw._data[channels]
tfr = mne.time_frequency.stft(data, wsize, tstep)
freqs = mne.time_frequency.stftfreq(wsize, sfreq)

# bandpass filter
hpass_idx = min(np.where(freqs >= hpass)[0])
lpass_idx = max(np.where(freqs <= lpass)[0])
freqs = freqs[hpass_idx:lpass_idx]
tfr = tfr[:, hpass_idx:lpass_idx, :]

STFTPlot(freqs, tfr)
