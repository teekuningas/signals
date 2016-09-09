import mne
import numpy as np
import matplotlib.pyplot as plt

from lib.load import get_raw

# front, back, right, left
channels = [11, 75, 108, 45]

layout_path = '/home/zairex/Code/cibr/materials/'
layout_filename = 'gsn_129.lout'
layout = mne.channels.read_layout(layout_filename, layout_path)

eoec_raw = get_raw('KH002', 'eoec', clean_channels=True)
med_raw = get_raw('KH002', 'med', clean_channels=True)

threshold = 0.02

raw = eoec_raw
raw.append(med_raw)

wsize = 4096
data = raw._data

stft = np.log(1 + np.abs(mne.time_frequency.stft(data, wsize)))
freqs = mne.time_frequency.stftfreq(wsize, raw.info['sfreq'])

# standardize
max_ = np.max(stft)
min_ = np.min(stft)
stft = (stft - min_) / (max_ - min_)

# clip
stft = np.clip(stft, 0, threshold)

stft = stft - threshold/2

tfr = mne.time_frequency.AverageTFR(
    raw.info, stft, np.arange(stft.shape[2], dtype=np.float64), freqs, 1
)

picks = mne.pick_types(raw.info, eeg=True)

for channel in channels:
    tfr.plot(picks=[channel], title=str(channel), fmin=1, fmax=100, show=False)

plt.show()

