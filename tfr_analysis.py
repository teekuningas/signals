import sys

import mne
import numpy as np
import matplotlib.pyplot as plt

from lib.load import cli_raws
from lib.load import get_raw
from lib.load import load_layout

# front, back, right, left
channels = [11, 75, 108, 45]

# eoec_raw = get_raw('KH002', 'eoec', clean_channels=True)
# med_raw = get_raw('KH002', 'med', clean_channels=True)

# raw = eoec_raw
# raw.append(med_raw)

raws = cli_raws()

if not raws:
    raws = [get_raw('KH002', 'med')]

raw = raws[0]
raw.append(raws[1:])

picks = mne.pick_types(raw.info, eeg=True, meg=True)
raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names'])
		   if idx not in picks])

args = sys.argv
if '--threshold' in args:
    threshold = float([args[idx+1] for idx, arg in enumerate(args) 
                 if arg == '--threshold'][0])
else:
    threshold = 0.05

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

