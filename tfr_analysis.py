import sys

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage.filters import convolve

from lib.load import cli_raws
from lib.load import get_raw
from lib.load import load_layout

from lib.utils import filter_triggers

# get data
raws = cli_raws()

if not raws:
    raws = [get_raw('KH003', 'eoec'), get_raw('KH003', 'med')]

for raw in raws:
    picks = mne.pick_types(raw.info, eeg=True, meg=True)
    raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names'])
                       if idx not in picks])

raw = raws[0]
raw.append(raws[1:])

# set triggers
triggers = [95000, 205000]

# get threshold parameter
args = sys.argv
if '--threshold' in args:
    threshold = float([args[idx+1] for idx, arg in enumerate(args) 
                       if arg == '--threshold'][0])
else:
    threshold = 0.05

if '--channel' in args:
    channels = [int([args[idx+1] for idx, arg in enumerate(args) 
                    if arg == '--channel'][0])]
else:
    # front, back, right, left
    channels = [10, 71, 107, 44]

wsize = 32768

stft = np.power(np.abs(mne.time_frequency.stft(raw._data, wsize)), 2)
freqs = mne.time_frequency.stftfreq(wsize, raw.info['sfreq'])

stft = convolve(stft, np.ones((1, 51, 3))/(1*51*3))

length_samples = stft.shape[2]
length_s = (raw.last_samp - raw.first_samp) / raw.info['sfreq']
scale = length_s / float(length_samples)

times = np.arange(0, length_samples, 1.0) * scale

tfr = mne.time_frequency.AverageTFR(
    raw.info, stft, times, freqs, 1
)

for channel in channels:
    fig_ = plt.figure()
    axes = fig_.add_subplot(1, 1, 1)
    tfr.plot(picks=[channel], title=str(channel), fmin=1, fmax=40, 
             show=False, axes=axes, mode='logratio', baseline=(0, 70))

    # plot triggers
    for trigger in triggers:
        axes.add_patch(
            patches.Rectangle((trigger - 10, 0.0), 20, 1000)
        )

plt.show()

