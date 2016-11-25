import sys

import numpy as np
import matplotlib.pyplot as plt

import mne

fnames = sys.argv[1:]

raws = [mne.io.Raw(fname, preload=True) for fname in fnames]
raw = mne.concatenate_raws(raws)

PLOTS = 1

picks = mne.pick_types(raw.info, meg=False, eog=True)

eog_data = raw._data[picks][0]

events = mne.find_events(raw, shortest_event=1)

event_data = np.zeros((len(eog_data),))
for event in events:
    if event[1] == 0:
        event_data[event[0] - raw.first_samp] = np.min(eog_data) 

for i in range(PLOTS):
    plt.subplot(PLOTS, 1, i+1)
    start = i*len(event_data) / PLOTS
    end = (i+1)*len(event_data) / PLOTS - 1
    plt.plot(np.arange(start, end)/raw.info['sfreq'], eog_data[start:end])
    plt.plot(np.arange(start, end)/raw.info['sfreq'], event_data[start:end])

plt.show()
