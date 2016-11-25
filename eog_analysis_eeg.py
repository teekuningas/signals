import sys

import numpy as np
import matplotlib.pyplot as plt

import mne

from lib.load import cli_raws

PLOTS = 6

raw = cli_raws()[0]

raw.filter(l_freq=1, h_freq=100)

ica = mne.preprocessing.ICA(n_components=0.9, method='fastica')
ica.fit(raw)

sources = ica.get_sources(raw)

# alter amplitudes to get better plot
for source in sources._data:
    for idx, amplitude in enumerate(source):
	source[idx] = amplitude / 5000.0

sources.plot()

index = raw_input("Please enter index of blink channel: ")

index = int(index) - 1

eog_data = sources._data[index]

events = mne.find_events(raw)

event_data = np.zeros((len(eog_data),))
for event in events:
    if event[1] == 0:
        event_data[event[0] - raw.first_samp] = np.min(eog_data) 

for i in range(PLOTS):
    plt.subplot(PLOTS, 1, i+1)
    start = i*len(event_data) / PLOTS
    end = (i+1)*len(event_data) / PLOTS - 1
    plt.plot(np.arange(start, end)/raw.info['sfreq'], eog_data[start:end])
    plt.plot(np.arange(start, end)/raw.info['sfreq'], event_data[start:end], color='r')

plt.show()
