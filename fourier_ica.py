import sys

import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lib.fourier_ica import FourierICA
from lib.component import ComponentPlot

MEG = False

raw = mne.io.Raw(sys.argv[-1], preload=True)

if MEG:
    layout = None
else:
    layout_fname = 'gsn_129.lout'
    layout_path = '/home/zairex/Code/cibr/materials/'
    layout = mne.channels.read_layout(layout_fname, layout_path)

wsize = 8192
n_components = 8
page = 8

# find triggers
try:
    triggers = [event[0] for event in mne.find_events(raw) if event[1] == 0]
except:
    print "No triggers found"
    triggers = []

# drop bad and non-data channels
picks = mne.pick_types(raw.info, eeg=True, meg=True)
raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names'])
                       if idx not in picks])
raw.drop_channels(raw.info['bads'])

# calculate fourier-ica
fica = FourierICA(wsize=wsize, n_components=n_components,
                  sfreq=raw.info['sfreq'], hpass=5, lpass=30,
                  maxiter=7000)
fica.fit(raw._data[:, raw.first_samp:raw.last_samp])

source_stft = fica.source_stft
freqs = fica.freqs

# plot components in sensor space on head topographies
fig_ = plt.figure()
for i in range(source_stft.shape[0]):
    sensor_component = np.abs(fica.component_in_sensor_space(i))
    tfr_ = mne.time_frequency.AverageTFR(raw.info, sensor_component, 
        range(sensor_component.shape[2]), freqs, 1)

    axes = fig_.add_subplot(source_stft.shape[0], 1, i+1)
    mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

# plot power spectra of source components
current_row = 0
while True:

    rows = min(source_stft.shape[0] - current_row, page)
    if rows <= 0:
        break

    fig_ = plt.figure()
    fig_.suptitle(str(current_row) + ' - ' + str(current_row + rows))
    for i in range(rows):
        y = np.mean(np.power(np.abs(source_stft[current_row + i]), 2), axis=-1)
        x = fica.freqs
        axes = fig_.add_subplot(rows, 1, i+1)
        axes.plot(x, y)
    current_row += page

# mock info
info = raw.info.copy()
info['chs'] = info['chs'][0:n_components]
info['ch_names'] = info['ch_names'][0:n_components]
info['nchan'] = n_components

current_row = 0
plots = []
while True:
    rows = min(source_stft.shape[0] - current_row, page)
    if rows <= 0:
        break
    
    plots.append(ComponentPlot(source_stft, freqs, triggers, current_row, rows, 
        info, raw.last_samp - raw.first_samp))
    current_row += page

# show everything
plt.show()
