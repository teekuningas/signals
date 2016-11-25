import sys

import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lib.fourier_ica import FourierICA
from lib.component import ComponentPlot
from lib.load import load_layout


MEG = False
DATA = '/home/zairex/Code/cibr/analysis/signals/data/ica_spectra/'
N_COMPONENTS = 13
PAGE = 13

raws = [
    mne.io.Raw(sys.argv[-2], preload=True, add_eeg_ref=False),
    mne.io.Raw(sys.argv[-1], preload=True, add_eeg_ref=False),
]

for raw in raws:
    raw.add_proj([], remove_existing=True)

if not MEG:
    # crop away eyes closed resting
    raws[0].crop(tmin=0, tmax=90)

# drop bad and non-data channels
for raw in raws:
    picks = mne.pick_types(raw.info, eeg=True, meg=True)
    raw.drop_channels([name for idx, name in enumerate(raw.info['ch_names'])
                       if idx not in picks])
    raw.drop_channels(raw.info['bads'])

raw = mne.concatenate_raws(raws)

if MEG:
    layout = None
else:
    layout = load_layout()

wsize = 4096
sfreq = raw.info['sfreq']
states = [(0, 85), (95, len(raw.times)/sfreq)]

# calculate fourier-ica
fica = FourierICA(wsize=wsize, n_components=N_COMPONENTS,
                  sfreq=sfreq, hpass=4, lpass=30,
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


def get_spectrum(idx, state):
    length_s = len(raw.times) / sfreq
    limit = [
        int(source_stft.shape[2] * (state[0] / length_s)),
        int(source_stft.shape[2] * (state[1] / length_s)),
    ]
    y = np.mean(10 * np.log10(np.abs(source_stft[idx, :, limit[0]:limit[1]])), axis=-1)
    x = fica.freqs
    return x, y


# plot power spectra of source components
current_row = 0
while True:

    rows = min(source_stft.shape[0] - current_row, PAGE)
    if rows <= 0:
        break

    fig_ = plt.figure()
    fig_.suptitle(str(current_row) + ' - ' + str(current_row + rows))
    for i in range(rows):
        for state in states:
            x, y = get_spectrum(current_row + i, state)
            axes = fig_.add_subplot(rows, 1, i+1)
            axes.plot(x, y)
    current_row += PAGE

# mock info
info = raw.info.copy()
info['chs'] = info['chs'][0:N_COMPONENTS]
info['ch_names'] = info['ch_names'][0:N_COMPONENTS]
info['nchan'] = N_COMPONENTS

current_row = 0
plots = []

while True:
    rows = min(source_stft.shape[0] - current_row, PAGE)
    if rows <= 0:
        break
    
    plots.append(ComponentPlot(source_stft, freqs, [], current_row, rows, 
        info, raw.last_samp - raw.first_samp))
    current_row += PAGE

# show everything
plt.show(block=False)

input_ = raw_input('Choose component: ')

data = []
header = [] 
name = sys.argv[-3]

for idx, state in enumerate(states):
    x, y = get_spectrum(int(input_) - 1, state)

    header.append("peak " + str(idx+1))
    data.append(np.max(y))
    
    header.append("average " + str(idx+1))
    data.append(np.mean(y[(freqs > 7) & (freqs < 13)]))
       
path = DATA + name + '.csv'
print "Saving to: " + path
np.savetxt(path, np.array([header, data]), fmt="%s", delimiter=',')
