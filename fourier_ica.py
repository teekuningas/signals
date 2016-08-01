import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lib.fourier_ica import FourierICA
from lib.stft import STFTPlot
from lib.utils import filter_triggers

raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH001_MED-raw.fif', preload=True)

layout_fname = 'gsn_129.lout'
layout_path = 'data/'
layout = mne.channels.read_layout(layout_fname, layout_path)

wsize = 2048
n_components = 5
radius = 15
sfreq = raw.info['sfreq']

# find triggers
triggers = mne.find_events(raw)[:, 0]
triggers = filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp, radius*sfreq)

# drop bad and non-data channels
raw.drop_channels(raw.info['ch_names'][128:] + raw.info['bads'])

# calculate fourier-ica
fica = FourierICA(wsize=wsize, n_components=n_components,
                  sfreq=raw.info['sfreq'], hpass=4, lpass=30)
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
fig_ = plt.figure()
for i in range(source_stft.shape[0]):
    y = np.mean(np.abs(source_stft[i]), axis=-1)
    x = fica.freqs
    axes = fig_.add_subplot(source_stft.shape[0], 1, i+1)
    axes.plot(x, y)

# mock info
info = raw.info.copy()
info['chs'] = info['chs'][0:5]
info['ch_names'] = info['ch_names'][0:5]
info['nchan'] = 5

import pdb; pdb.set_trace()

fig_ = plt.figure()
position = 0
window = 30

def update_tfr_plot(position):

    fig_.clear()
    for i in range(source_stft.shape[0]):
        times = np.arange(window * position, window * (position + 1), 1.0)

        # select subset of data
        data = np.abs(source_stft[:, :, window*position:window*(position+1)])
        
        # pad if necessary
        data = np.pad(data, ((0,0), (0,0), (0, window - data.shape[2])), 
                      constant_values=0, mode='constant')

        # use mne's AverageTFR for plotting
	tfr_ = mne.time_frequency.AverageTFR(info, data, times, freqs, 1)
	axes = fig_.add_subplot(source_stft.shape[0], 1, i + 1)
	tfr_.plot(picks=[i], axes=axes, show=False, mode='logratio')

        # add triggers
        print triggers
        for trigger in triggers:
            if (trigger >= window*(position+1)*sfreq 
                    or trigger < window*position*sfreq):
                continue
            x = trigger - window*position
            height = axes.get_ylim()[1] - axes.get_ylim()[0]
            width = float(axes.get_xlim()[1] - axes.get_xlim()[0]) / 50
            axes.add_patch(
                patches.Rectangle((x - width / 2, 0.0), width, height)
            )

    plt.draw()

def on_key_release(event):
    global position
    if event.key == 'left' and position != 0:
        position -= 1
    elif event.key == 'right' and position < source_stft.shape[2] / window:
        position += 1

    update_tfr_plot(position)

update_tfr_plot(position)

cid = fig_.canvas.mpl_connect('key_release_event', on_key_release)

# show everything
plt.show()
