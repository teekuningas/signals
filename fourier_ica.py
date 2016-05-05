import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.stft import STFTPlot


raw = mne.io.Raw('data/eoec-raw.fif', preload=True)

layout_fname = 'gsn_129.lout'
layout_path = 'data/'
layout = mne.channels.read_layout(layout_fname, layout_path)

wsize = 2000
n_components = 5
sfreq = raw.info['sfreq']

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

# plot tfr's of source components
plot_ = STFTPlot(freqs, source_stft, window_width=50, 
                 window_height=n_components,
                 ch_names=[str(i+1) for i in range(len(source_stft))])

# show everything
plt.show()
