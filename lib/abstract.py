import matplotlib.pyplot as plt

import mne
import numpy as np

from lib.component import ComponentPlot


class ComponentData(object):
    """
    """
    def __init__(self, source_stft, sensor_stft, sensor_topo, source_psd, freqs, length, info):
        self.source_stft = source_stft
        self.sensor_stft = sensor_stft
        self.source_psd = source_psd
        self.sensor_topo = sensor_topo
        self.freqs = freqs
        self.info = info
        self.length = length


def plot_components(components, layout):

    # create figure for head topographies
    fig_ = plt.figure()
    for i, component in enumerate(components):
        sensor_component = np.abs(component.sensor_stft)
        tfr_ = mne.time_frequency.AverageTFR(component.info, sensor_component,
            range(sensor_component.shape[2]), component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

    # create figure for psd
    fig_ = plt.figure()
    for i, component in enumerate(components):
        y = component.source_psd
        x = component.freqs
        axes = fig_.add_subplot(len(components), 1, i + 1)
        axes.plot(x, y)

    # create figure ica components
    len_ = min([component.source_stft.shape[1] for component in components])
    source_stft = np.array([component.source_stft[:, 0:len_]
                            for component in components])

    component = components[0]
    freqs = component.freqs
    length = component.length

    info = component.info.copy()
    info['chs'] = info['chs'][0:len(components)]
    info['ch_names'] = info['ch_names'][0:len(components)]
    info['nchan'] = len(components)

    cp = ComponentPlot(source_stft, freqs, [], 0, len(components), info, length)

    plt.show(block=False)

    return cp

