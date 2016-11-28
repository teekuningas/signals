import pickle
import sys
import os

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.cluster import cluster_components as cluster_matrix
from lib.load import load_layout
from lib.load import get_raw
from lib.component import ComponentPlot
from lib.abstract import ComponentData

MEG = False
BAND = [1, 20]
COMPONENTS = 14
WSIZE = 2048

PATH = '/home/zairex/Code/cibr/analysis/signals/data/fica/'


def _create_image(topo, layout, info, size=32):
    image = np.zeros((size, size))
    for i in range(len(topo)):
        location = layout.pos[layout.names.index(info['ch_names'][i])]
        x = int(location[0] * 32)
        y = int(location[1] * 32)
        image[x, y] += topo[i]

    return image


def _create_topo(sensor_stft, layout, info):

    # calculate "images"
    topo = np.sum(np.abs(sensor_stft), axis=(1, 2))

    # normalize
    topo = topo / np.linalg.norm(topo)

    # create comparable images
    image = _create_image(topo, layout, info)

    return image


def get_components(fica, info, length, layout):
    source_stft = fica.source_stft

    components = []
    for i in range(COMPONENTS):
        source_component = source_stft[i] 
        sensor_stft = fica.component_in_sensor_space(i)
        sensor_topo = _create_topo(sensor_stft, layout, info)
        source_psd = np.mean(10 * np.log10(np.abs(source_stft[i])), axis=-1)
        component_data = ComponentData(
            source_stft=source_component,
            sensor_stft=sensor_stft,
            sensor_topo=sensor_topo,
            source_psd=source_psd,
            freqs=fica.freqs,
            length=length,
            info=info.copy(),
        )
        components.append(component_data)
    return components


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


def main():

    mne.utils.set_log_level('ERROR')

    if MEG:
        layout = None
    else:
        layout = load_layout()

    print "Reading and processing data from files.."

    filenames = sys.argv[1:]

    new_components = []

    for idx, fname in enumerate(filenames):
        print "Handling " + str(idx+1) + ". subject"
        raw = mne.io.read_raw_fif(fname, preload=True)

        picks = mne.pick_types(raw.info, eeg=True, meg=True)
        raw.drop_channels([ch_name for ix, ch_name in 
            enumerate(raw.info['ch_names']) if ix not in picks])
        raw.drop_channels(raw.info['bads'])

        fica = FourierICA(wsize=WSIZE, n_components=COMPONENTS, maxiter=7000,
                          sfreq=raw.info['sfreq'], hpass=BAND[0], lpass=BAND[1])
        fica.fit(raw._data)

        components = get_components(fica, raw.info, len(raw.times), layout)

        plot_components(components, layout)
        input_ = int(raw_input("Which component to use: ")) - 1
        if input_ == -1:
            continue
        new_components.append(components[input_])

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
