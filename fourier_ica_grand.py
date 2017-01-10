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
from lib.abstract import ComponentData
from lib.abstract import plot_components

MEG = False
BAND = [3, 17]
COMPONENTS = 15
WSIZE = 1024

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


def get_components(fica, info, length, layout, events):
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
            events=events
        )
        components.append(component_data)
    return components


def main():

    mne.utils.set_log_level('ERROR')

    layout = load_layout(MEG)

    print "Reading and processing data from files.."

    filenames = sys.argv[1:]

    new_components = []

    for idx, fname in enumerate(filenames):
        try:
            plt.close('all')
            print "Handling " + str(idx+1) + ". subject"
            raw = mne.io.read_raw_fif(fname, preload=True)

            try:
                events = mne.find_events(raw)
            except:
                events = []

            picks = mne.pick_types(raw.info, eeg=True, meg='grad')
            raw.drop_channels([ch_name for ix, ch_name in 
                enumerate(raw.info['ch_names']) if ix not in picks])
            raw.drop_channels(raw.info['bads'])

            fica = FourierICA(wsize=WSIZE, n_components=COMPONENTS, maxiter=7000, conveps=1e-10,
                              sfreq=raw.info['sfreq'], hpass=BAND[0], lpass=BAND[1])
            fica.fit(raw._data)

            components = get_components(fica, raw.info, len(raw.times), layout, events)

            handle = plot_components(components, layout, title='Fourier-ICA components from one subject')
            
            # input_ = raw_input("Components: ")
            # selections = [int(val) for val in input_.split(' ')]
            # handle = plot_components(list(np.array(components)[selections]), layout, title='Fourier-ICA components from one subject')

            input_ = int(raw_input("Which component to use: "))
            if input_ == -1:
                continue
            new_components.append(components[input_ - 1])
        except:
            import pdb; pdb.set_trace()
            print "Smoething went wrong."

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
