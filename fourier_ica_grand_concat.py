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

BAND = [3, 17]
COMPONENTS = 15
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


def main():

    mne.utils.set_log_level('ERROR')

    layout = load_layout(MEG=False)

    print "Reading and processing data from files.."

    filenames = sorted(sys.argv[1:])

    new_components = []

    for idx in range(0, len(filenames), 2):
        try:
            plt.close('all')

            print "Handling " + str(idx / 2 + 1) + ". subject"
            print "EOEC filename: ", filenames[idx+1]
            print "MED filename: ", filenames[idx]

            # eoec
            eoec_raw = mne.io.read_raw_fif(filenames[idx+1], preload=True)
            eoec_raw.crop(tmin=0, tmax=85)
            picks = mne.pick_types(eoec_raw.info, eeg=True)
            chns = [ch_name for ix, ch_name in enumerate(eoec_raw.info['ch_names'])
                    if ix not in picks]
            eoec_raw.drop_channels(chns)
            eoec_raw.add_proj([], remove_existing=True)
            # raw
            med_raw = mne.io.read_raw_fif(filenames[idx], preload=True)
            picks = mne.pick_types(med_raw.info, eeg=True)
            chns = [ch_name for ix, ch_name in enumerate(med_raw.info['ch_names'])
                    if ix not in picks]
            med_raw.drop_channels(chns)
            med_raw.add_proj([], remove_existing=True)

            raw = mne.concatenate_raws([eoec_raw, med_raw])

            fica = FourierICA(wsize=WSIZE, n_components=COMPONENTS, maxiter=7000, conveps=1e-11,
                              sfreq=raw.info['sfreq'], hpass=BAND[0], lpass=BAND[1])
            fica.fit(raw._data)

            components = get_components(fica, raw.info, len(raw.times), layout)

            handle = plot_components(components, layout)
            input_ = int(raw_input("Which component to use: "))
            if input_ == -1:
                continue
            new_components.append(components[input_ - 1])
        except:
            import pdb; pdb.set_trace()
            print "Exception!"

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
