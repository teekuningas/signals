import pickle
import sys

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.cluster import cluster_components as cluster_matrix
from lib.load import load_layout
from lib.load import get_raw


BAND = [6, 14]
COMPONENTS = 8
RADIUS = 15
WSIZE = 8192

RESULT_FOLDER = '/home/zairex/Code/cibr/analysis/data'

SUBJECTS = [
    ('KH001', 'med', 'experienced', 'preprocessed'),
    ('KH002', 'med', 'experienced', 'preprocessed'),
    ('KH003', 'med', 'experienced', 'preprocessed'),
    ('KH004', 'med', 'experienced', 'preprocessed'),
    ('KH005', 'med', 'experienced', 'preprocessed'),
#   ('KH007', 'med', 'experienced', 'preprocessed'),
#   ('KH009', 'med', 'experienced', 'preprocessed'),
#   ('KH016', 'med', 'experienced', 'preprocessed'),
#   ('KH017', 'med', 'experienced', 'preprocessed'),
#   ('KH024', 'med', 'experienced', 'preprocessed'),
#   ('KH011', 'med', 'novice', 'preprocessed'),
#   ('KH013', 'med', 'novice', 'preprocessed'),
#   ('KH014', 'med', 'novice', 'preprocessed'),
#   ('KH015', 'med', 'novice', 'preprocessed'),
#   ('KH019', 'med', 'novice', 'preprocessed'),
#   ('KH021', 'med', 'novice', 'preprocessed'),
#   ('KH023', 'med', 'novice', 'preprocessed'),
#   ('KH025', 'med', 'novice', 'preprocessed'),
#   ('KH026', 'med', 'novice', 'preprocessed'),
#   ('KH028', 'med', 'novice', 'preprocessed'),
#   ('KH029', 'med', 'novice', 'preprocessed'),
#   ('KH030', 'med', 'novice', 'preprocessed'),
#   ('KH031', 'med', 'novice', 'preprocessed'),
]


class ComponentData(object):
    """
    """
    def __init__(self, source_stft, sensor_stft, sensor_topo, source_psd, freqs, info):
        self.source_stft = source_stft
        self.sensor_stft = sensor_stft
        self.source_psd = source_psd
        self.sensor_topo = sensor_topo
        self.freqs = freqs
        self.info = info


class SubjectData(object):
    """
    """
    def __init__(self, components, path, type_):
        self.components = components
        self.path = path
        self.type = type_


def _filter_triggers(triggers, sfreq, start, end):
    """
    check if this trigger is ok with following conditions:
      * it is at least ``rad`` seconds away from other triggers
      * it is at least ``rad`` seconds away from start or end
    """

    rad = RADIUS * sfreq

    valid_triggers = []
    for trigger in triggers:

        if trigger < start + rad:
            continue

        if trigger > end - rad:
            continue

        valid = True

        for other in triggers:
            # don't compare to itself
            if other == trigger:
                continue

            if trigger < other + rad and trigger > other - rad:
                valid = False

        if not valid:
            continue

        valid_triggers.append(trigger)

    return np.array(valid_triggers, dtype=triggers.dtype)


def get_epochs(raw):
    sfreq = raw.info['sfreq']
    rad = RADIUS*sfreq

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp)

    # drop bad and non-data channels
    raw.drop_channels(raw.info['ch_names'][128:] + raw.info['bads'])

    data = raw._data

    # create epochs
    epochs = []
    for trigger in triggers:
        epochs.append(data[:, int(trigger-rad):int(trigger+rad)])

    print str(len(epochs)) + " epochs found!"

    return epochs


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


def get_components(fica, info, layout):
    source_stft = fica.source_stft

    components = []
    for i in range(COMPONENTS):
        source_component = source_stft[i] 
        sensor_stft = fica.component_in_sensor_space(i)
        sensor_topo = _create_topo(sensor_stft, layout, info)
        source_psd = np.mean(np.power(np.abs(source_stft[i]), 2), axis=-1)
        component_data = ComponentData(
            source_stft=source_component,
            sensor_stft=sensor_stft,
            sensor_topo=sensor_topo,
            source_psd=source_psd,
            freqs=fica.freqs,
            info=info.copy(),
        )
        components.append(component_data)
    return components


def plot_components(components, layout, average=False):

    # create figure for head topographies
    fig_ = plt.figure()
    for i, component in enumerate(components):
        sensor_component = np.abs(component.sensor_stft)
        tfr_ = mne.time_frequency.AverageTFR(component.info, sensor_component, 
            range(sensor_component.shape[2]), component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

    # create figure ica components
    fig_ = plt.figure()
    for i, component in enumerate(components):

        # mock info
        info = component.info.copy()
        info['chs'] = [info['chs'][0]]

        rad = RADIUS
        data = np.power(np.abs((component.source_stft)[np.newaxis, :]), 2)

        times = np.arange(-rad, rad, float(rad*2)/data.shape[2])
        times = times / info['sfreq']
        tfr_ = mne.time_frequency.AverageTFR(info, data, 
            times, component.freqs, 1)

        axes = fig_.add_subplot(len(components), 1, i + 1)
        tfr_.plot(picks=[0], axes=axes, show=False)

    plt.show()


def cluster_components(subjects):

    # first create a flattened matrix out of subjects' component hierarchy
    component_matrix = []
    for subject in subjects:
        component_matrix.append(subject.components)

    # do the actual clustering
    ordered_matrix = cluster_matrix(component_matrix)

    # recreate the structured component hierarchy
    clustered = []
    for idx, subject in enumerate(subjects):
        clustered.append(SubjectData(
            components=ordered_matrix[idx],
            path=subject.path,
            type_=subject.type
        ))

    return clustered


def main():

    mne.utils.set_log_level('ERROR')

    try:
        result_arg = filter(lambda x: 'RESULT=' in x, sys.argv)[0].split('=')[1]
    except:
        result_arg = RESULT_FOLDER

    layout = load_layout()

    print "Reading and processing data from files.."
    subjects = []

    for fname, data_type, subject_type, preprocessed in SUBJECTS:
        raw = get_raw(fname, data_type)

        # find triggers

        picks = mne.pick_types(raw.info, eeg=True)
        raw.drop_channels([ch_name for idx, ch_name in 
            enumerate(raw.info['ch_names']) if idx not in picks])

        fica = FourierICA(wsize=WSIZE, n_components=COMPONENTS, 
                          sfreq=raw.info['sfreq'], hpass=BAND[0], lpass=BAND[1])
        fica.fit(raw._data[:, raw.first_samp:raw.last_samp])

        components = get_components(fica, raw.info, layout)

        subject = SubjectData(components=components, path=fname, 
                              type_=subject_type)
        subjects.append(subject)

    subjects = cluster_components(subjects)

    while True:
        input_ = raw_input("Which component to plot: ")
        if input_ == 'q':
            break
        components_to_plot = [subject.components[int(input_)] for subject in subjects]
        plot_components(components_to_plot, layout)

    # epoch stuff here

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
