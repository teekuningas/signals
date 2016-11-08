import pickle
import sys

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.cluster import cluster_components as cluster_matrix
from lib.load import load_layout
from lib.load import get_raw
from lib.component import ComponentPlot


BAND = [5, 18]
COMPONENTS = 10
RADIUS = 15
WSIZE = 8192

RESULT_FOLDER = '/home/zairex/Code/cibr/analysis/data'

SUBJECTS = [
    ('KH001', 'med', 'experienced', 'preprocessed'),
    ('KH002', 'med', 'experienced', 'preprocessed'),
    ('KH003', 'med', 'experienced', 'preprocessed'),
    ('KH005', 'med', 'experienced', 'preprocessed'),
    ('KH007', 'med', 'experienced', 'preprocessed'),
    ('KH009', 'med', 'experienced', 'preprocessed'),
    ('KH016', 'med', 'experienced', 'preprocessed'),
    ('KH017', 'med', 'experienced', 'preprocessed'),
    ('KH024', 'med', 'experienced', 'preprocessed'),
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
    def __init__(self, source_stft, sensor_stft, sensor_topo, source_psd, freqs, length, info):
        self.source_stft = source_stft
        self.sensor_stft = sensor_stft
        self.source_psd = source_psd
        self.sensor_topo = sensor_topo
        self.freqs = freqs
        self.info = info
        self.length = length


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


def _get_epochs(component, raw):
    sfreq = raw.info['sfreq']
    rad = RADIUS*sfreq

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp)

    # create epochs
    source_stft = component.source_stft
    length_samples = source_stft.shape[1]
    length_s = (raw.last_samp - raw.first_samp)
    scale = length_s / float(length_samples)

    epochs = []
    for trigger in triggers:
        # select nearest neighbor
        scaled_trigger = round(trigger / scale)
        interval = 4 # ~15s
        start = scaled_trigger - interval
        end = scaled_trigger + interval
        
        epochs.append(source_stft[:, start:end])

    print str(len(epochs)) + " epochs found!"

    return epochs


def split_to_epochs(components, subjects):
    epochs = []
    for idx, component in enumerate(components):
        epochs.extend(_get_epochs(component, 
            get_raw(subjects[idx][0], subjects[idx][1])))
    return epochs


def plot_epochs(epochs, freqs, info):
    info = info.copy()
    info['chs'] = info['chs'][0:epochs.shape[0]]
    info['ch_names'] = info['ch_names'][0:epochs.shape[0]]
    info['nchan'] = epochs.shape[0]

    return ComponentPlot(np.array(epochs), freqs, [], 0, epochs.shape[0], info, 30000,
                         window=9)


def main():

    mne.utils.set_log_level('ERROR')

    try:
        result_arg = filter(lambda x: 'RESULT=' in x, sys.argv)[0].split('=')[1]
    except:
        result_arg = RESULT_FOLDER

    layout = load_layout()

    components = pickle.load(open(".components.p", "rb"))

    epochs = split_to_epochs(components, SUBJECTS)

    print len(epochs)

    # so normalize and average
    for epoch in epochs:
        epoch = np.abs(epoch) / np.max(np.abs(epoch))

    epochs = np.average(epochs, axis=0)[np.newaxis, :]
    # puh = plot_epochs(np.array(epochs[0:5]), components[0].freqs, components[0].info)
    # pah = plot_epochs(np.array(epochs[5:10]), components[0].freqs, components[0].info)
    # poh = plot_epochs(np.array(epochs[10:]), components[0].freqs, components[0].info)

    cp = plot_epochs(epochs, components[0].freqs, components[0].info)

    import pdb; pdb.set_trace()
    print "kissa"


if __name__ == '__main__':
    main()
