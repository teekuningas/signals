import pickle
import sys
import os

import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA
from lib.cluster import cluster_components as cluster_matrix
from lib.load import load_layout
from lib.abstract import plot_components


MEG = True
LIMITS = [20, 10]


def _filter_triggers(triggers, sfreq, start, end):
    """
    remove triggers that are too close to start, end or others
    """

    left_limit = LIMITS[0] * sfreq
    right_limit = LIMITS[1] * sfreq

    valid_triggers = []
    for trigger in triggers:

        if trigger < start + left_limit:
            continue

        if trigger > end - right_limit:
            continue

        valid = True

        for other in triggers:
            limit = left_limit + right_limit

            # don't compare to itself
            if other == trigger:
                continue
           
            if other > trigger - limit and other < trigger + limit:
                valid = False

        if not valid:
            continue

        valid_triggers.append(trigger)

    return np.array(valid_triggers, dtype=triggers.dtype)


def _get_tse(component):

    # first find maxarg in alpha range
    freqs = component.freqs
    range_ = np.where((freqs > 6) & (freqs < 14))[0]
    start = range_[0]
    max_idx = start + np.argmax(component.source_psd[range_])
    max_ = freqs[max_idx]
    
    # take range of 2 hertz for tse
    tse_range = np.where((freqs > max_-1) & (freqs < max_+1))[0]

    tse = np.mean(np.abs(component.source_stft[tse_range, :]), axis=0)

    return tse


def _get_epochs(component, tse):

    # load raw temporarily to find triggers
    raw = mne.io.read_raw_fif(component.info['filename'], preload=False)

    sfreq = raw.info['sfreq']

    # find and filter triggers
    events = [event for event in mne.find_events(raw) if event[1] == 0]
    triggers = np.array(events)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp)
    triggers = [trigger - raw.first_samp for trigger in triggers]

    # create epochs
    source_stft = component.source_stft
    length_samples = source_stft.shape[1]
    length_raw = len(raw.times)
    scale = length_raw / float(length_samples)

    epochs = []
    for trigger in triggers:
        # select nearest neighbor
        scaled_trigger = int(trigger / scale + 0.5)
        left_interval = int((LIMITS[0]) * sfreq / scale)
        right_interval = int((LIMITS[1]) * sfreq / scale)
        start = scaled_trigger - left_interval
        end = scaled_trigger + right_interval
        
        epochs.append(tse[start:end])

    print str(len(epochs)) + " epochs found!"

    return epochs


def main():

    mne.utils.set_log_level('ERROR')

    layout = load_layout(MEG)

    filenames = sys.argv[1:]

    components = []
    for fname in filenames:
        print "Opening " + fname
        part = pickle.load(open(fname, "rb"))
        components.extend(part)

    tses = []
    for component in components:
        tses.append(_get_tse(component))

    epochs = []
    for i in range(len(components)):
        component_epochs = _get_epochs(components[i], tses[i])
        # normalize
        for j, epoch in enumerate(component_epochs):
            component_epochs[j] = epoch / np.mean(tses[i])

        epochs.extend(component_epochs)

    print "Total: " + str(len(epochs)) + " epochs."

    average = np.mean(epochs, axis=0)

    # smoothen?
    # average = np.convolve(average, np.ones((3,))/3, mode='valid')

    step = (LIMITS[0]+LIMITS[1]) / float(len(average))
    times = np.array(range(len(average)))
    times = times * step - LIMITS[0] + step/2

    plt.plot(times, average)
    plt.show()


if __name__ == '__main__':
    main()
