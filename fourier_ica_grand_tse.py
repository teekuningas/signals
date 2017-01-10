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
LIMITS = [10, 5]


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
    
    # take range of 5 hertz for tse
    tse_range = np.where((freqs > max_-1) & (freqs < max_+1))[0]

    tse = np.mean(np.abs(component.source_stft[tse_range, :]), axis=0)

    # smoothen?
    tse = np.convolve(tse, np.ones((3,))/3, mode='valid')

    return tse


def _get_epochs(component, tse):

    # load raw temporarily to find triggers
    raw = mne.io.read_raw_fif(component.info['filename'], preload=False)

    sfreq = component.info['sfreq']

    # find and filter triggers
    events = [event for event in component.events if event[1] == 0]
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

    subject_epochs = []
    for i in range(len(components)):
        epochs = _get_epochs(components[i], tses[i])
        # normalize
        # for j, epoch in enumerate(epochs):
        #     epochs[j] = epoch / np.mean(tses[i])

        if len(epochs) == 0:
            continue

        max_value = np.max(np.mean(epochs, axis=0))
        limit = 1.8
        if max_value > limit:
            continue

        subject_epochs.append(np.array(epochs))

    print str(sum([len(epochs) for epochs in subject_epochs])) + " epochs found."

    # average = np.mean(epochs, axis=0)

    length = len(subject_epochs[0][0])
    step = (LIMITS[0]+LIMITS[1]) / float(length)
    times = np.array(range(length))
    times = times * step - LIMITS[0] + step/2

    # average
    average = np.mean([np.mean(epoch, axis=0) for epoch in subject_epochs], 
                      axis=0)
    fig, ax = plt.subplots()
    ax.set(title="TSE averaged over subjects and events", xlabel="Time (ms)")
    plt.plot(times, average)

    fig, ax = plt.subplots()
    ax.set(title='TSE', xlabel='Time (s)')
    for epochs in subject_epochs:
        ax.plot(times, np.mean(epochs, axis=0))

    # for i, epochs in enumerate(subject_epochs):
    #     average = np.mean(epochs, axis=0)
    #     fig, ax = plt.subplots()
    #     ax.plot(times, average)

    plt.show()
    import pdb; pdb.set_trace()
    print "Miau"


if __name__ == '__main__':
    main()
