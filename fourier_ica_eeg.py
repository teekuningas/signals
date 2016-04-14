import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA

from lib.difference import DifferencePlot
from lib.stft import STFTPlot


def plot_whole_measurement(raw):
    """
    """
    wsize = 2000
    fica = FourierICA(wsize=wsize, n_pca_components=20, 
                      n_ica_components=15,
                      sfreq=raw.info['sfreq'], hpass=4, lpass=15)
    fica.fit(raw._data[:128, raw.first_samp:raw.last_samp])
    freqs, source_stft = fica.source_stft

    try:
        triggers = mne.find_events(raw)[:, 0] - raw.first_samp
        triggers = triggers.astype(np.float64) / (wsize/2)
        triggers = triggers[np.where(triggers > 1)[0]]
    except:
        triggers = []

    # ignore_times = range(480, 510)
    ignore_times = []

    STFTPlot(freqs, source_stft, window_width=50, triggers=triggers,
             ch_names=[str(i+1) for i in range(len(source_stft))],
             ignore_times=ignore_times)


def _filter_triggers(triggers, sfreq, start, end, rad=10, overlap=5):
    """
    check if this trigger is ok with following conditions:
      * it is at least ``2 * rad - overlap`` seconds away from 
        the last one
      * it is at least ``rad`` seconds away from start or end
    """
    start = start / float(sfreq)
    end = end / float(sfreq)

    valid_triggers = []
    for trigger in triggers:
        current = trigger / float(sfreq)

        if valid_triggers:
            last = valid_triggers[-1] / float(sfreq)
            if current < last + 2 * rad - overlap:
                continue

        if current < start + rad:
            continue

        if current > end - rad:
            continue

        valid_triggers.append(trigger)

    return np.array(valid_triggers, dtype=triggers.dtype)


def plot_epochs(raw):
    wsize = 1000
    rad = 15
    sfreq = raw.info['sfreq']

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp,
                                rad=rad, overlap=rad/2)

    # create epochs
    epochs = []
    for trigger in triggers:
        data = raw._data
        epochs.append(data[:128, (trigger-rad*sfreq):(trigger+rad*sfreq)])
    
    # remove outliers or noisy data

    # calculate fourier ica for each of them
    stfts = []
    for epoch in epochs:
        data = epoch
        fica = FourierICA(wsize=wsize, n_pca_components=20, 
                          n_ica_components=5, conveps=1e-13,
                          sfreq=sfreq, hpass=8, lpass=14)
        fica.fit(data)
        freqs, source_stft = fica.source_stft

        STFTPlot(freqs, source_stft, window_width=60, window_height=5,
                 ch_names=[str(i+1) for i in range(len(source_stft))])


if __name__ == '__main__':

    raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH005_MED-pre.fif',
                     preload=True)

    # plot_whole_measurement(raw)
    plot_epochs(raw)
