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
    n_ica_components = 10

    try:
        triggers = mne.find_events(raw)[:, 0] - raw.first_samp
        triggers = triggers.astype(np.float64) / (wsize/2)
        triggers = triggers[np.where(triggers > 1)[0]]
    except:
        triggers = []

    # drop non-data channels
    raw.drop_channels(raw.info['ch_names'][128:])

    fica = FourierICA(wsize=wsize, n_pca_components=20, 
                      n_ica_components=n_ica_components,
                      sfreq=raw.info['sfreq'], hpass=4, lpass=15)
    fica.fit(raw._data[:, raw.first_samp:raw.last_samp])

    source_stft = fica.source_stft
    freqs = fica.freqs

    # ignore_times = range(480, 510)
    ignore_times = []

    plot_ = STFTPlot(freqs, source_stft, window_width=50, triggers=triggers,
                     ch_names=[str(i+1) for i in range(len(source_stft))],
                     ignore_times=ignore_times)

    plt.show()


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


def plot_epochs(raw, layout):
    wsize = 1000 # step size 500
    rad = 15
    sfreq = raw.info['sfreq']
    n_ica_components = 5

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp,
                                rad=rad, overlap=rad/2)

    # drop non-data channels
    raw.drop_channels(raw.info['ch_names'][128:])

    # create epochs
    epochs = []
    for trigger in triggers:
        data = raw._data
        epochs.append(data[:, (trigger-rad*sfreq):(trigger+rad*sfreq)])
    
    # remove outliers or noisy data

    # calculate fourier ica for each of them
    stfts = []
    for epoch in epochs:
        data = epoch
        fica = FourierICA(wsize=wsize, n_pca_components=20, 
                          n_ica_components=n_ica_components, conveps=1e-13,
                          sfreq=sfreq, hpass=8, lpass=14)
        fica.fit(data)
        freqs = fica.freqs
        source_stft = fica.source_stft

        fig_ = plt.figure()
        for i in range(source_stft.shape[0]):
            sensor_component = np.abs(fica.component_in_sensor_space(i))
            tfr_ = mne.time_frequency.AverageTFR(raw.info, sensor_component, range(sensor_component.shape[2]), freqs, 1)
            axes = fig_.add_subplot(source_stft.shape[0], 1, i + 1)
            mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

        plot_ = STFTPlot(freqs, source_stft, window_width=4*rad, 
                         window_height=n_ica_components,
                         ch_names=[str(i+1) for i in range(len(source_stft))])

        plt.show()


if __name__ == '__main__':

    raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH005_MED-pre.fif',
                     preload=True)

    layout_fname = 'gsn_129.lout'
    layout_path = '/home/zairex/Code/cibr/materials/'
    layout = mne.channels.read_layout(layout_fname, layout_path)


    # plot_whole_measurement(raw)
    plot_epochs(raw, layout)
