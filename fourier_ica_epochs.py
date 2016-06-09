import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA

from lib.stft import STFTPlot


def _filter_triggers(triggers, sfreq, start, end, rad=10, overlap=5):
    """
    check if this trigger is ok with following conditions:
      * it is at least ``rad`` seconds away from other triggers
      * it is at least ``rad`` seconds away from start or end
    """

    rad = rad * sfreq

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


def plot_epochs(raw, layout, band=[8, 14], n_components=5):
    """
    """
    raw = raw.copy()

    wsize = 1000 # step size 500
    rad = 15
    sfreq = raw.info['sfreq']

    # find and filter triggers
    triggers = mne.find_events(raw)[:, 0]
    triggers = _filter_triggers(triggers, sfreq, raw.first_samp, raw.last_samp,
                                rad=rad)

    # drop bad and non-data channels
    raw.drop_channels(raw.info['ch_names'][128:] + raw.info['bads'])

    data = raw._data

    # create epochs
    epochs = []
    for trigger in triggers:
        epochs.append(data[:, (trigger-rad*sfreq):(trigger+rad*sfreq)])
    
    # calculate fourier icas
    ficas = []

    for epoch in epochs:
        try:
            fica = FourierICA(wsize=wsize, n_components=n_components, 
                              sfreq=sfreq, hpass=band[0], lpass=band[1])
            fica.fit(epoch)
            ficas.append(fica)
        except:
            # did not converge
            pass

    # plot icas
    for fica in ficas:
        freqs = fica.freqs
        source_stft = fica.source_stft

        fig_ = plt.figure()
        for i in range(source_stft.shape[0]):
            sensor_component = np.abs(fica.component_in_sensor_space(i))
            tfr_ = mne.time_frequency.AverageTFR(raw.info, sensor_component, 
                range(sensor_component.shape[2]), freqs, 1)

            axes = fig_.add_subplot(source_stft.shape[0], 1, i + 1)
            mne.viz.plot_tfr_topomap(tfr_, layout=layout, axes=axes, show=False)

        plot_ = STFTPlot(freqs, source_stft, window_width=4*rad, 
                         window_height=5,
                         ch_names=[str(i+1) for i in range(len(source_stft))])

        plt.show()


if __name__ == '__main__':

    fname = '/home/zairex/Code/cibr/data/graduprosessoidut/kokeneet/KH001_MED-raw.fif'  # noqa
    raw = mne.io.Raw(fname, preload=True)

    layout_fname = 'gsn_129.lout'
    layout_path = '/home/zairex/Code/cibr/materials/'
    layout = mne.channels.read_layout(layout_fname, layout_path)

    plot_epochs(raw, layout, band=[4, 25], n_components=5)