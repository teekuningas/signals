# Gets brain data and plots topographies of its alpha and delta channels in different mind states
# Usage:
#     python analysis.py
import numpy as np
import mne
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

FREQUENCY_BANDS = {
    'alpha': (8, 12),
    'theta': (4, 8),
}


def _draw_translated_topography(fig, layout, info, x, y, 
                                layout_scale, band, freqs, psds,
                                min_power, max_power, name=''):
    """ Draws translated topography as axes to a figure
    """

    # translate locations to where the axes will be created
    layout = deepcopy(layout)
    for idx, (x0, y0, w, h) in enumerate(layout.pos):
        layout.pos[idx] = [
            x0 + x,
            y0 + y,
            w,
            h,
        ]

    # create axes and add them to figure
    for ax, idx in mne.viz.iter_topography(info, 
                                           layout=layout,
                                           fig=fig,
                                           axis_facecolor='0.8',
                                           layout_scale=layout_scale):
        # calculate and modify color
        good_indices = [i for i, freq in enumerate(freqs)
                        if freq <= FREQUENCY_BANDS[band][1]
                        and freq >= FREQUENCY_BANDS[band][0]]

        power = sum([psds[idx][freq_idx] for freq_idx in good_indices]) 

        band_length = len(good_indices)

        color = (power - min_power*band_length) / ((max_power - min_power)*band_length)
        ax.patch.set_facecolor(str(color)) 

        # add to figure
        fig.add_axes(ax)


    # find out locations for borders
    min_x, min_y, max_x, max_y = 10, 10, 0, 0
    for idx, (x0, y0, w, h) in enumerate(layout.pos):
        if min_x >= x0:
            min_x = x0
        if min_y >= y0:
            min_y = y0
        if max_x <= x0:
            max_x = x0
        if max_y <= y0:
            max_y = y0

    # add borders to figure
    rectangle_axes = plt.axes([
        min_x*layout_scale, 
        min_y*layout_scale, 
        (max_x-min_x)*layout_scale+w, 
        (max_y-min_y)*layout_scale + h
    ])
    rectangle_axes.patch.set_fill(False)
    plt.setp(list(rectangle_axes.spines.values()), color='red')
    rectangle_axes.set_xticklabels([])
    rectangle_axes.set_yticklabels([])
    plt.setp(rectangle_axes.get_xticklines(), visible=False)
    plt.setp(rectangle_axes.get_yticklines(), visible=False)

    rectangle_axes.text(0.1, -0.1, band + ', ' + name, color='green')

    fig.add_axes(rectangle_axes)


def _average_epochs_out(psds):

    averaged_psds = np.zeros(psds.shape[1:])
    for j in range(psds.shape[1]):
        for k in range(psds.shape[2]):
            sum_of_powers = 0
            for i in range(psds.shape[0]):
                sum_of_powers = sum_of_powers + psds[i][j][k]
            average = sum_of_powers / psds.shape[0]
            averaged_psds[j][k] = average
    return averaged_psds


def _find_power_limits(averaged_psds):
    min_power = 1
    max_power = 0
    for i in range(averaged_psds.shape[0]):
        for j in range(averaged_psds.shape[1]):
            if averaged_psds[i][j] <= min_power:
                min_power = averaged_psds[i][j]
            if averaged_psds[i][j] >= max_power:
                max_power = averaged_psds[i][j]
    return min_power, max_power


def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw



def main():

    layout_path = '/home/zairex/Code/cibr/materials/'
    layout_filename = 'gsn_129.lout'
    raw_file = '/home/zairex/Code/cibr/data/MI_eggie/MI_KH007_Meditaatio 201302.002'

    layout = mne.channels.read_layout(layout_filename, layout_path)

    # mne.viz.iter_topography expects these and the ones in raw.info to match
    layout.names = ['EEG ' + name.zfill(3) for name in layout.names]
    # also prepare for drawing by decreasing size of the boxes
    layout.pos = np.array([[x, y, w/2, h/2] for x, y, w, h in layout.pos])

    raw = read_raw(raw_file)

    print "Find good events for meditation and mind wandering from triggers"

    events = mne.find_events(raw, verbose=False)

    meditation_events = None
    wander_events = None 

    sampling_rate = 1000
    epoch_size = 3*sampling_rate
    after_trigger_interval = 5*sampling_rate
    before_trigger_interval = 1*sampling_rate
    meditation_wander_distance = 5*sampling_rate

    for idx in range(len(events)):

        if idx == 0:
            last_trigger = 0
        else:
            last_trigger = events[idx-1][0]

        trigger = events[idx][0]

        if trigger - epoch_size - before_trigger_interval > last_trigger + after_trigger_interval:  # noqa
            event = deepcopy(events[idx])
            event[0] = trigger - before_trigger_interval + epoch_size
            if wander_events is not None:
                wander_events = np.concatenate([wander_events, event.reshape(1, 3)])  # noqa
            else:
                wander_events = event.reshape(1, 3)

        if trigger - 2*epoch_size - before_trigger_interval - meditation_wander_distance > last_trigger + after_trigger_interval:  # noqa
            event = deepcopy(events[idx])
            event[0] = trigger - before_trigger_interval - 2*epoch_size - meditation_wander_distance  # noqa
            if meditation_events is not None:
                meditation_events = np.concatenate([meditation_events, event.reshape(1, 3)])  # noqa
            else:
                meditation_events = event.reshape(1, 3)


    # drop stimulus channels
    picks = mne.pick_types(raw.info, eeg=True)

    meditation_epochs = mne.Epochs(raw, meditation_events, event_id=None, picks=picks, 
                                   tmin=0, tmax=4, verbose=False)

    wander_epochs = mne.Epochs(raw, wander_events, event_id=None, picks=picks, 
                                   tmin=0, tmax=4, verbose=False)

    print "Find out power densities for wandering thoughts"
    psds, freqs = mne.time_frequency.psd.compute_epochs_psd(wander_epochs,
                                                            fmin=4,
                                                            fmax=20,
                                                            n_fft=4096,
                                                            verbose=False)
    wander_averaged_psds = _average_epochs_out(psds)
    wander_freqs = freqs

    print "Find out power densities for meditation"
    psds, freqs = mne.time_frequency.psd.compute_epochs_psd(meditation_epochs,
                                                            fmin=4,
                                                            fmax=20,
                                                            n_fft=4096,
                                                            verbose=False)
    meditation_averaged_psds = _average_epochs_out(psds)
    meditation_freqs = freqs

    # create figure
    fig = plt.figure()

    print "Find out min and max powers for plotting"
    med_min, med_max = _find_power_limits(meditation_averaged_psds)
    wander_min, wander_max = _find_power_limits(wander_averaged_psds)

    print "Start plotting"
    # plot wandering thoughts topographies
    _draw_translated_topography(fig, layout, raw.info, 0.2, 1.2, 0.3, 'alpha', 
                                wander_freqs, wander_averaged_psds, 
                                wander_min, wander_max, 'wandering thoughts')
    _draw_translated_topography(fig, layout, raw.info, 1.4, 1.2, 0.3, 'theta', 
                                wander_freqs, wander_averaged_psds, 
                                wander_min, wander_max, 'wandering thoughts')

    # plot meditation topographies
    _draw_translated_topography(fig, layout, raw.info, 0.2, 0.2, 0.3, 'alpha', 
                                meditation_freqs, meditation_averaged_psds, 
                                med_min, med_max, 'meditation')
    _draw_translated_topography(fig, layout, raw.info, 1.4, 0.2, 0.3, 'theta', 
                                meditation_freqs, meditation_averaged_psds, 
                                med_min, med_max, 'meditation')

    fig.show()

    quit = raw_input('Quit?')


if __name__ == '__main__':
    main()
