import sys

import matplotlib.pyplot as plt
import numpy as np
import mne

from mne.viz import iter_topography
from mne.time_frequency import psd_welch

from lib.load import get_raw
from lib.load import cli_raws
from lib.load import load_layout
from lib.remove_bad_parts import remove_bad_parts


N_FFT = 2048
SAVE = True
FMIN, FMAX = 3, 17

SUBJECT = sys.argv[1]

raw = mne.io.Raw('/home/zairex/Code/cibr/data/clean/' + SUBJECT + '_EOEC-raw.fif', preload=True)
raw = remove_bad_parts(raw)

picks = mne.pick_types(raw.info, eeg=True)
tmin, tmax = 1, 75

rest_psds, rest_freqs = psd_welch(raw, picks=picks, tmin=tmin, tmax=tmax,
                                  fmin=FMIN, fmax=FMAX, n_fft=N_FFT)
rest_psds = 20 * np.log10(rest_psds)

raw = mne.io.Raw('/home/zairex/Code/cibr/data/clean/' + SUBJECT + '-raw.fif', preload=True)
raw = remove_bad_parts(raw)

picks = mne.pick_types(raw.info, eeg=True)

mind_psds, mind_freqs = psd_welch(raw, picks=picks,
                                  fmin=FMIN, fmax=FMAX, n_fft=N_FFT)
mind_psds = 20 * np.log10(mind_psds)


def my_callback(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    ax.plot(rest_freqs, rest_psds[ch_idx], color='red')
    ax.plot(mind_freqs, mind_psds[ch_idx], color='blue')
    ax.set_xlabel = 'Frequency (Hz)'
    ax.set_ylabel = 'Power (dB)'
    plt.show()

layout = load_layout()

for ax, idx in iter_topography(raw.info, layout=layout,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback):
    ax.plot(rest_psds[idx], color='red')
    ax.plot(mind_psds[idx], color='blue')

plt.gcf().suptitle('Power spectral densities of rest and mindfulness')
# plt.show()

if SAVE:
    # Oz: 75 | 82 74 70 71 76 83
    # Pz: 62 | 72 67 61 78 77
    # Cz:    | 55 31 7 106 80
    # Fz: 11 | 12 19 18 16 10 4 5
    # C4: 104 | 93 87 105 111 110 103
    # C3: 36 | 41 35 29 30 37 42
    selections = {
        'Oz': [val - 1 for val in [75, 82, 74, 70, 71, 76, 83]],
   #     'Pz': [62, 72, 67, 61, 78, 77],
   #     'Cz': [55, 31, 7, 106, 80],
   #     'Fz': [11, 12, 19, 18, 16, 10, 4, 5],
   #     'C4': [104, 93, 87, 105, 111, 110, 103],
   #     'C3': [36, 41, 35, 29, 30, 37, 42],
    }
      
    mind_alpha = mind_psds[:, (mind_freqs > 7) & (mind_freqs < 13)]
    rest_alpha = rest_psds[:, (rest_freqs > 7) & (rest_freqs < 13)]
    mind_means = np.mean(mind_alpha, axis=1)
    rest_means = np.mean(rest_alpha, axis=1)

    data = []
    header = []

    for key, value in selections.items():
        data.append(np.mean(mind_means[value]))
        header.append('mind_avg_' + str(key).lower())

        data.append(mind_means[value][np.argmax(mind_means[value])])
        header.append('mind_peak_' + str(key).lower())
        
    for key, value in selections.items():
        data.append(np.mean(rest_means[value]))
        header.append('rest_avg_' + str(key).lower())

        data.append(rest_means[value][np.argmax(rest_means[value])])
        header.append('rest_peak_' + str(key).lower())

    path = 'data/sensor_alpha/' + SUBJECT + '.csv'
    print "Saving: " + path
    np.savetxt(path, np.array([header, data]), fmt="%s", delimiter=',') 
    
