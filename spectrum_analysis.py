import matplotlib.pyplot as plt
import numpy as np
import mne

from mne.viz import iter_topography
from mne.time_frequency import psd_welch

from lib.load import get_raw
from lib.load import cli_raws
from lib.load import load_layout
from lib.remove_bad_parts import remove_bad_parts


SUBJECT = 'KH031'
N_FFT = 2048
SAVE = True
MEG = True

# read and preprocess rest
# raw = get_raw(SUBJECT, 'eoec')
# raw = mne.io.Raw('/home/zairex/Code/cibr/data/clean/' + SUBJECT + '_EOEC-raw.fif', preload=True)

raws = cli_raws()

# if meg, drop magnetometers 
if MEG:
    for raw in raws:
        raw.drop_channels([ch_name for ch_name in raw.info['ch_names']
                           if 'MEG' not in ch_name or ch_name.endswith('1')])

raw = remove_bad_parts(raws[0])

picks = mne.pick_types(raw.info, eeg=True)
# tmin, tmax = 1, 90
tmin, tmax = 1, 270
fmin, fmax = 1, 40
# fmin, fmax = 1, 20

rest_psds, rest_freqs = psd_welch(raw, picks=picks, tmin=tmin, tmax=tmax,
                                  fmin=fmin, fmax=fmax, n_fft=N_FFT)
rest_psds = 20 * np.log10(rest_psds)


# read and preprocess mindfulness
# raw = get_raw(SUBJECT, 'med')
# raw = mne.io.Raw('/home/zairex/Code/cibr/data/clean/' + SUBJECT + '-raw.fif', preload=True)

raw = remove_bad_parts(raws[1])

# crop wandering thoughts 
# ...

picks = mne.pick_types(raw.info, eeg=True)
fmin, fmax = 1, 40
# fmin, fmax = 1, 20

mind_psds, mind_freqs = psd_welch(raw, picks=picks,
                                  fmin=fmin, fmax=fmax, n_fft=N_FFT)
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
if MEG:
    layout = None

import pdb; pdb.set_trace()

for ax, idx in iter_topography(raw.info, layout=layout,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback):
    ax.plot(rest_psds[idx], color='red')
    ax.plot(mind_psds[idx], color='blue')

plt.gcf().suptitle('Power spectral densities of rest and mindfulness')
plt.show()

if SAVE and not MEG:
    # Oz: 75 | 82 74 70 71 76 83
    # Pz: 62 | 72 67 61 78 77
    # Cz:    | 55 31 7 106 80
    # Fz: 11 | 12 19 18 16 10 4 5
    # C4: 104 | 93 87 105 111 110 103
    # C3: 36 | 41 35 29 30 37 42
    selections = {
        'Oz': [75, 82, 74, 70, 71, 76, 83],
        'Pz': [62, 72, 67, 61, 78, 77],
        'Cz': [55, 31, 7, 106, 80],
        'Fz': [11, 12, 19, 18, 16, 10, 4, 5],
        'C4': [104, 93, 87, 105, 111, 110, 103],
        'C3': [36, 41, 35, 29, 30, 37, 42],
    }
      
    mind_alpha = mind_psds[:, (mind_freqs > 0) & (mind_freqs < 12)]
    rest_alpha = rest_psds[:, (rest_freqs > 0) & (rest_freqs < 12)]
    mind_means = np.mean(mind_alpha, axis=1)
    rest_means = np.mean(rest_alpha, axis=1)

    data = []
    header = []

    for key, value in selections.items():
        data.append(np.mean(mind_means[value]))
        header.append('mind_' + str(key).lower())
        
    for key, value in selections.items():
        data.append(np.mean(rest_means[value]))
        header.append('rest_' + str(key).lower())

    path = 'data/alpha2/' + SUBJECT + '.csv'
    print "Saving: " + path
    np.savetxt(path, np.array([header, data]), fmt="%s", delimiter=',') 
    
