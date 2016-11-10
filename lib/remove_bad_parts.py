import matplotlib.pyplot as plt
import numpy as np
import mne
import math
from mne.time_frequency import psd_welch
from mne.viz import iter_topography

from load import load_layout


def sliding_crop(raw, interval):
    """ Slides data backwards and then crops to get a clean result """
    start = raw.time_as_index(interval[0])
    end = raw.time_as_index(interval[1])
    part = raw.copy()
    part._data = raw._data[:, start:end]
    part = part.crop(tmin=0, tmax=interval[1]-interval[0]-1)
    return part


def remove_bad_parts(raw):
    """ find intervals that have outlying spectral density and remove them
    """

    picks = mne.pick_types(raw.info, eeg=True)
    picks = picks[0:128]

    block_size = 5
    length = math.floor(raw._data.shape[1] / raw.info['sfreq'])

    current_block = 0
    values = []
    while True:
        if current_block+block_size > length:
            raw.crop(tmax=current_block)
            break
        psds, freqs = psd_welch(raw, picks=picks, tmin=current_block, 
                                tmax=current_block+block_size,
                                fmin=60, fmax=90, n_fft=1024)
        values.append((current_block, np.average(psds)))

        current_block += block_size

    sorted_values = sorted(values, key=lambda x: x[1])
    
    quartile_3 = sorted_values[int(len(sorted_values) * 0.75)][1]
    quartile_1 = sorted_values[int(len(sorted_values) * 0.25)][1]

    # find points marking bad intervals
    timepoints = []

    idx = len(sorted_values) - 1
    while True:
        if sorted_values[idx][1] > quartile_3 + 3 * (quartile_3 - quartile_1):
            # crop
            print ("Dropping interval " + str(sorted_values[idx][0]) + 
                   " - " + str(sorted_values[idx][0] + block_size))
            timepoints.append(sorted_values[idx][0])
        else:
            break

        idx -= 1

    timepoints.sort()
    timepoints.append(math.floor(raw.times[raw.last_samp - raw.first_samp]))

    # find good intervals
    min_length = 10
    last_point = 0
    retained_intervals = []
    for timepoint in timepoints:

        if timepoint - last_point >= min_length:
            retained_intervals.append((last_point, timepoint))

        last_point = timepoint + block_size

    # divide
    parts = []
    for interval in retained_intervals:
        parts.append(sliding_crop(raw, interval))

    # concat
    raw = parts[0]
    raw.append(parts[1:])

    return raw


if __name__ == '__main__':
    raw = mne.io.Raw('/home/zairex/Code/cibr/data/clean/KH010-raw.fif', preload=True)
    remove_bad_parts(raw)
    raw.plot()


