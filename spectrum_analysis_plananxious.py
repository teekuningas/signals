import sys

import numpy as np
import mne
from mne.time_frequency import psd_welch
from mne.viz import iter_topography

import matplotlib.pyplot as plt

filenames = sys.argv[1:]

raw = mne.io.Raw(filenames, preload=True)
raw.filter(l_freq=1, h_freq=100, n_jobs=2)

print "Finding events"

events = mne.find_events(raw)

picks = mne.pick_types(raw.info, meg='grad')
raw.drop_channels([ch_name for idx, ch_name in enumerate(raw.info['ch_names']) 
                   if idx not in picks])

def get_spectrum(trigger):
    intervals = []
    for idx, (time, start, id_) in enumerate(events):
        if id_ == trigger:
            start = (time - raw.first_samp) / raw.info['sfreq']
            end = (events[idx+1][0] - raw.first_samp) / raw.info['sfreq']
            intervals.append((start, end))

    spectra = [psd_welch(raw, tmin=ival[0], tmax=ival[1], n_jobs=3,
                         fmin=1, fmax=80, n_fft=2048) 
               for ival in intervals]

    freqs = spectra[0][1]
    return freqs, np.average([spectrum[0] for spectrum in spectra], axis=0)


print "Calculating spectrums"

freqs, mind_spectrum = get_spectrum(10)
freqs, rest_spectrum = get_spectrum(11)
freqs, plan_spectrum = get_spectrum(12)
freqs, anxious_spectrum = get_spectrum(13)

print "Plotting"

def my_callback(ax, ch_idx):
    """
    """
    ax.plot(freqs, mind_spectrum[ch_idx], color='red')
    ax.plot(freqs, plan_spectrum[ch_idx], color='green')
    ax.plot(freqs, anxious_spectrum[ch_idx], color='blue')
    ax.plot(freqs, rest_spectrum[ch_idx], color='yellow')
    ax.set_xlabel = 'Frequency (Hz)'
    ax.set_ylabel = 'Power (dB)'
    plt.show()

for ax, idx in iter_topography(raw.info,
                               fig_facecolor='white',
                               axis_facecolor='white',
                               axis_spinecolor='white',
                               on_pick=my_callback):
    ax.plot(mind_spectrum[idx], color='red')
    ax.plot(plan_spectrum[idx], color='green')
    ax.plot(anxious_spectrum[idx], color='blue')
    ax.plot(rest_spectrum[idx], color='yellow')

plt.show()
