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

events = mne.find_events(raw, shortest_event=1)

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
                         fmin=1, fmax=40, n_fft=2048) 
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

selections = {
    'Oz': ['211', '192', '234', '204', '203'],
    'Pz': ['183', '224', '201', '202'],
    'Cz': ['071', '072', '074', '073'],
    'Fz': ['061', '101', '102', '064', '062', '103'],
    'C4': ['113', '134', '222', '241'],
    'C3': ['023', '044', '162', '181'],
}

for key, value in selections.items():
    selections[key] = [idx for idx, ch_name in 
                       enumerate(raw.info['ch_names'])
                       if ch_name[3:6] in value]

mind_alpha = mind_spectrum[:, (freqs > 8) & (freqs < 12)]
rest_alpha = rest_spectrum[:, (freqs > 8) & (freqs < 12)]
plan_alpha = plan_spectrum[:, (freqs > 8) & (freqs < 12)]
anxious_alpha = anxious_spectrum[:, (freqs > 8) & (freqs < 12)]
mind_means = np.mean(mind_alpha, axis=1)
rest_means = np.mean(rest_alpha, axis=1)
plan_means = np.mean(plan_alpha, axis=1)
anxious_means = np.mean(anxious_alpha, axis=1)

data = []
header = []

for key, value in selections.items():
    data.append(np.mean(mind_means[value]))
    header.append('mind_' + str(key).lower())

for key, value in selections.items():
    data.append(np.mean(rest_means[value]))
    header.append('rest_' + str(key).lower())

for key, value in selections.items():
    data.append(np.mean(plan_means[value]))
    header.append('plan_' + str(key).lower())

for key, value in selections.items():
    data.append(np.mean(anxious_means[value]))
    header.append('anxious_' + str(key).lower())


fname = filenames[0].split('/')[-1].split(".")[0]

path = 'data/plananxious/stats/' + fname + '.csv'
print "Saving: " + path
np.savetxt(path, np.array([header, data]), fmt="%s", delimiter=',')
