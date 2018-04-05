# -*- coding: utf-8 -*-
import sys

import pyface.qt

import mne
import numpy as np
import matplotlib.pyplot as plt

from time import sleep


raw_file = sys.argv[1]
fwd_file = sys.argv[2]

raw = mne.io.Raw(raw_file, preload=True)
fwd = mne.forward.read_forward_solution(fwd_file)

events = mne.find_events(raw)
rest_start = ([event for event in events if event[2] == 11][0][0] - raw.first_samp) / raw.info['sfreq']
rest_end = rest_start + 120
cov = mne.compute_raw_covariance(raw, tmin=rest_start, tmax=rest_end)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)

med_start = ([event for event in events if event[2] == 10][0][0] - raw.first_samp) / raw.info['sfreq']
med_end = med_start + 53

med_raw_alpha = raw.copy().crop(tmin=med_start, tmax=med_end).filter(l_freq=8, h_freq=12).resample(5)

stc = mne.minimum_norm.apply_inverse_raw(med_raw_alpha, inv, lambda2=0.1)

def stc_data_development():
    plt.show()
    ax = plt.gca()

    for i in range(stc.shape[1]):
        ax.cla()
        ax.plot(stc.data[:, i])
        ax.set_ylim(0, 0.002)
        plt.draw()
        plt.pause(1e-17)
        sleep(0.05)

stc_data_development()
        
# plot brains

def plot_brains():
    fmin = np.percentile(stc.data, 15)
    fmid = np.percentile(stc.data, 50)
    fmax = np.percentile(stc.data, 85)

    brain = stc.plot(initial_time=30.0, hemi='both', clim={'kind': 'value', 'lims': [fmin, fmid, fmax]})
    brain.show_view(view='parietal')

    # brain.save_image_sequence(range(10, 100), "images/brain-frame_%0.4d.jpeg", montage=['par', 'fro'])

# plot_brains()

import pdb; pdb.set_trace()
print "miau"
