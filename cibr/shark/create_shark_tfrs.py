# Usage:
# ipython create_shark_tfrs /path/to/fif

import os
import sys

import mne
import numpy as np

from mne.time_frequency.tfr import tfr_morlet
from mne.time_frequency.tfr import write_tfrs

SAVE_PATH = 'output/shark_tfrs/'

raw = mne.io.Raw(sys.argv[-1], preload=True)

left_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1, mask=811)  # noqa
right_events = mne.find_events(raw, stim_channel='STI101', shortest_event=1, mask=791)  # noqa

picks = mne.pick_types(raw.info, meg='grad')
raw.drop_channels([ch_name for idx, ch_name in enumerate(raw.info['ch_names']) 
                   if idx not in picks])

freqs = np.arange(6., 30., 2.)
n_cycles = freqs / 2.0
tmin, tmax = -1.0, 2.0

import pdb; pdb.set_trace()

left_epochs = mne.Epochs(raw, left_events, [4,  16], tmin, tmax,
                         preload=True)

right_epochs = mne.Epochs(raw, right_events, [8,  32], tmin, tmax,
                          preload=True)

print "Doing left tfr"
left_tfr = tfr_morlet(left_epochs, freqs=freqs, n_cycles=n_cycles, 
                      return_itc=False, n_jobs = 2, decim = 8)
left_tfr.comment = 'left_cue'

import pdb; pdb.set_trace()

print "Doing right tfr"
right_tfr = tfr_morlet(right_epochs, freqs=freqs, n_cycles=n_cycles, 
                       return_itc=False, n_jobs = 2, decim = 8)
right_tfr.comment = 'right_cue'

fname = sys.argv[-1].split('/')[-1].split('.')[0]

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

write_tfrs(SAVE_PATH + fname + '-tfr.h5', [left_tfr, right_tfr], 
           overwrite=True)
