""" Fix negative values in trigger channels 
Usage: ipython fix_triggers.py filename
"""

import mne
import sys

argv = sys.argv

raw = mne.io.Raw(argv[-1], preload=True)

print "Fixing stim channel values"

stim_channel = raw.info['ch_names'].index('STI101')
for i, x in enumerate(raw._data[stim_channel]):
    if x < 0:
        raw._data[stim_channel, i] = x + 2**16

raw.save('fixed_' + argv[-1])
