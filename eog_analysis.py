import mne
import sys

fname = sys.argv[-1]

raw = mne.io.Raw(fname, preload=True)

picks = mne.pick_types(raw.info, meg=False, eog=True)

eog_data = raw._data[picks]

import pdb; pdb.set_trace()
print "kissa"
