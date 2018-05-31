import mne
import sys
import numpy as np

paths = sys.argv[2:]
raws = []
for path in paths:
    print path
    raw = mne.io.Raw(path, preload=True, verbose='error')
    picks = mne.pick_types(raw.info, meg='grad')
    raw.drop_channels([ch for idx, ch in enumerate(raw.info['ch_names'])
                       if idx not in picks])
    raw._data = (raw._data - np.mean(raw._data)) / np.std(raw._data)
    raws.append(raw)

raw = mne.concatenate_raws(raws)

raw.save(sys.argv[1])

