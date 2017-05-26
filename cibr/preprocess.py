import matplotlib.pyplot as plt
import mne
import numpy as np
import sys

raw = mne.io.Raw(sys.argv[1], preload=True)

import pdb; pdb.set_trace()

input_ = raw_input('Resample to (empty means do not): ')
if input_ != '':
    raw.resample(int(input_))

input_ = raw_input('Low-pass filter to (empty means do not): ')
if input_ == '':
    h_freq = None
else:
    h_freq = int(input_)

input_ = raw_input('High-pass filter to (empty means do not): ')
if input_ == '':
    l_freq = None
else:
    l_freq = int(input_)

raw.filter(l_freq=l_freq, h_freq=h_freq)

input_ = raw_input('Drop magnetometers (y for yes)?')
if input_ == 'y':
    try:
        magnetometers = mne.pick_types(raw.info, meg='mag')
        raw.drop_channels([ch_name for idx, ch_name in enumerate(raw.info['ch_names']) if idx in magnetometers])
    except Exception as e:
        print "Could not drop magnetometers: " + str(e)

print "Start doing ICA"

ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')

ica.fit(raw)

sources = ica.get_sources(raw)

# alter amplitudes to get better plot 
for source in sources._data:
    for idx, amplitude in enumerate(source):
        source[idx] = amplitude / 5000.0

ica.plot_components(show=False)

sources.plot()

indices = raw_input('Please enter indices (starts from one) of '
                    'ICA components to be zeroed out '
                    '(separated with spaces, empty for none): ')
if indices:
    indices = map(int, indices.split(' '))
    indices = [i-1 for i in indices]

    # project out selected ica components
    ica.apply(raw, exclude=indices)

raw.plot()

to_be_saved = raw_input('Do you want to save (y, n)? ')

if to_be_saved == 'y':
    raw.save(sys.argv[2], overwrite=True)
