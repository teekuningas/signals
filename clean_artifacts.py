# Reads raw EGI and saves as FIF
# Usage:
#     python clean_artifacts.py
import matplotlib.pyplot as plt
import mne
import numpy as np

# set filenames
clean_file = '/home/zairex/Code/cibr/data/cleaned/KH007-raw.fif'
temp_ica = '/home/zairex/Code/cibr/analysis/scripts/data/temp-ica.fif'
raw_file = '/home/zairex/Code/cibr/data/MI_eggie/MI_KH007_Meditaatio 201302.002'

if raw_file.endswith('.fif'):
    raw = mne.io.Raw(raw_file, preload=True)
else:
    # assume egi
    raw = mne.io.read_raw_egi(raw_file)

raw.filter(l_freq=1, h_freq=40)

raw.plot()

picks = mne.pick_types(raw.info, eeg=True, meg=True, stim=False, exclude='bads')

use_saved = raw_input('Use saved ica solution if found (y, n): ')

ica = None
if use_saved == 'y': 
    try:
        ica = mne.preprocessing.ica.read_ica(temp_ica)
    except IOError:
        pass
if not ica:
    ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    ica.fit(raw, picks=picks)
    ica.save(temp_ica)

sources = ica.get_sources(raw)

# alter amplitudes to get better plot 
for source in sources._data:
    for idx, amplitude in enumerate(source):
        source[idx] = amplitude / 5000.0

sources.plot()

indices = raw_input('Please enter indices (starts from zero) of '
                    'ICA components to be zeroed out '
                    '(separated with spaces, empty for none): ')
if indices:
    indices = map(int, indices.split(' '))

    # project out selected ica components
    ica.apply(raw, exclude=indices)

raw.plot()

to_be_saved = raw_input('Do you want to save (y, n)? ')

if to_be_saved == 'y':
    raw.save(clean_file, overwrite=True)
