# Allows to project ICA components out of raw data
# Usage:
#     python clean_artifacts.py /path/to/dirty/file /path/to/clean/file
import matplotlib.pyplot as plt
import mne
import numpy as np
import sys


def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def get_ica_path(filename):
    last_part = filename.split('/')[-1]
    without_extension = ''.join(last_part.split('.')[:-1])
    with_new_extension = '.' + without_extension + '-ica.fif'
    final = '/'.join(filename.split('/')[:-1] + [with_new_extension])
    return final


def main(from_file, to_file):

    temp_ica = get_ica_path(to_file)
    raw = read_raw(from_file)

    raw.filter(l_freq=1, h_freq=80)

    raw.plot()

    use_saved = raw_input('Use saved ica solution if found (y, n): ')

    ica = None
    if use_saved == 'y': 
        try:
            ica = mne.preprocessing.ica.read_ica(temp_ica)
        except IOError:
            print "Saved solution not found"
    if not ica:
        ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
        ica.fit(raw)
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
        raw.save(to_file, overwrite=True)


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1], cla[2])
