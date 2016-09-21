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
        raw = mne.io.read_raw_egi(filename, preload=True)
        # assume egi
        if filename.endswith('.001'):
            stem = filename.split('.001')[0]
            idx = 2
            while True:
                fname = stem + "." + '%03d' % idx
                try:
                    raw.append(mne.io.read_raw_egi(fname, preload=True))
                except:
                    break
                idx += 1
    return raw

def main(from_file, to_file):

    layout_fname = 'gsn_129.lout'
    layout_path = '/home/zairex/Code/cibr/materials/'
    layout = mne.channels.read_layout(layout_fname, layout_path)

    raw = read_raw(from_file)

    raw.filter(l_freq=1, h_freq=100)

    ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    ica.fit(raw)

    sources = ica.get_sources(raw)

    # alter amplitudes to get better plot 
    for source in sources._data:
        for idx, amplitude in enumerate(source):
            source[idx] = amplitude / 5000.0

    ica.plot_components(layout=layout, show=False)
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
