# Reads EGI or FIF and plots it 
# Usage:
#     python plot_raw_data.py /path/to/source/file
import mne
import sys


def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def main(filename):
    raw = read_raw(filename)
    raw.filter(l_freq=1, h_freq=40)
    raw.plot()
    wait_for_input = raw_input('Enter anything to quit.')


if __name__ == '__main__':
    cla = sys.argv
    main(cla[1])
