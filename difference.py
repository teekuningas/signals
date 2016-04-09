# Plots listed raw files on top of each other
# Usage:
#     python difference.py /path/to/first/file /path/to/second/file ...
import mne
import sys

from lib.difference import DifferencePlot


def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def main(paths):

    time_series_size = raw_input('Please enter how many samples are shown in the time series plot window (default 10000): ')
    try:
        time_series_size = int(time_series_size)
    except ValueError:
        time_series_size = 10000

    power_series_size = raw_input('Please enter size of the power plot window (default 50): ')
    try:
        power_series_size = int(power_series_size)
    except ValueError:
        power_series_size = 50

    raw_objects = [read_raw(path) for path in paths]
    
    # plot time series
    ch_names = raw_objects[0].info['ch_names']
    datasets = [raw._data for raw in raw_objects]
    difference_plot = DifferencePlot(datasets, ch_names=ch_names, window_width=time_series_size, window_height=5)

    # plot power spectrum
    psds = [mne.time_frequency.psd.compute_raw_psd(raw) for raw in raw_objects]
    x_range = psds[0][1]
    datasets = [psd[0] for psd in psds]
    difference_plot = DifferencePlot(datasets, ch_names=ch_names, x_range=x_range,
                                     window_width=power_series_size, window_height=5)
    

if __name__ == '__main__':
    cla = sys.argv
    main(cla[1:])

