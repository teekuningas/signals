# Plots listed raw files on top of each other
# Usage:
#     python difference.py /path/to/first/file /path/to/second/file ...
import numpy as np
import mne
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

class DifferencePlot:
    """ Plot EEG datasets on top of each other
    """

    def __init__(self, datasets, ch_names=None, x=0, y=0, window_width=2500, window_height=3):
        if len(set([dataset.shape for dataset in datasets])) != 1:
            raise Exception("Dataset shapes must be identical")

        self.window_width = window_width
        self.window_height = window_height
        self.ch_names = ch_names
        self.x = x
        self.y = y

        # pad with zeros
        padded_datasets = []
        for dataset in datasets:
            residue = dataset.shape[1] % window_width
            zero_matrix = np.zeros((dataset.shape[0], window_width - residue))
            padded_datasets.append(np.concatenate([dataset, zero_matrix], axis=1))
        self.datasets = padded_datasets

        self.figure = plt.figure()
        self.plot_window()
        self.key_release_cid = self.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

        plt.show()

    def plot_window(self):
        self.figure.clear()
        for channel in range(self.window_height):
            real_channel = (self.y + channel) % (self.datasets[0].shape[0])
            width = self.window_width
            left = (self.x * width) % self.datasets[0].shape[1]

            ax = self.figure.add_subplot(self.window_height, 1, channel + 1)
            ax.set_xlim([left, left + width])

            if self.ch_names:
                ax.set_title(self.ch_names[real_channel])

            for dataset in self.datasets:
                data = dataset[real_channel]
                ax.plot(range(left, left + width), data[left:left+width])

        plt.draw()


    def on_key_release(self, event):
        if event.key == 'left':
            self.x = self.x - 1
        elif event.key == 'right':
            self.x = self.x + 1
        elif event.key == 'up':
            self.y = self.y - 1
        elif event.key == 'down':
            self.y = self.y + 1

        self.plot_window()


def read_raw(filename):
    if filename.endswith('.fif'):
        raw = mne.io.Raw(filename, preload=True)
    else:
        # assume egi
        raw = mne.io.read_raw_egi(filename)
    return raw


def main(paths):
    window_size = raw_input('Please enter how many samples are shown in a plot window (default 10000): ')
    try:
        window_width = int(window_size)
    except ValueError:
        window_width = 10000
    raw_objects = [read_raw(path) for path in paths]
    ch_names = raw_objects[0].info['ch_names']
    datasets = [raw._data for raw in raw_objects]
    difference_plot = DifferencePlot(datasets, ch_names=ch_names, window_width=window_width, window_height=5)
    

if __name__ == '__main__':
    cla = sys.argv
    main(cla[1:])
