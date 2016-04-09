import numpy as np
import mne
import matplotlib.pyplot as plt
import sys
from copy import deepcopy

class DifferencePlot:
    """ Plot EEG datasets on top of each other
    """

    def __init__(self, datasets, ch_names=None, x=0, y=0, x_range=None,
                 window_width=2500, window_height=3, scaletype=None):

        if len(set([dataset.shape for dataset in datasets])) != 1:
            raise Exception("Dataset shapes must be identical")

        self.window_width = window_width
        self.window_height = window_height
        self.ch_names = ch_names
        self.x = x
        self.y = y
        self.scaletype = scaletype

        # pad with zeros
        padded_datasets = []
        for dataset in datasets:
            residue = dataset.shape[1] % window_width
            zero_matrix = np.zeros((dataset.shape[0], window_width - residue))
            padded_datasets.append(np.concatenate([dataset, zero_matrix], axis=1))
        self.datasets = padded_datasets

        # interpolate x_range
        if x_range is not None:
            self.x_range = np.interp(range(self.datasets[0].shape[1]), range(len(x_range)), x_range)
        else:
            self.x_range = x_range

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

            max_y, min_y = None, None
            for dataset in self.datasets:
                data = dataset[real_channel]
                if not max_y or max(data) > max_y:
                    max_y = max(data)

                if not min_y or min(data) < min_y:
                    min_y = min(data)

            ax = self.figure.add_subplot(self.window_height, 1, channel + 1)

            if self.ch_names:
                ax.set_title(self.ch_names[real_channel])

            if self.x_range is not None:
                left = (self.x * width) % self.datasets[0].shape[1]
                left = (self.x * width) % self.datasets[0].shape[1]
                ax.set_xlim([self.x_range[left], self.x_range[left+width-1]])
                x_range = self.x_range[left : left+width]
            else:
                ax.set_xlim([left, left + width])
                x_range = range(left, left + width)

            if self.scaletype:
                ax.set_yscale(self.scaletype)
            
            ax.set_ylim([min_y, max_y])

            for dataset in self.datasets:
                data = dataset[real_channel]
                ax.plot(x_range, data[left:left+width])

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

