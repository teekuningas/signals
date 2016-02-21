import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne
import math
from scipy import signal


FILES = [
    {
        'meditation': '/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif',
        'resting': '/home/zairex/Code/cibr/demo/MI_KH009_EOEC-raw-pre.fif'
    },
]

CHANNELS = {
    'Fz': 11, # middle front 
    'Oz': 75, # middle back
    'T3': 108, # middle right
    'T4': 45, # middle left
}

BANDS = {
    'alpha': (8, 12),
    'beta': (12, 20),
    'theta': (4, 8),
    'delta': (1, 4),
}

BUTTER_ORDER = 4


class PlottableSubject(object):
    """
    """
    def __init__(self, files, band, channel, title=''):

        self._title = title

        med_raw = mne.io.Raw(files['meditation'], preload=True)
        rest_raw = mne.io.Raw(files['resting'], preload=True)
        
        med_data = med_raw._data[:128]
        rest_data = rest_raw._data[:128]

        med_sample_rate = med_raw.info['sfreq']
        rest_sample_rate = rest_raw.info['sfreq']

        self._sample_rate = med_sample_rate
        
        self._y = self._process_data(med_data, med_sample_rate, band, channel)

        self._x = np.arange(0, float(len(self._y))/med_sample_rate, 
                            1/float(med_sample_rate))

        try: 
            self._triggers = mne.find_events(med_raw)[:, 0]
        except: 
            self._triggers = []

    def _process_data(self, data, sample_rate, band, channel):

        # band-pass filter
        l_cutoff = 1.0 * band[0] / (sample_rate / 2)
        h_cutoff = 1.0 * band[1] / (sample_rate / 2)
        b, a = signal.butter(BUTTER_ORDER, [l_cutoff, h_cutoff], btype='band')

        filtered = signal.filtfilt(b, a, data[channel-1])
        if math.isnan(filtered[0]):
            raise Exception("Please adjust butterworth " 
                            "order or frequency band")

        # rectify
        absolute = np.absolute(filtered)

        # use running mean to smoothen the signal
        N = int(10 * sample_rate)
        averaged = np.zeros(len(absolute) - N + 1)
        averaged = np.convolve(absolute, np.ones((N,))/N, mode='valid')

        return averaged
       

    @property
    def triggers(self):
        return self._triggers

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def title(self):
        return self._title

    @property
    def sample_rate(self):
        return self._sample_rate


class TSEPlot(object):

    def __init__(self, subjects, window):

        self.subjects = subjects
        self.window = window
        self.position = 0

        if len(subjects) == 0:
            raise ValueError('Must have at least one plot')

        self.width = len(subjects[0].x)

        max_value = 0
        for y in [subject.y for subject in self.subjects]:
            temp_max = y.max()
            if temp_max > max_value:
                max_value = temp_max
        self.max_value = max_value

        fig, self.axarray = plt.subplots(len(subjects))

        key_release_cid = fig.canvas.mpl_connect('key_release_event', 
                                                 self.on_key_release)

        self.plot_window(self.position, self.window)

        plt.show()

    def on_key_release(self, event):
        if event.key == 'left':
            if self.position - self.window >= 0:
                self.position = self.position - self.window
            else:
                self.position = 0
        if event.key == 'right':
            if self.position + 2*self.window < self.width:
                self.position = self.position + self.window
            else:
                self.position = self.width - self.window - 1
        
        self.plot_window(self.position, self.position + self.window)

    def plot_window(self, start, end):

        for ax in self.axarray:
            ax.clear()

        for idx, subject in enumerate(self.subjects):
            temp_x = subject.x[start:end]
            temp_y = subject.y[start:end]
            
            ax = self.axarray[idx]

            ax.set_title(subject.title)
            ax.plot(temp_x, temp_y)
            ax.set_ylim([0, self.max_value])
            ax.set_xlim([subject.x[start], subject.x[end]])

            for trigger in subject.triggers:
                patch_x = float((trigger - subject.x[start])) / subject.sample_rate  # noqa
                ax.add_patch(
                    patches.Rectangle((patch_x, 0), 0.2, self.max_value/2))

            plt.draw()


if __name__ == '__main__':
    subjects = [
        PlottableSubject(FILES[0], BANDS['alpha'], CHANNELS['Oz'], title='alpha Oz'),
        PlottableSubject(FILES[0], BANDS['alpha'], CHANNELS['Fz'], title='alpha Fz'),
    ]
    stft = TSEPlot(subjects, window=32000)
