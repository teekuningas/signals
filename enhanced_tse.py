import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne
import math
from scipy import signal


FILES = {
    'KH009':
        {
            'med': '/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif',
            'rest': '/home/zairex/Code/cibr/demo/MI_KH009_EOEC-raw-pre.fif'
        },
    }

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


class PlottableRange(object):
    """
    """
    def __init__(self, filename, band, channel, title=''):
        pass


class PlottableTriggers(object):
    """
    """
    def __init__(self, filename):
        raw = mne.io.Raw(filename, preload=True)
        try: 
            self._times = mne.find_events(raw)[:, 0]
        except: 
            self._times = []

        self._sample_rate = raw.info['sfreq']

    @property
    def times(self):
        return self._times

    @property
    def sample_rate(self):
        return self._sample_rate


class PlottableCustomTriggers(object):
    """
    """
    def __init__(self, times, sample_rate):
        self._times = times
        self._sample_rate = sample_rate

    @property
    def times(self):
        return self._times

    @property
    def sample_rate(self):
        return self._sample_rate


class PlottableTSE(object):
    """
    """
    def __init__(self, filename, band, channel, color='Blue', title=''):

        raw = mne.io.Raw(filename, preload=True)
        data = raw._data[:128]
        sample_rate = raw.info['sfreq']

        self._title = title
        self._color = color
        self._sample_rate = sample_rate
        self._y = self._process_data(data, sample_rate, band, channel)
        self._x = np.arange(0, float(len(self._y))/sample_rate, 
                            1/float(sample_rate))

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

    @property
    def color(self):
        return self._color


class TSEPlot(object):

    def __init__(self, plottables, window):

        self.plottables = plottables
        self.window = window
        self.position = 0

        # max width is derived from the longest tse's length
        max_width = 0
        for plottable in plottables:
            tse_list = plottable['tse']
            for tse in tse_list:
                if len(tse.x) > max_width:
                    max_width = len(tse.x)
        self.plot_width = max_width

        self.fig = plt.figure()
        key_release_cid = self.fig.canvas.mpl_connect('key_release_event', 
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
            if self.position + 2*self.window < self.plot_width:
                self.position = self.position + self.window
            else:
                self.position = self.plot_width - self.window - 1
        
        self.plot_window(self.position, self.position + self.window)

    def plot_window(self, start, end):

        self.fig.clear()

        for idx, plottable in enumerate(self.plottables):
            orig_ax = plt.subplot(len(self.plottables), 1, idx+1)
            orig_ax.set_title(plottable.get('title', ''))
            orig_ax.set_xlabel('seconds')
 
            # do tse's!
            for j, tse in enumerate(plottable.get('tse', ())):
                if j == 0:
                    ax = orig_ax
                else:
                    ax = orig_ax.twinx()
                if j > 1:
                    ax.spines['right'].set_position(('axes', 1.0 + (j-1)*0.1))
                    self.fig.subplots_adjust(right=0.85 - 0.1*(j-2))
                    ax.set_frame_on(True)
                    ax.patch.set_visible(False)
                temp_x = tse.x[start:end]
                temp_y = tse.y[start:end]
                ax.plot(temp_x, temp_y, color=tse.color)
                ax.set_ylim([tse.y.min(), tse.y.max()])
                ax.set_xlim([tse.x[start], tse.x[end]])
                ax.set_ylabel(tse.title, color=tse.color)
                ax.tick_params(axis='y', colors=tse.color)
                ax.ticklabel_format(style='plain')

            # do ranges!
            # ...

            # do triggers!
            for trigger in plottable['trigger']:
                for time in trigger.times:
                    # this comparison is in samples
                    if time >= start and time < end:
                        # this x is in seconds
                        patch_x = float(time) / trigger.sample_rate
                        patch_width = 0.2
                        orig_ax.add_patch(
                            patches.Rectangle((patch_x - patch_width/2, 0), 
                                              patch_width, 0.01)
                        )
        plt.draw()   



if __name__ == '__main__':
    plottables = [
        {
            'title': 'alpha Oz',
            'tse': [
                PlottableTSE(FILES['KH009']['med'], BANDS['alpha'], CHANNELS['Oz'], color='Blue', title='alpha'),  # noqa
                PlottableTSE(FILES['KH009']['med'], BANDS['theta'], CHANNELS['Oz'], color='Red', title='theta'),  # noqa
                PlottableTSE(FILES['KH009']['med'], BANDS['beta'], CHANNELS['Oz'], color='Green', title='beta'),  # noqa
            ],
            'range': [
                PlottableRange(FILES['KH009']['med'], BANDS['alpha'], CHANNELS['Oz'], title='rest alpha range')  # noqa
            ],
            'trigger': [
                PlottableTriggers(FILES['KH009']['med'])
            ]
        },
        {
            'title': 'alpha Fz',
            'tse': [
                PlottableTSE(FILES['KH009']['med'], BANDS['alpha'], CHANNELS['Fz'], title='alpha'),  # noqa
            ],
            'range': [
                PlottableRange(FILES['KH009']['med'], BANDS['alpha'], CHANNELS['Fz'], title='rest alpha range')  # noqa
            ],
            'trigger': [
                PlottableTriggers(FILES['KH009']['med'])
            ]
        },
    ]
    stft = TSEPlot(plottables, window=32000)
