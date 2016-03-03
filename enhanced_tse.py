import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne
import math
from scipy import signal


FILES = {
    # experienced
    'KH009':
        {
            'med': '/home/zairex/Code/cibr/data/gradudemo/KH009_MED-pre.fif',
            'rest': '/home/zairex/Code/cibr/data/gradudemo/KH009_EOEC-pre.fif'
        },
    # experienced
    'KH005':
        {
            'med': '/home/zairex/Code/cibr/data/gradudemo/KH005_MED-pre.fif',
            'rest': '/home/zairex/Code/cibr/data/gradudemo/KH005_EOEC-pre.fif'
        },
    # experienced
    'KH007':
        {
            'med': '/home/zairex/Code/cibr/data/gradudemo/KH007_MED-pre.fif',
            'rest': '/home/zairex/Code/cibr/data/gradudemo/KH007_EOEC-pre.fif'
        },
    # not experienced
    'KH013':
        {
            'med': '/home/zairex/Code/cibr/data/gradudemo/KH013_MED-pre.fif',
            'rest': '/home/zairex/Code/cibr/data/gradudemo/KH013_EOEC-pre.fif'
        },
    # not experienced
    'KH028':
        {
            'med': '/home/zairex/Code/cibr/data/gradudemo/KH028_MED-pre.fif',
            'rest': '/home/zairex/Code/cibr/data/gradudemo/KH028_EOEC-pre.fif'
        },
    }

CHANNELS = {
    'Fz': 11, # middle front 
    'Oz': 75, # middle back
    'T3': 108, # middle right
    'T4': 45, # middle left
}

BANDS = {
    'gamma': (20, 40),
    'alpha': (8, 12),
    'beta': (12, 20),
    'theta': (4, 8),
    'delta': (1, 4),
}

BUTTER_ORDER = 4


def _create_tse_data(data, sample_rate, band, channel):

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


class PlottableRange(object):
    """
    """
    def __init__(self, filename, band, channel, intervals=[], color='Blue', title=''):  # noqa

        raw = mne.io.Raw(filename, preload=True)
        data = raw._data[:128]
        sample_rate = raw.info['sfreq']

        self._title = title
        self._color = color

        data = _create_tse_data(data, sample_rate, band, channel)

        # if not specified intervals, use the whole data
        if not intervals:
            intervals = [(0, len(data))]

        max_value = None
        min_value = None
        average_value = 0
        element_count = 0
        for x0, x1 in intervals:
            for x in range(x0, x1):
                if max_value is None or data[x] > max_value:
                    max_value = data[x]
                if min_value is None or data[x] < min_value:
                    min_value = data[x]
                average_value += data[x]
                element_count += 1

        self._low_limit = min_value
        self._high_limit = max_value
        self._average = float(average_value) / element_count

    @property
    def average(self):
        return self._average

    @property
    def low_limit(self):
        return self._low_limit

    @property
    def high_limit(self):
        return self._high_limit

    @property
    def title(self):
        return self._title

    @property
    def color(self):
        return self._color


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
    def __init__(self, filename, band, channel, color='Blue', title='', ranges=[]):  # noqa

        raw = mne.io.Raw(filename, preload=True)
        data = raw._data[:128]
        sample_rate = raw.info['sfreq']

        self._title = title
        self._color = color
        self._sample_rate = sample_rate
        self._ranges = ranges
        self._y = _create_tse_data(data, sample_rate, band, channel)
        self._x = np.arange(0, float(len(self._y))/sample_rate, 
                            1/float(sample_rate))

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

    @property
    def ranges(self):
        return self._ranges



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
                ax.set_xlim([tse.x[start], tse.x[end]])
                ax.set_ylabel(tse.title, color=tse.color)
                ax.tick_params(axis='y', colors=tse.color)
                ax.ticklabel_format(style='plain')

                # do ranges!
                max_ = None
                min_ = None
                for range_ in tse.ranges:
                    high = range_.high_limit
                    low = range_.low_limit

                    if max_ is None or high > max_:
                        max_ = high
                    if min_ is None or low < min_:
                         min_ = low

                    average = range_.average
                    color = range_.color
                    height = high - low

                    # range
                    rectangle = patches.Rectangle((tse.x[start], low), tse.x[end], high - low)
                    rectangle.set_alpha(0.20)
                    rectangle.set_color(color)
                    ax.add_patch(rectangle)

                    # average
                    rectangle = patches.Rectangle((tse.x[start], average - height/50.0), tse.x[end], 2*height/50.0)
                    rectangle.set_alpha(0.4)
                    rectangle.set_color(color)
                    ax.add_patch(rectangle)

                if max_:
                    max_ = max(max_, tse.y.max())
                else:
                    max_ = tse.y.max()
                if min_:
                    min_ = min(min_, tse.y.min())
                else:
                    min_ = tse.y.min()

                ax.set_ylim([min_ - abs(min_*0.1), max_ + abs(max_*0.1)])

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

    def get_baseline_intervals(filename):
        raw = mne.io.Raw(filename, preload=True)
        sfreq = int(raw.info['sfreq'])
        try: 
            events = mne.find_events(raw)[:, 0]
        except: 
            events = []

        intervals = []

        for idx in range(len(events) - 1):
            # lets take all intervals that are at least 10 seconds in length
            # and at least 20 seconds away from events
            if events[idx+1] - events[idx] >= 50*sfreq:
                intervals.append((events[idx] + 20*sfreq, events[idx+1] - 20*sfreq))  # noqa

        return intervals

    subject = 'KH009'
    plottables = [
        {
            'title': 'many bands at Oz',
            'tse': [
                PlottableTSE(FILES[subject]['med'], BANDS['alpha'], CHANNELS['Oz'], color='Blue', title='alpha'),  # noqa
                PlottableTSE(FILES[subject]['med'], BANDS['theta'], CHANNELS['Oz'], color='Red', title='theta'),  # noqa
                PlottableTSE(FILES[subject]['med'], BANDS['beta'], CHANNELS['Oz'], color='Green', title='beta'),  # noqa
            ],
            'trigger': [
                PlottableTriggers(FILES[subject]['med'])
            ]
        },

        {
            'title': 'alpha Oz with ranges',
            'tse': [
                PlottableTSE(FILES[subject]['med'], BANDS['alpha'], CHANNELS['Oz'], title='alpha',  # noqa
                    ranges=[
                        # PlottableRange(FILES[subject]['rest'], BANDS['alpha'], CHANNELS['Oz'], intervals=[(5000, 80000)], title='rest alpha range'),  # noqa
                        PlottableRange(FILES[subject]['med'], BANDS['alpha'], CHANNELS['Oz'], intervals=get_baseline_intervals(FILES[subject]['med']), title='meditation alpha range'),  # noqa
                    ]
                ), 
            ],
            'trigger': [
                PlottableTriggers(FILES[subject]['med'])
            ]
        },
    ]
    stft = TSEPlot(plottables, window=32000)
