import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne
import math
from scipy import signal


BUTTER_ORDER = 4


def create_tse_data(data, sfreq, band):
    """
    """

    # band-pass filter
    l_cutoff = 1.0 * band[0] / (sfreq / 2)
    h_cutoff = 1.0 * band[1] / (sfreq / 2)
    b, a = signal.butter(BUTTER_ORDER, [l_cutoff, h_cutoff], btype='band')

    filtered = signal.filtfilt(b, a, data)
    if math.isnan(filtered[0]):
        raise Exception("Please adjust butterworth " 
                        "order or frequency band")

    # take envelope
    env = np.abs(signal.hilbert(filtered))
    return env


class PlottableTriggers(object):
    """
    """
    def __init__(self, times):
        self._times = times

    @property
    def times(self):
        return self._times


class PlottableTSE(object):
    """
    """
    def __init__(self, data, color='Blue', title=''):

        self._title = title
        self._color = color
        self._y = data
        self._x = np.arange(0, len(self._y))

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
            orig_ax.set_xlabel('samples')

            grand_high = None
            grand_low = None

            # do tse's!
            for j, tse in enumerate(plottable.get('tse', ())):

                if j == 0 or plottable.get('unix'):
                    ax = orig_ax
                else:
                    ax = orig_ax.twinx()

                if j > 1 and not plottable.get('unix'):
                    ax.spines['right'].set_position(('axes', 1.0 + (j-1)*0.1))
                    self.fig.subplots_adjust(right=0.85 - 0.1*(j-2))
                    ax.set_frame_on(True)
                    ax.patch.set_visible(False)

                temp_x = tse.x[start:end]
                temp_y = tse.y[start:end]
                ax.plot(temp_x, temp_y, color=tse.color)
                ax.set_xlim([tse.x[start], tse.x[end - 1]])
                ax.set_ylabel(tse.title, color=tse.color)
                ax.tick_params(axis='y', colors=tse.color)
                ax.ticklabel_format(style='plain')

                max_ = tse.y.max()
                min_ = tse.y.min()

                if not plottable.get('unix'):
                    ax.set_ylim([min_ - abs(min_*0.1), max_ + abs(max_*0.1)])
                else:
                    if not grand_high or max_ > grand_high:
                        grand_high = max_
                    if not grand_low or min_ < grand_low:
                        grand_low = min_

            if plottable.get('unix'):
                ax.set_ylim(grand_low, grand_high)

            # do triggers!
            for trigger in plottable['trigger']:
                for time in trigger.times:
                    
                    # this comparison is in samples
                    if time >= start and time < end:
                        patch_x = time
                        patch_width = 0.2
                        orig_ax.add_patch(
                            patches.Rectangle((patch_x - patch_width/2, 0), 
                                              patch_width, 0.01)
                        )
        plt.draw()   

