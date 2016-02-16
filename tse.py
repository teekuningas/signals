import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne
import math
from scipy import signal

raw = mne.io.Raw('/home/zairex/Code/cibr/demo/MI_KH009_MED-bads-raw-pre.fif', preload=True)
# raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH004_EOEC-raw.fif', preload=True)

data = raw._data[:128]
band = [8, 12]
butter_order = 4
sample_rate = raw.info['sfreq']
channels = [
    11, # middle front 
    75, # middle back
    108, # middle right
    45, # middle left
]


class TSEPlot():

    def __init__(self):
        l_cutoff = 1.0 * band[0] / (sample_rate / 2)
        h_cutoff = 1.0 * band[1] / (sample_rate / 2)
	b, a = signal.butter(butter_order, [l_cutoff, h_cutoff], btype='band')
        filtered_data = [] 
        for channel in channels:
            filtered = signal.filtfilt(b, a, data[channel-1])
            if math.isnan(filtered[0]):
                raise Exception("Please adjust butterworth " 
                                "order or frequency band")
            filtered_data.append(filtered)

        absolute = np.absolute(filtered_data)

        # running mean
        N = 5000
        averaged = np.zeros((4, absolute.shape[1] - N + 1))
        for i in range(absolute.shape[0]):
            averaged[i] = np.convolve(absolute[i], np.ones((N,))/N, mode='valid')

        self.tse_data = averaged

        self.x = np.arange(0, float(self.tse_data.shape[1])/sample_rate, 1/sample_rate)

        self.window = 80000
        self.position = 0

        try: 
            self.triggers = mne.find_events(raw)[:, 0]
        except: 
            self.triggers = []

        fig, self.axarray = plt.subplots(2,2)
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
            if self.position + 2*self.window < len(self.x):
                self.position = self.position + self.window
            else:
                self.position = len(self.x) - self.window - 1
        
        self.plot_window(self.position, self.position + self.window)

    def plot_window(self, start, end):
        for i in range(self.axarray.shape[0]):
            for j in range(self.axarray.shape[1]):
                self.axarray[i, j].clear()

        max_value = np.amax(self.tse_data)

        for idx, channel in enumerate(channels):
            temp_x = self.x[start:end]
            temp_y = self.tse_data[idx][start:end]

            ax = self.axarray[idx%2, idx/2]
            ax.set_title(str(channel))
            ax.plot(temp_x, temp_y)
            ax.set_ylim([0, max_value])
            ax.set_xlim([self.x[start], self.x[end]])

            for trigger in self.triggers:
                patch_x = float((trigger - self.x[start])) / sample_rate
                ax.add_patch(
                    patches.Rectangle((patch_x, 0), 0.2, max_value/2))

        plt.draw()


if __name__ == '__main__':
    stft = TSEPlot()
