import numpy as np
import matplotlib.pyplot as plt
import mne

raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/meditaatio/KH004_MED-raw.fif', preload=True)
# raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH004_EOEC-raw.fif', preload=True)

data = raw._data
info = raw.info
sfreq = info['sfreq']
wsize = int(sfreq*2)
tstep = int(wsize/4)
freq_limit = 100
channels = [
    11, # middle front 
    75, # middle back
    108, # middle right
    45, # middle left
]


class STFTPlot():

    def __init__(self):
        self.tfr = mne.time_frequency.stft(data, wsize, tstep)
        self.x = np.arange(0, self.tfr.shape[2]*tstep, tstep) / sfreq
        self.y = mne.time_frequency.stftfreq(wsize, sfreq)
        self.window = 50
        self.position = 0

        # find index for frequency limit
        self.freq_idx = int(self.y.shape[0] / ((sfreq / 2) / freq_limit))

        fig, self.axarray = plt.subplots(2,2)
        key_release_cid = fig.canvas.mpl_connect('key_release_event', 
                                                 self.on_key_release)
        self.plot_window(self.position, self.window)
        # self.plot_window(0, len(self.x))
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

        for idx, channel in enumerate(channels):
            temp_x = self.x[start:end]
            temp_y = self.y[:self.freq_idx]
            temp_z = self.tfr[channel-1][:self.freq_idx, start:end]

            ax = self.axarray[idx%2, idx/2]
            ax.pcolormesh(temp_x, temp_y, 10 * np.log10(temp_z), 
                          shading='gouraud')
            ax.axis('tight')

        plt.draw()


if __name__ == '__main__':
    stft = STFTPlot()
