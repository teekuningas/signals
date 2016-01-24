import numpy as np
import matplotlib.pyplot as plt
import mne

raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/meditaatio/KH004_MED-raw.fif', preload=True)
# raw = mne.io.Raw('/home/zairex/Code/cibr/data/graduaineisto/EOEC/KH004_EOEC-raw.fif', preload=True)

data = raw._data
info = raw.info
sfreq = info['sfreq']
wsize = int(sfreq*4)
tstep = int(wsize/2)
freq_limit = 20
channels = [
    16, # middle front 
    75, # middle back
    114, # middle right
    44, # middle left
]


class STFTPlot():

    def __init__(self):
        self.tfr = mne.time_frequency.stft(data, wsize, tstep)
        self.x = np.arange(0, self.tfr.shape[2]*tstep, tstep) / sfreq
        self.y = mne.time_frequency.stftfreq(wsize, sfreq)

        # find index for frequency limit
        self.freq_idx = int(self.y.shape[0] / ((sfreq / 2) / freq_limit))

        fig, self.axarray = plt.subplots(2,2)
        key_release_cid = fig.canvas.mpl_connect('key_release_event', 
                                                 self.on_key_release)
        self.plot_window(0, len(self.x))
        plt.show()

    def on_key_release(self, event):
        pass

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
