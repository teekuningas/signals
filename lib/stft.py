import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne

class STFTPlot():

    def __init__(self, tfr, wsize, sfreq, tstep, freq_limit, ch_names=None):
        self.tfr = tfr
        self.x = np.arange(0, self.tfr.shape[2]*tstep, tstep) / sfreq
        self.y = mne.time_frequency.stftfreq(wsize, sfreq)
        self.plot_window = 100
        self.position = 0
        self.ch_names = ch_names

        # find index for frequency limit
        self.freq_idx = int(self.y.shape[0] / ((sfreq / 2) / freq_limit))

        try:
            self.triggers = mne.find_events(raw)[:, 0]
        except:
            self.triggers = []

        fig, self.axarray = plt.subplots(2,2)
        key_release_cid = fig.canvas.mpl_connect('key_release_event', 
                                                 self.on_key_release)
        self.plot_tfr(self.position, self.plot_window)
        # self.plot_window(0, len(self.x))
        plt.show()

    def on_key_release(self, event):
        if event.key == 'left':
            if self.position - self.plot_window >= 0:
                self.position = self.position - self.plot_window
            else:
                self.position = 0
        if event.key == 'right':
            if self.position + 2*self.plot_window < len(self.x):
                self.position = self.position + self.plot_window
            else:
                self.position = len(self.x) - self.plot_window - 1
        
        self.plot_tfr(self.position, self.position + self.plot_window)

    def plot_tfr(self, start, end):
        for i in range(self.axarray.shape[0]):
            for j in range(self.axarray.shape[1]):
                self.axarray[i, j].clear()

        for idx in range(self.tfr.shape[0]):
            temp_x = self.x[start:end]
            temp_y = self.y[:self.freq_idx]
            temp_z = self.tfr[idx][:self.freq_idx, start:end]

            ax = self.axarray[idx%2, idx/2]
            if self.ch_names:
                ax.set_title(str(self.ch_names[idx]))
            ax.pcolormesh(temp_x, temp_y, 10 * np.log10(temp_z))
            ax.axis('tight')

            for trigger in self.triggers:
                patch_x = float((trigger - self.x[start])) / sfreq
                ax.add_patch(
                    patches.Rectangle((patch_x, 0), 0.3, freq_limit/4))

        plt.draw()


if __name__ == '__main__':
    stft = STFTPlot()
