import mne
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class ComponentPlot(object):

    def __init__(self, source_stft, freqs, triggers, current_row, 
                 rows, info, raw_length, window=15, title=""):
        self.fig = plt.figure()
        self.position = 0
        self.window = window
        self.current_row = current_row
        self.rows = rows
        self.source_stft = source_stft
        self.info = info
        self.raw_length = raw_length
        self.freqs = freqs
        self.title=title
        self.triggers = triggers
        
        self.update_tfr_plot()
        self.cid = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def update_tfr_plot(self):

        position = self.position
        window = self.window
        rows = self.rows
        current_row = self.current_row
        source_stft = self.source_stft
        info = self.info
        sfreq = info['sfreq']
        raw_length = self.raw_length
        freqs = self.freqs
        triggers = self.triggers

        self.fig.clear()
        for i in range(rows):

            length_samples = source_stft.shape[2]
            length_s = raw_length / sfreq
            scale = length_s / float(length_samples)

            # find out times
            times = np.arange(window * position, 
                              window * (position + 1), 
                              1.0) * scale

            data = np.power(np.abs(source_stft), 2)
            # data = np.abs(source_stft)
            # data = 10 * np.log10(np.abs(source_stft))

            # select subset of data
            data = data[:, :, window*position:window*(position+1)]
            
            # pad if necessary
            data = np.pad(data, ((0,0), (0,0), (0, window - data.shape[2])), 
                          constant_values=0, mode='constant')

            # use mne's AverageTFR for plotting
            tfr_ = mne.time_frequency.AverageTFR(info, data, times, freqs, 1)
            axes = self.fig.add_subplot(rows, 1, i + 1)

            title = str(current_row) + ' - ' + str(current_row+rows)
            if self.title:
                title = self.title

            tfr_.plot(picks=[current_row + i], axes=axes, show=False, 
                      title=title) 
            axes.set(xlabel="Time (ms)", ylabel="Frequency (Hz)")

            # add triggers
            for trigger in triggers:
                x0 = window*position*scale*sfreq
                x1 = window*(position+1)*scale*sfreq
                if trigger >= x1 or trigger < x0:
                    continue

                width = int(float(axes.get_xlim()[1] - axes.get_xlim()[0]) / 100)
                axes.add_patch(
                    patches.Rectangle((trigger - width/2, 0.0), width, 1000)
                )

        self.fig.canvas.draw()

    def on_key_release(self, event):
        position = self.position
        window = self.window
        source_stft = self.source_stft
        if event.key == 'left' and position != 0:
            position -= 1
        elif event.key == 'right' and position < source_stft.shape[2] / window:
            position += 1

        self.position = position
        self.update_tfr_plot()

# show everything
plt.show()
