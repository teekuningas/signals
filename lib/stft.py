import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mne

class STFTPlot():

    def __init__(self, freqs, tfr, window_width=50, window_height=3, 
                 ch_names=None, ignore_times=None, triggers=[]):
        self.tfr = tfr
        self.window_width = window_width
        self.window_height = window_height
        self.x = 0
        self.y = 0
        self.ch_names = ch_names
        self.triggers = triggers
        self.ignore_times = ignore_times

        # pad with average
        self.residue = tfr.shape[2] % window_width or window_width
        
        average_ = np.average(tfr.flatten())
        fill_matrix = np.empty((tfr.shape[0], tfr.shape[1], 
                                window_width - self.residue), 
                               dtype=tfr.dtype)
        fill_matrix.fill(average_)
        self.tfr = np.concatenate([tfr, fill_matrix], axis=2)

        self.y_values = freqs
        self.x_values = np.arange(self.tfr.shape[2])

        self.fig = plt.figure()
        self.plot_window()
        key_release_cid = self.fig.canvas.mpl_connect('key_release_event', 
                                                      self.on_key_release)

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

    def plot_window(self):
        self.fig.clear()

        for idx in range(self.window_height):
            real_idx = (self.y + idx) % self.tfr.shape[0]
            width = self.window_width
            height = self.window_height
            start = (self.x * width) % self.tfr.shape[2]
            end = start + width

            temp_x = self.x_values[start:end]
            temp_y = self.y_values
            temp_z = np.abs(self.tfr[real_idx][:, start:end])

            ax = self.fig.add_subplot(height, 1, idx + 1)

            if self.ch_names:
                ax.set_title(str(self.ch_names[real_idx]))

            tfr_ = self.tfr[real_idx, :, :self.residue]

            if self.ignore_times:
                tfr_ = np.delete(tfr_, self.ignore_times, axis=1)
                
            # find min and max values for colors
            vvalues = 10 * np.log10(np.abs(tfr_.flatten()))
            vmax = max(vvalues)
            # vmin = np.average(vvalues)
            # try to get nice compromise for coloring
            vmin = (1/5.0)*min(vvalues) + (4/5.0)*np.average(vvalues)

            ax.imshow(10 * np.log10(temp_z), vmin=vmin, vmax=vmax,
                      extent=[temp_x.min(), temp_x.max(), 
                              temp_y.min(), temp_y.max()],
                      interpolation='nearest', origin='lower',
                      cmap='OrRd',)

            ax.axis('tight')

            for trigger in self.triggers:
                patch_width = 0.2
                ax.add_patch(patches.Rectangle((trigger - patch_width/2, 0),
                                               patch_width, 10))


        plt.draw()


if __name__ == '__main__':
    stft = STFTPlot()
