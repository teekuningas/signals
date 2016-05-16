import sys

import mne
import matplotlib.pyplot as plt

import numpy as np

fnames = sys.argv[1:]

if not fnames:
    raise ValueError('Give filenames as parameters')

raws = []
for fname in fnames:
    raws.append(mne.io.Raw(fname, preload=True))

for raw in raws:
    title = raw.info['filename'].split('/')[-1]
    psd, freqs = mne.time_frequency.psd.psd_welch(raw, fmin=1, fmax=40, 
                                                  n_fft=1024)
    psd_ave = np.mean(psd, axis=0)
    plt.plot(freqs, psd_ave, label=title) 

plt.legend()
plt.show()

