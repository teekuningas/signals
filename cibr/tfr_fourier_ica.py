# Usage:
# ipython tfr_fourier_ica.py example_tfr.h5

import os
import sys

import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import tfr_morlet
from mne.time_frequency.tfr import read_tfrs

from lib.fourier_ica import FourierICA

SAVE_PATH = 'output/tfr_fourier_ica/'
N_COMPONENTS = 3

tfrs = read_tfrs(sys.argv[-1])

ficas = []
for tfr in tfrs:
    print "Handling tfr with condition " + tfr.comment
    while True:
        try:
            fica = FourierICA(tfr.data, tfr.freqs, n_components=N_COMPONENTS, 
                              maxiter=500, conveps=1e-7)
            fica.fit()
            break
        except KeyboardInterrupt:
            raise
        except:
            pass
    ficas.append(fica)


for fica in ficas:
    source_stft = fica.source_stft
    freq = fica.freqs
    fig_ = plt.figure()
    for i in range(source_stft.shape[0]):
        y = np.mean(20*np.log10(np.abs(source_stft[i])), axis=-1)
        x = fica.freqs
        axes = fig_.add_subplot(1, source_stft.shape[0], i+1)
        axes.plot(x,y)
    plt.show()

import pdb; pdb.set_trace()
print "kissa"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
