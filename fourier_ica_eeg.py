import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA

from lib.difference import DifferencePlot
from lib.stft import STFTPlot

if __name__ == '__main__':

    raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH005_MED-pre.fif',
                     preload=True)

    wsize = 1000

    fica = FourierICA(wsize=wsize, n_pca_components=40, 
                      n_ica_components=20,
                      sfreq=1000, hpass=4, lpass=15)
    fica.fit(raw._data[:128, raw.first_samp:raw.last_samp])

    freqs, source_stft = fica.source_stft

    # data = np.abs(fica._concat(source_stft))
    # components = DifferencePlot([data],
    #                             ch_names=[str(i) for i in range(len(data))],
    #                             window_width=1000, window_height=5)

    data = np.abs(source_stft)

    try:
        triggers = mne.find_events(raw)[:, 0] - raw.first_samp
        triggers = triggers.astype(np.float64) / (wsize/2)
        triggers = triggers[np.where(triggers > 1)[0]]
    except:
        triggers = []

    # ignore_times = range(480, 510)
    ignore_times = []

    STFTPlot(freqs, source_stft, window_width=50, triggers=triggers,
             ch_names=[str(i+1) for i in range(len(source_stft))],
             ignore_times=ignore_times)
