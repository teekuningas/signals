import mne
import matplotlib.pyplot as plt
import numpy as np

from lib.fourier_ica import FourierICA

from lib.difference import DifferencePlot

if __name__ == '__main__':

    raw = mne.io.Raw('/home/zairex/Code/cibr/data/gradudemo/KH005_EOEC-pre.fif',
                     preload=True)

    fica = FourierICA(wsize=2000, n_pca_components=20, 
                      n_ica_components=8,
                      sfreq=1000, hpass=5, lpass=20)
    fica.fit(raw._data)

    source_stft = fica.source_stft_components
    source_time = fica.source_time_components

    data = np.abs(fica._concat(source_stft))

    components = DifferencePlot([data],
                                ch_names=[str(i) for i in range(len(data))],
                                window_width=1000, window_height=2)
